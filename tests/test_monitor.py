import json
import os
import socket
import sys
import types
from datetime import UTC, date, datetime, timedelta

import pytest

from harness import monitor
from harness.monitor import (
    IST,
    DayStats,
    Rule,
    aggregate,
    collect,
    detect_alerts,
    is_test_traffic,
    parse_record,
    render_markdown,
    trailing_baseline,
)
from harness.monitor_synth import N_DAYS, SPIKE_DAY, START, STEP_DAY, make_log, self_test


def rec(**kw):
    base = {
        "session_id": "web-1",
        "escalated": False,
        "output_flags": [],
        "injection_suspected": False,
        "tool_calls": [],
        "citations": [],
        "retrieved": [],
        "latency_ms": 1000,
    }
    base.update(kw)
    return base


def req(record, ts="2026-11-01T12:00:00+05:30"):
    return parse_record(record, datetime.fromisoformat(ts))


# ---- definitions ----


def test_hand_off_is_the_escalated_flag():
    assert req(rec(escalated=True)).handoff is True
    assert req(rec(escalated=False)).handoff is False
    assert (
        req(rec(text="I'm sorry, I can't answer that reliably")).handoff is False
    )  # text alone is not a hand-off


def test_guardrail_flag_is_output_flags_or_injection_suspected():
    assert req(rec(output_flags=["LARK100"])).flagged
    assert req(rec(injection_suspected=True)).flagged
    assert not req(rec(output_flags=[], injection_suspected=False)).flagged


def test_tool_error_is_a_result_with_an_error_key():
    calls = [
        {"name": "lookup_order", "args": {}, "result": {"error": "no such order"}},
        {"name": "check_stock", "args": {}, "result": {"stock": 4}},
        {"name": "start_return", "args": {}, "result": {"error": "outside window"}},
        {"name": "x", "args": {}, "result": "plain string with the word error"},
    ]
    r = req(rec(tool_calls=calls))
    assert r.tool_calls == 4 and r.tool_errors == 2


def test_latency_falls_back_to_usage_and_citations_and_articles_are_counted():
    r = req(
        {
            "session_id": "s",
            "usage": {"latency_ms": 777},
            "citations": ["KB-001", "KB-002"],
            "retrieved": [{"article_id": "KB-001"}, {"article_id": "KB-001"}, {"article_id": "KB-003"}],
        }
    )
    assert r.latency_ms == 777 and r.citations == 2 and r.articles == ("KB-001", "KB-003")


def test_day_boundaries_follow_the_configured_timezone():
    ts = datetime(2026, 11, 1, 19, 0, tzinfo=UTC)  # 00:30 IST on 2 November
    assert parse_record(rec(), ts, tz=IST).day == date(2026, 11, 2)
    assert parse_record(rec(), ts, tz=UTC).day == date(2026, 11, 1)


def test_record_without_any_timestamp_is_dropped_by_parse_record():
    assert parse_record(rec(), None) is None
    assert parse_record(rec(ts="2026-11-01T10:00:00+05:30"), None).day == date(2026, 11, 1)
    assert parse_record(rec(ts="not a time"), None) is None


def test_test_traffic_prefixes():
    for sid in ("eval-agent-A01-1", "sim-angry-01", "smoke", "smoke3-x", "warmup"):
        assert is_test_traffic(sid), sid
    for sid in ("web-1", "slack-U123", "retail-eval", "session-smoke"):
        assert not is_test_traffic(sid), sid


# ---- aggregation ----


def test_aggregate_per_day_numbers():
    reqs = [
        req(rec(escalated=True, latency_ms=4000, retrieved=[{"article_id": "KB-001"}])),
        req(
            rec(
                latency_ms=1000,
                citations=["KB-001"],
                retrieved=[{"article_id": "KB-001"}, {"article_id": "KB-002"}],
                tool_calls=[{"name": "a", "result": {"error": "x"}}, {"name": "b", "result": {"ok": 1}}],
            )
        ),
        req(rec(latency_ms=2000, citations=["KB-001", "KB-002"], output_flags=["c"])),
        req(rec(latency_ms=3000, citations=[]), ts="2026-11-02T10:00:00+05:30"),
    ]
    d1, d2 = aggregate(reqs)
    assert d1.day == date(2026, 11, 1) and d1.requests == 3
    assert d1.handoff_rate == pytest.approx(1 / 3) and d1.flag_rate == pytest.approx(1 / 3)
    assert d1.tool_calls == 2 and d1.tool_error_rate == 0.5
    assert d1.p50_ms == 2000 and d1.p95_ms == 4000
    assert d1.citations_per_answer == 1.5  # over the two non-hand-off requests: (1 + 2) / 2
    assert d1.top_articles[0] == ("KB-001", 2)
    assert d2.requests == 1 and d2.tool_error_rate is None and d2.handoff_rate == 0


def test_citations_per_answer_is_none_when_everything_was_a_hand_off():
    (d,) = aggregate([req(rec(escalated=True)), req(rec(escalated=True))])
    assert d.citations_per_answer is None and d.handoff_rate == 1.0


# ---- alert rules, one at a time ----


def day(i, n=400, handoff=0.08, flag=0.02, err=0.02, p50=10_000, p95=20_000, cites=1.8, calls=300):
    return DayStats(
        day=date(2026, 10, 1) + timedelta(days=i),
        requests=n,
        handoff_rate=handoff,
        flag_rate=flag,
        tool_calls=calls,
        tool_error_rate=err,
        p50_ms=p50,
        p95_ms=p95,
        citations_per_answer=cites,
        top_articles=[],
    )


def with_history(today: DayStats, k=7):
    hist = [day(i) for i in range(k)]
    today.day = hist[-1].day + timedelta(days=1)
    return [*hist, today]


@pytest.mark.parametrize(
    "kwargs,metric",
    [
        ({"handoff": 0.20}, "handoff_rate"),
        ({"flag": 0.08}, "flag_rate"),
        ({"err": 0.08}, "tool_error_rate"),
        ({"p50": 15_500}, "p50_ms"),
        ({"p95": 31_000}, "p95_ms"),
        ({"cites": 1.2}, "citations_per_answer"),
        ({"n": 900}, "requests"),
        ({"n": 190}, "requests"),
    ],
)
def test_each_rule_fires_past_its_threshold(kwargs, metric):
    got = {a.metric for a in detect_alerts(with_history(day(0, **kwargs)))}
    assert got == {metric}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"handoff": 0.17},
        {"flag": 0.065},
        {"err": 0.065},
        {"p50": 14_900},
        {"p95": 29_000},
        {"cites": 1.4},
        {"n": 700},
        {"n": 250},
    ],
)
def test_each_rule_stays_quiet_just_inside_its_threshold(kwargs):
    assert detect_alerts(with_history(day(0, **kwargs))) == []


def test_alert_needs_history_volume_and_calls():
    spike = day(0, handoff=0.5)
    assert detect_alerts(with_history(spike, k=2)) == []  # fewer than 3 prior days
    assert [a.metric for a in detect_alerts(with_history(day(0, handoff=0.5), k=3))] == ["handoff_rate"]
    thin_day = detect_alerts(with_history(day(0, n=9, handoff=0.9)))  # too few requests for any rate alert
    assert [a.metric for a in thin_day] == ["requests"]  # but a collapse in volume is exactly what must alert
    thin = with_history(day(0, err=0.5, calls=4))  # 4 tool calls is under the minimum of 5
    assert detect_alerts(thin) == []  # too few tool calls for a rate


def test_baseline_is_the_median_of_the_trailing_seven_calendar_days():
    days = [day(i, handoff=h) for i, h in enumerate([0.08, 0.08, 0.60, 0.08, 0.09, 0.07, 0.08])]
    days.append(day(7, handoff=0.35))
    base, used, needed = trailing_baseline(days, 7, "handoff_rate")
    assert used == 7 and base == 0.08  # one terrible day does not move the median
    # day 3 (0.60) had only 2 prior days, so it was quiet; day 8 alerts against the median baseline
    assert [(a.day, a.metric) for a in detect_alerts(days)] == [(days[7].day, "handoff_rate")]


def test_days_older_than_a_week_and_low_volume_days_leave_the_baseline():
    days = [day(0, handoff=0.9), day(1), day(2), day(3), day(12)]
    base, used, needed = trailing_baseline(
        days, 4, "handoff_rate"
    )  # day 12: only days 5..11 count, none exist
    assert base is None and used == 0 and needed == 3
    days = [day(0, n=5, handoff=0.9), day(1), day(2), day(3), day(4)]
    base, used, needed = trailing_baseline(days, 4, "handoff_rate")
    assert used == 3 and base == 0.08  # the 5-request day is not trusted


def test_custom_rules():
    rules = (Rule("handoff_rate", "abs_rise", 0.01, "hand-off up 1 point"),)
    assert [a.metric for a in detect_alerts(with_history(day(0, handoff=0.15)), rules)] == ["handoff_rate"]


# ---- the synthetic log with planted drift ----


def test_planted_drift_fires_and_nothing_else_does():
    assert self_test(lambda _s: None) == 0


def test_the_synthetic_log_is_seeded_and_mixes_in_test_traffic():
    a, b = make_log(7), make_log(7)
    assert a == b and a != make_log(8)
    assert any(is_test_traffic(r["session_id"]) for r in a) and any(
        not is_test_traffic(r["session_id"]) for r in a
    )


def test_planted_alerts_on_exact_days():
    reqs = [
        r for rec_ in make_log(7) if (r := parse_record(rec_, None)) and not is_test_traffic(r.session_id)
    ]
    alerts = detect_alerts(aggregate(reqs))
    spike = START + timedelta(days=SPIKE_DAY - 1)
    step = START + timedelta(days=STEP_DAY - 1)
    assert (spike, "handoff_rate") in {(a.day, a.metric) for a in alerts}
    first_latency = min(a.day for a in alerts if a.metric in ("p50_ms", "p95_ms"))
    assert first_latency == step
    quiet = [a for a in alerts if a.day < spike]
    assert quiet == []  # the stable stretch before the spike: no alert at all
    assert [a for a in alerts if spike < a.day < step] == []  # and none on the two normal days after it
    assert max(a.day for a in alerts) <= START + timedelta(days=N_DAYS - 1)


def test_test_traffic_would_distort_every_day_if_it_were_kept():
    everything = [r for rec_ in make_log(7) if (r := parse_record(rec_, None))]
    real = [r for r in everything if not is_test_traffic(r.session_id)]
    day_all, day_real = aggregate(everything)[0], aggregate(real)[0]
    assert day_all.handoff_rate > 0.3 > 0.15 > day_real.handoff_rate  # eval-/sim- lines are all hand-offs
    assert day_all.p95_ms > 60_000 > day_real.p95_ms and day_all.requests > day_real.requests


def test_planted_drift_always_fires_and_false_alarms_are_rare_across_seeds():
    """Not tuned to one seed. The planted drift must fire on every seed; stray alerts are statistical
    flukes at p < 0.0001 per rate test, so a few logs out of many may have one (measured: 3 of 100)."""
    from harness.monitor_synth import evaluate

    results = [evaluate(seed) for seed in range(1, 41)]
    assert [r for r in results if r["missed"]] == []
    assert sum(1 for r in results if r["stray"]) <= 2
    assert evaluate(7)["stray"] == []


def test_a_rate_alert_needs_statistical_evidence_not_just_a_big_move():
    # 10 requests a day is the minimum; 4 hand-offs of 10 (40%) against about 8% is a big move but weak evidence
    small = [day(i, n=10, handoff=0.08) for i in range(7)] + [day(7, n=10, handoff=0.4)]
    assert detect_alerts(small) == []
    large = [day(i, n=200, handoff=0.08) for i in range(7)] + [day(7, n=200, handoff=0.2)]
    assert [a.metric for a in detect_alerts(large)] == ["handoff_rate"]
    from harness.monitor import binom_tail

    assert binom_tail(10, 4, 0.08) > 0.0001 > binom_tail(200, 40, 0.08)
    assert binom_tail(5, 0, 0.5) == 1.0 and binom_tail(5, 6, 0.5) == 0.0
    assert binom_tail(4, 4, 0.5) == pytest.approx(1 / 16)


# ---- sources: database, JSONL fallback ----


def test_jsonl_without_timestamps_is_dated_by_file_mtime_and_says_so(tmp_path):
    p = tmp_path / "requests.jsonl"
    p.write_text(
        "\n".join(json.dumps(rec(session_id=s)) for s in ("web-1", "eval-1", "web-2")) + "\n",
        encoding="utf-8",
    )
    kept, desc, notes, total, excluded = collect("jsonl", "unused", p, False, IST)
    assert total == 3 and excluded == 1 and len(kept) == 2
    assert "JSONL fallback" in desc and any("carry no timestamp" in n for n in notes)
    assert len({r.day for r in kept}) == 1


def test_jsonl_with_timestamps_uses_them(tmp_path):
    p = tmp_path / "requests.jsonl"
    lines = [
        rec(session_id="web-1", ts="2026-11-01T10:00:00+05:30"),
        rec(session_id="web-2", ts="2026-11-03T10:00:00+05:30"),
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    kept, _, notes, _, _ = collect("jsonl", "unused", p, False, IST)
    assert sorted(r.day for r in kept) == [date(2026, 11, 1), date(2026, 11, 3)] and notes == []


def test_include_test_traffic_keeps_everything(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(
        "\n".join(json.dumps(rec(session_id=s, ts="2026-11-01T10:00:00+05:30")) for s in ("web-1", "eval-1")),
        encoding="utf-8",
    )
    kept, _, _, total, excluded = collect("jsonl", "", p, True, IST)
    assert len(kept) == total == 2 and excluded == 0


def test_auto_falls_back_to_jsonl_and_says_the_database_was_down(tmp_path, monkeypatch):
    def down(dsn, tz):
        raise ConnectionError("refused")

    monkeypatch.setattr(monitor, "load_db", down)
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec(ts="2026-11-01T10:00:00+05:30")), encoding="utf-8")
    kept, desc, notes, _, _ = collect("auto", "dsn", p, False, IST)
    assert len(kept) == 1 and "JSONL fallback" in desc and "the database was not used" in desc
    assert any("Database unavailable (ConnectionError)" in n for n in notes)
    with pytest.raises(ConnectionError):
        collect("db", "dsn", p, False, IST)  # an explicit db source does not silently fall back


def test_load_db_reads_request_log_in_a_read_only_transaction(monkeypatch):
    seen = {}

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql):
            seen["sql"] = sql

        def __iter__(self):
            yield datetime(2026, 11, 1, 5, 0, tzinfo=UTC), "web-1", rec(escalated=True)
            yield datetime(2026, 11, 1, 6, 0, tzinfo=UTC), "eval-1", json.dumps(rec())  # JSONB as text

    class Conn:
        read_only = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return Cur()

    conn = Conn()
    fake = types.SimpleNamespace(connect=lambda dsn, connect_timeout=None: conn)
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    reqs = monitor.load_db("postgresql://x", IST)
    assert conn.read_only is True and seen["sql"].lstrip().upper().startswith("SELECT")
    assert "request_log" in seen["sql"] and "created_at" in seen["sql"]
    assert (
        [r.session_id for r in reqs] == ["web-1", "eval-1"]
        and reqs[0].handoff
        and reqs[0].day == date(2026, 11, 1)
    )


# ---- report ----


def test_markdown_report_sections_and_empty_case():
    days = aggregate([req(rec()) for _ in range(3)])
    md = render_markdown(days, [], source="request_log table", notes=["a note"], total=10, excluded=7)
    assert "Source: request_log table" in md and "Excluded as test traffic: 7" in md and "- a note" in md
    assert "## Per day" in md and "## Alerts" in md and "## Definitions" in md and "2026-11-01" in md
    empty = render_markdown([], [], source="s", notes=[], total=5, excluded=5)
    assert "No requests to report" in empty
    alert_md = render_markdown(
        with_history(day(0, handoff=0.5)),
        detect_alerts(with_history(day(0, handoff=0.5))),
        source="s",
        notes=[],
        total=0,
        excluded=0,
    )
    assert "| handoff_rate |" in alert_md and "50.0%" in alert_md


def test_cli_writes_the_report(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec(ts="2026-11-01T10:00:00+05:30")), encoding="utf-8")
    out = tmp_path / "out" / "monitor.md"
    assert monitor.main(["--source", "jsonl", "--jsonl", str(p), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Monitor: support agent")
    assert monitor.main(["--self-test"]) == 0
    assert "PASS" in capsys.readouterr().out


def db_up(port: int = 5433) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.p1
@pytest.mark.skipif(not db_up() or os.environ.get("CI"), reason="project 1 Postgres is not running")
def test_real_request_log_is_readable_and_has_timestamps():
    reqs = monitor.load_db(monitor.DEFAULT_DSN, IST)
    assert reqs and all(isinstance(r.day, date) for r in reqs)


# ---- log-space tail, per-rule minimums, weekday volume, partial day ----


def exact_tail(n, k, p):
    from math import comb

    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


@pytest.mark.parametrize(
    "n,k,p", [(10, 4, 0.08), (120, 25, 0.08), (400, 80, 0.08), (50, 50, 0.3), (60, 1, 0.01)]
)
def test_log_space_tail_equals_the_exact_sum(n, k, p):
    from harness.monitor import binom_tail

    assert binom_tail(n, k, p) == pytest.approx(exact_tail(n, k, p), rel=1e-9, abs=1e-300)


def test_log_space_tail_survives_sizes_that_break_the_naive_formula():
    from harness.monitor import binom_tail

    # comb(20000, 10000) * p**k overflows or underflows floats; the log-space sum does not
    assert (
        0.0 < binom_tail(20_000, 5_000, 0.2) < 1e-50
    )  # 17 standard deviations above the mean: tiny, and finite
    assert binom_tail(20_000, 4_000, 0.2) == pytest.approx(0.5, abs=0.02)  # k at the mean: about half
    assert binom_tail(5_000, 0, 0.3) == 1.0 and binom_tail(10, 11, 0.5) == 0.0
    assert binom_tail(10, 3, 0.0) == 0.0 and binom_tail(10, 3, 1.0) == 1.0


def test_p95_and_citation_rules_need_more_volume_than_p50():
    # 40 requests a day: enough for p50 (20+) but not for p95 (50+) or citations per answer (30+ is met: 40)
    base = [day(i, n=40) for i in range(7)]
    slow = day(0, n=40, p50=20_000, p95=40_000)
    slow.day = base[-1].day + timedelta(days=1)
    assert sorted(a.metric for a in detect_alerts([*base, slow])) == ["p50_ms"]
    big = [day(i, n=60) for i in range(7)]
    slow_big = day(0, n=60, p50=20_000, p95=40_000)
    slow_big.day = big[-1].day + timedelta(days=1)
    assert sorted(a.metric for a in detect_alerts([*big, slow_big])) == ["p50_ms", "p95_ms"]
    few = [day(i, n=25) for i in range(7)]
    fewer = day(0, n=25, cites=1.0, p50=20_000)
    fewer.day = few[-1].day + timedelta(days=1)
    assert sorted(a.metric for a in detect_alerts([*few, fewer])) == ["p50_ms"]  # citations rule needs 30+


def test_volume_baseline_days_need_enough_requests_but_a_collapse_has_no_minimum():
    thin_history = [
        day(i, n=20) for i in range(7)
    ]  # baseline days under 30 requests are not a trustworthy volume
    collapse = day(0, n=2)
    collapse.day = thin_history[-1].day + timedelta(days=1)
    assert detect_alerts([*thin_history, collapse]) == []
    solid = [day(i, n=100) for i in range(7)]
    collapse2 = day(0, n=2)
    collapse2.day = solid[-1].day + timedelta(days=1)
    assert [a.metric for a in detect_alerts([*solid, collapse2])] == ["requests"]


def week(start_index, weekday_n, sunday_n):
    """Seven consecutive days from 2026-10-01 + start_index: volume by weekday (Sunday is the closed day)."""
    out = []
    for i in range(start_index, start_index + 7):
        d = day(i, n=weekday_n)
        if d.day.weekday() == 6:
            d.requests = sunday_n
        out.append(d)
    return out


def test_sundays_are_compared_with_sundays_not_with_the_working_week():
    # Thursday 1 Oct start: a normal pattern of 300 a day Monday to Saturday and 100 on Sundays
    days = week(0, 300, 100) + week(7, 300, 100) + week(14, 300, 100)
    assert [
        a for a in detect_alerts(days) if a.metric == "requests"
    ] == []  # Sundays (1/3) never read as a collapse
    # a Sunday of 30 against earlier Sundays of 100 is a real drop for a Sunday
    days[17].requests = 30  # 2026-10-18, a Sunday
    assert days[17].day.weekday() == 6
    alerts = [a for a in detect_alerts(days) if a.metric == "requests"]
    assert [(a.day, a.value, a.baseline) for a in alerts] == [(date(2026, 10, 18), 30.0, 100.0)]
    # a Monday of 100 against a Monday to Saturday baseline of 300 is a drop too
    days = week(0, 300, 100) + week(7, 300, 100)
    monday = next(d for d in days[8:] if d.day.weekday() == 0)
    monday.requests = 100
    assert [a.day for a in detect_alerts(days) if a.metric == "requests"] == [monday.day]


def test_a_sunday_needs_two_earlier_sundays_before_the_volume_rule_speaks():
    from harness.monitor import baseline_days

    days = week(0, 300, 100) + week(7, 300, 100)
    first_sunday = next(i for i, d in enumerate(days) if d.day.weekday() == 6)
    second_sunday = next(i for i, d in enumerate(days) if d.day.weekday() == 6 and i > first_sunday)
    assert (
        baseline_days(days, first_sunday, "requests", 30)[1] == 2
        and baseline_days(days, first_sunday, "requests", 30)[0] == []
    )
    assert len(baseline_days(days, second_sunday, "requests", 30)[0]) == 1  # one earlier Sunday: not enough
    days[second_sunday].requests = 10
    assert [a for a in detect_alerts(days) if a.metric == "requests"] == []


def test_a_day_still_going_on_is_not_read_as_a_volume_collapse():
    days = week(0, 300, 100)
    partial = day(7, n=40)  # 2026-10-08 (a Thursday), checked mid-morning
    days.append(partial)
    assert [a.metric for a in detect_alerts(days)] == ["requests"]
    assert detect_alerts(days, partial_day=partial.day) == []
    partial.handoff_rate = 0.9
    assert [a.metric for a in detect_alerts(days, partial_day=partial.day)] == [
        "handoff_rate"
    ]  # other metrics still count


def test_a_failure_in_the_alert_logic_does_not_cost_the_report(tmp_path, monkeypatch, capsys):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec(ts="2026-11-01T10:00:00+05:30")), encoding="utf-8")

    def broken(*a, **k):
        raise RuntimeError("alert bug")

    monkeypatch.setattr(monitor, "detect_alerts", broken)
    out = tmp_path / "m.md"
    assert monitor.main(["--source", "jsonl", "--jsonl", str(p), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Alert detection failed (RuntimeError: alert bug)" in text and "| 2026-11-01 |" in text
    capsys.readouterr()


def test_report_states_the_minimum_valid_volume_and_marks_low_days():
    low = day(0, n=20)
    low.day = date(2026, 11, 1)
    ok = day(1, n=monitor.VALID_MIN_DAILY_REQUESTS + 20)
    ok.day = date(2026, 11, 2)
    md = render_markdown([low, ok], [], source="s", notes=[], total=0, excluded=0)
    assert f"Valid for about {monitor.VALID_MIN_DAILY_REQUESTS}+ requests a day" in md
    assert "| 2026-11-01 | 20* |" in md and f"| 2026-11-02 | {ok.requests} |" in md
    assert "Fewer than" in md and "Sundays with Sundays" in md and "p95" in md


def test_low_volume_log_stays_quiet_rather_than_guessing():
    from harness.monitor_synth import low_volume_check

    lines: list[str] = []
    assert low_volume_check(per_day=12, seeds=range(1, 8), write=lines.append) is True
    assert "false alarms in 0" in lines[0]


def test_the_valid_volume_claim_matches_what_the_synthetic_sweep_measures():
    """VALID_MIN_DAILY_REQUESTS is where the planted drift is found in (nearly) every log. A light sweep guards the claim."""
    from harness.monitor_synth import volume_sweep

    rows = {
        r["per_day"]: r
        for r in volume_sweep(volumes=(10, monitor.VALID_MIN_DAILY_REQUESTS, 120), seeds=range(1, 11))
    }
    assert (
        rows[10]["spike_detected"] < 0.5
    )  # far below the minimum the monitor misses real drift, and says so
    assert rows[monitor.VALID_MIN_DAILY_REQUESTS]["spike_detected"] >= 0.9
    assert rows[120]["spike_detected"] == 1.0 and rows[120]["step_detected"] == 1.0
    assert all(r["false_alarm_logs"] <= 0.1 for r in rows.values())
