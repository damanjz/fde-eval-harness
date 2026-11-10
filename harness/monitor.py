"""Production monitoring for the support agent: per-day metrics and drift alerts.

    python -m harness.monitor [--out reports/monitor.md] [--source auto|db|jsonl] [--include-test-traffic]
    python -m harness.monitor --self-test

Source: project 1's `request_log` table (created_at, record JSONB; Postgres on port 5433), read-only.
Its `logs/requests.jsonl` records carry no timestamp, so the file is used only when the database is
down, and the report says so. Test traffic (session ids starting eval-, sim-, smoke, warmup) is
excluded unless --include-test-traffic is given.

Definitions (one request = one agent reply):
  hand-off rate        share of requests with escalated = true
  guardrail flag rate  share of requests with non-empty output_flags or injection_suspected = true
  tool error rate      tool calls whose result has an "error" key, over all tool calls that day
  latency p50 / p95    record latency_ms, nearest-rank percentile
  citations per answer mean number of citations over requests that were not hand-offs
Alert: a day's metric moves past its threshold against the median of the same metric over the
trailing 7 calendar days (days that have data), given enough history and enough requests that day.
A rate alert also has to be statistically unlikely, not just big: the day's event count must have a
one-sided exact binomial p below 0.0001 under the pooled rate of the trailing days. Without that, small
daily samples trip the point thresholds by chance (on seeded synthetic logs they alone gave a false alarm
in about one log in four). A recent spike raises the pooled rate for the following week, so a second,
smaller spike right after it can be masked.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from harness import harden_console
from harness.stats import median, percentile

TEST_PREFIXES = ("eval-", "sim-", "smoke", "warmup")
DEFAULT_REPO = Path(os.environ.get("FDE_SUPPORT_AGENT_REPO", str(Path(__file__).resolve().parents[2] / "fde-support-agent")))
DEFAULT_DSN = os.environ.get(
    "FDE_SUPPORT_AGENT_DB", "postgresql://larkfield:larkfield@localhost:5433/larkfield"
)
IST = timezone(timedelta(hours=5, minutes=30))
WINDOW_DAYS = 7


@dataclass(frozen=True)
class Req:
    day: date
    session_id: str
    handoff: bool
    flagged: bool
    tool_calls: int
    tool_errors: int
    latency_ms: float | None
    citations: int
    articles: tuple[str, ...]


@dataclass
class DayStats:
    day: date
    requests: int
    handoff_rate: float
    flag_rate: float
    tool_calls: int
    tool_error_rate: float | None  # None when there were no tool calls
    p50_ms: float
    p95_ms: float
    citations_per_answer: float | None  # None when every request was a hand-off
    top_articles: list[tuple[str, int]]

    def rate_n(self, name: str) -> int:
        """How many trials the rate metric is a share of: requests, or tool calls for the tool error rate."""
        return self.tool_calls if name == "tool_error_rate" else self.requests

    def metric(self, name: str) -> float | None:
        return {
            "requests": float(self.requests),
            "handoff_rate": self.handoff_rate,
            "flag_rate": self.flag_rate,
            "tool_error_rate": self.tool_error_rate,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "citations_per_answer": self.citations_per_answer,
        }[name]


@dataclass(frozen=True)
class Rule:
    """metric, how it is compared with the baseline, the threshold, and the volume it needs to be trusted."""

    metric: str
    kind: str  # "abs_rise": value - baseline >= threshold; "ratio_rise": value / baseline >= threshold;
    #            "ratio_drop": value / baseline <= threshold; "ratio_both": rise or drop by that factor
    threshold: float
    label: str
    min_abs: float = (
        0.0  # ratio rules also need the absolute move to be at least this (ignores tiny baselines)
    )
    min_today: int = 10  # requests the day itself needs before this rule may fire
    min_base: int = 10  # requests a trailing day needs to count towards the baseline


# Minimum volumes. p95 of fewer than 20 values is just the maximum, so it needs more than p50; citations are a
# noisy mean; the volume rule fires on a collapse, so today's count has no minimum, but its baseline days do.
DEFAULT_RULES = (
    Rule("handoff_rate", "abs_rise", 0.10, "hand-off rate up 10+ points"),
    Rule("flag_rate", "abs_rise", 0.05, "guardrail flag rate up 5+ points"),
    Rule("tool_error_rate", "abs_rise", 0.05, "tool error rate up 5+ points"),
    Rule(
        "p50_ms", "ratio_rise", 1.5, "p50 latency 1.5x the baseline", min_abs=500, min_today=20, min_base=20
    ),
    Rule(
        "p95_ms", "ratio_rise", 1.5, "p95 latency 1.5x the baseline", min_abs=500, min_today=50, min_base=50
    ),
    Rule(
        "citations_per_answer", "ratio_drop", 0.7, "citations per answer down 30%+", min_today=30, min_base=30
    ),
    Rule("requests", "ratio_both", 2.0, "request volume doubled or halved", min_today=0, min_base=30),
)
RATE_ALPHA = 0.0001  # one-sided exact binomial p a rate alert must beat (about 60 looks per 3 weeks)
MIN_HISTORY_DAYS = 3  # trailing days with data needed before any alert
MIN_REQUESTS = 10  # a day with fewer requests is reported but cannot raise a rate alert
MIN_TOOL_CALLS = 5
SUNDAY_HISTORY_DAYS, SUNDAY_WINDOW_DAYS = 2, 28  # Sundays (support closed) are compared with Sundays
# Daily volume from which the planted drift in the synthetic proof is detected reliably, measured by
# monitor_synth.volume_sweep (see docs/runbook.md). Below it the report still shows the numbers.
VALID_MIN_DAILY_REQUESTS = 80


@dataclass
class Alert:
    day: date
    metric: str
    value: float
    baseline: float
    label: str


# ---- loading ----


def parse_record(
    record: dict, ts: datetime | None, session_id: str | None = None, tz: timezone = IST
) -> Req | None:
    """Turn one log record into a Req. None when it has no usable timestamp."""
    if ts is None:
        raw = record.get("ts") or record.get("timestamp") or record.get("created_at")
        if isinstance(raw, str):
            try:
                ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                ts = None
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(tz)
    calls = [c for c in (record.get("tool_calls") or []) if isinstance(c, dict)]
    errors = sum(1 for c in calls if isinstance(c.get("result"), dict) and "error" in c["result"])
    lat = record.get("latency_ms")
    if lat is None and isinstance(record.get("usage"), dict):
        lat = record["usage"].get("latency_ms")
    return Req(
        day=ts.date(),
        session_id=str(session_id if session_id is not None else record.get("session_id") or ""),
        handoff=bool(record.get("escalated")),
        flagged=bool(record.get("output_flags")) or bool(record.get("injection_suspected")),
        tool_calls=len(calls),
        tool_errors=errors,
        latency_ms=float(lat) if isinstance(lat, (int, float)) else None,
        citations=len(record.get("citations") or []),
        articles=tuple(
            sorted(
                {
                    r.get("article_id")
                    for r in (record.get("retrieved") or [])
                    if isinstance(r, dict) and r.get("article_id")
                }
            )
        ),
    )


def load_db(dsn: str, tz: timezone = IST) -> list[Req]:
    """Read request_log in a read-only transaction. Raises on any connection problem."""
    import psycopg  # lazy: only the monitor needs it, and only for the database source

    reqs: list[Req] = []
    with psycopg.connect(dsn, connect_timeout=3) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SELECT created_at, session_id, record FROM request_log ORDER BY created_at, id")
            for created_at, session_id, record in cur:
                rec = record if isinstance(record, dict) else json.loads(record)
                req = parse_record(rec, created_at, session_id, tz)
                if req:
                    reqs.append(req)
    return reqs


def load_jsonl(path: Path, tz: timezone = IST) -> tuple[list[Req], int, date | None]:
    """Read the JSONL log. Returns (requests, records without a timestamp, file modification day).

    Records with a ts field are used as they are. Records without one are dated by the file's
    modification time (all in one bucket), because the log has nothing better.
    """
    mtime_day = datetime.fromtimestamp(path.stat().st_mtime, tz).date() if path.exists() else None
    reqs: list[Req] = []
    undated = 0
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        req = parse_record(rec, None, tz=tz)
        if req is None:
            undated += 1
            req = parse_record(
                rec, datetime.combine(mtime_day, datetime.min.time(), tzinfo=tz) if mtime_day else None, tz=tz
            )
        if req:
            reqs.append(req)
    return reqs, undated, mtime_day


def is_test_traffic(session_id: str, prefixes: tuple[str, ...] = TEST_PREFIXES) -> bool:
    return session_id.startswith(prefixes)


# ---- aggregation ----


def aggregate(reqs: list[Req]) -> list[DayStats]:
    by_day: dict[date, list[Req]] = {}
    for r in reqs:
        by_day.setdefault(r.day, []).append(r)
    out: list[DayStats] = []
    for day in sorted(by_day):
        rs = by_day[day]
        n = len(rs)
        calls = sum(r.tool_calls for r in rs)
        answers = [r for r in rs if not r.handoff]
        lat = [r.latency_ms for r in rs if r.latency_ms is not None]
        top = Counter(a for r in rs for a in r.articles).most_common(3)
        out.append(
            DayStats(
                day=day,
                requests=n,
                handoff_rate=sum(r.handoff for r in rs) / n,
                flag_rate=sum(r.flagged for r in rs) / n,
                tool_calls=calls,
                tool_error_rate=(sum(r.tool_errors for r in rs) / calls) if calls else None,
                p50_ms=percentile(lat, 50),
                p95_ms=percentile(lat, 95),
                citations_per_answer=(sum(r.citations for r in answers) / len(answers)) if answers else None,
                top_articles=top,
            )
        )
    return out


def is_sunday(d: date) -> bool:
    return d.weekday() == 6


def baseline_days(days: list[DayStats], i: int, metric: str, min_base: int) -> tuple[list[DayStats], int]:
    """The earlier days that form the baseline for days[i], and how many are needed.

    Normally the previous 7 calendar days that have enough requests. Request volume compares like with like:
    support runs Monday to Saturday, so Sundays are compared with earlier Sundays (4 weeks back, 2 needed) and
    Monday to Saturday with Monday to Saturday (7 days back, 3 needed).
    """
    day = days[i].day
    window, needed = WINDOW_DAYS, MIN_HISTORY_DAYS
    same_class = None
    if metric == "requests":
        same_class = is_sunday(day)
        if same_class:
            window, needed = SUNDAY_WINDOW_DAYS, SUNDAY_HISTORY_DAYS
    out = []
    for prev in days[:i]:
        if not (timedelta(days=1) <= day - prev.day <= timedelta(days=window)):
            continue
        if same_class is not None and is_sunday(prev.day) != same_class:
            continue
        if prev.requests >= min_base:
            out.append(prev)
    return out, needed


def trailing_baseline(
    days: list[DayStats], i: int, metric: str, min_base: int = MIN_REQUESTS
) -> tuple[float | None, int, int]:
    """Median of the metric over the baseline days. Returns (value, days used, days needed)."""
    prev_days, needed = baseline_days(days, i, metric, min_base)
    vals = [v for p in prev_days if (v := p.metric(metric)) is not None]
    return (median(vals), len(vals), needed) if vals else (None, 0, needed)


def binom_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact, computed in log space so a large n cannot overflow or underflow."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    base = math.lgamma(n + 1)
    terms = [
        base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def pooled_baseline(
    days: list[DayStats], i: int, metric: str, min_base: int = MIN_REQUESTS
) -> tuple[int, int]:
    """(events, trials) summed over the trailing 7 calendar days with enough requests: the pooled
    baseline rate, a steadier estimate of the true rate than any one day or the median of daily rates."""
    events = trials = 0
    for prev in baseline_days(days, i, metric, min_base)[0]:
        n = prev.rate_n(metric)
        events += round((prev.metric(metric) or 0.0) * n)
        trials += n
    return events, trials


def rate_is_unlikely(days: list[DayStats], i: int, metric: str) -> bool:
    """Is today's event count too high to be chance at the pooled baseline rate? A zero baseline is
    floored at half an event per day so a single event is not treated as impossible."""
    d = days[i]
    n = d.rate_n(metric)
    k = round((d.metric(metric) or 0.0) * n)
    events, trials = pooled_baseline(days, i, metric)
    p0 = events / trials if trials else 0.0
    p0 = min(max(p0, 0.5 / max(n, 1)), 0.999)
    return binom_tail(n, k, p0) < RATE_ALPHA


def detect_alerts(
    days: list[DayStats], rules: tuple[Rule, ...] = DEFAULT_RULES, partial_day: date | None = None
) -> list[Alert]:
    """partial_day is a day that is still going on (today): its request volume is not compared, because a
    half day looks like a collapse."""
    alerts: list[Alert] = []
    for i, d in enumerate(days):
        for rule in rules:
            if d.requests < rule.min_today:
                continue
            if rule.metric == "requests" and d.day == partial_day:
                continue
            value = d.metric(rule.metric)
            if value is None or (rule.metric == "tool_error_rate" and d.tool_calls < MIN_TOOL_CALLS):
                continue
            base, used, needed = trailing_baseline(days, i, rule.metric, rule.min_base)
            if base is None or used < needed:
                continue
            fire = False
            if rule.kind == "abs_rise":
                fire = value - base >= rule.threshold and rate_is_unlikely(days, i, rule.metric)
            elif base > 0:
                ratio = value / base
                moved = abs(value - base) >= rule.min_abs
                if rule.kind == "ratio_rise":
                    fire = ratio >= rule.threshold and moved
                elif rule.kind == "ratio_drop":
                    fire = ratio <= rule.threshold
                elif rule.kind == "ratio_both":
                    fire = ratio >= rule.threshold or ratio <= 1 / rule.threshold
            if fire:
                alerts.append(Alert(d.day, rule.metric, value, base, rule.label))
    return alerts


# ---- report ----


def _fmt(metric: str, v: float | None) -> str:
    if v is None:
        return "-"
    if metric in ("handoff_rate", "flag_rate", "tool_error_rate"):
        return f"{100 * v:.1f}%"
    if metric in ("p50_ms", "p95_ms"):
        return f"{v / 1000:.1f}s"
    if metric == "requests":
        return f"{v:.0f}"
    return f"{v:.2f}"


def render_markdown(
    days: list[DayStats],
    alerts: list[Alert],
    *,
    source: str,
    notes: list[str],
    total: int,
    excluded: int,
    generated: str | None = None,
    test_traffic_included: bool = False,
) -> str:
    L = [
        "# Monitor: support agent",
        "",
        f"Generated {generated or datetime.now().strftime('%Y-%m-%d %H:%M')}.",
        "",
        f"- Source: {source}",
        f"- Requests read: {total}. Excluded as test traffic: {excluded}"
        + (
            f" (session ids starting {', '.join(TEST_PREFIXES)})."
            if not test_traffic_included
            else " (kept: --include-test-traffic)."
        ),
        f"- Requests reported: {sum(d.requests for d in days)} over {len(days)} day(s).",
    ]
    L += [f"- {n}" for n in notes]
    L += ["", "## Per day", ""]
    if not days:
        L += ["No requests to report. All traffic in the source is test traffic, or the source is empty.", ""]
    else:
        L += [
            "| day | requests | hand-off | guardrail flags | tool errors | p50 | p95 | citations per answer | top retrieved articles |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for d in days:
            top = ", ".join(f"{a} ({n})" for a, n in d.top_articles) or "-"
            low = "*" if d.requests < VALID_MIN_DAILY_REQUESTS else ""
            L.append(
                f"| {d.day.isoformat()} | {d.requests}{low} | {_fmt('handoff_rate', d.handoff_rate)} | "
                f"{_fmt('flag_rate', d.flag_rate)} | {_fmt('tool_error_rate', d.tool_error_rate)}"
                f"{f' ({d.tool_calls} calls)' if d.tool_calls else ''} | {_fmt('p50_ms', d.p50_ms)} | "
                f"{_fmt('p95_ms', d.p95_ms)} | {_fmt('citations_per_answer', d.citations_per_answer)} | {top} |"
            )
        if any(d.requests < VALID_MIN_DAILY_REQUESTS for d in days):
            L += [
                "",
                f"\\* Fewer than {VALID_MIN_DAILY_REQUESTS} requests: shown, but too few for the alert rules to be "
                "reliable (see Alerts).",
            ]
    L += ["", "## Alerts", ""]
    if alerts:
        L += ["| day | metric | value | trailing 7-day median | rule |", "|---|---|---|---|---|"]
        L += [
            f"| {a.day.isoformat()} | {a.metric} | {_fmt(a.metric, a.value)} | {_fmt(a.metric, a.baseline)} | {a.label} |"
            for a in alerts
        ]
    else:
        L.append(
            f"None. An alert needs at least {MIN_HISTORY_DAYS} earlier days with enough requests in the trailing "
            f"{WINDOW_DAYS} days (Sundays: {SUNDAY_HISTORY_DAYS} earlier Sundays), and enough requests on the day."
        )
    L += [
        "",
        f"Valid for about {VALID_MIN_DAILY_REQUESTS}+ requests a day. Below that the alert rules can miss real drift "
        "(a rate spike cannot be told from chance on a few dozen requests) and p95 and citation rules stay silent; "
        "the per-day numbers are still correct. Minimum volumes per rule: "
        + "; ".join(
            f"{r.metric} {r.min_today if r.min_today else 'any'} (baseline days {r.min_base}+)"
            for r in DEFAULT_RULES
        )
        + ". Request volume compares Sundays with Sundays and Monday to Saturday with Monday to Saturday, and "
        "skips a day that is still going on.",
    ]
    L += [
        "",
        "Rules: "
        + "; ".join(r.label for r in DEFAULT_RULES)
        + ". Rate rules also need the day's count to be "
        f"unlikely by chance (one-sided binomial p below {RATE_ALPHA}).",
        "",
        "## Definitions",
        "",
        "- Hand-off: the reply was escalated (`escalated`). Guardrail flag: non-empty `output_flags` or `injection_suspected`.",
        "- Tool error: a `tool_calls` entry whose `result` has an `error` key; the rate is errors over tool calls that day.",
        "- Citations per answer: mean citations over requests that were not hand-offs.",
        "- Baseline: median of the same metric over the previous 7 calendar days that have enough requests.",
        "",
    ]
    return "\n".join(L)


# ---- CLI ----


def collect(
    source: str, dsn: str, jsonl: Path, include_test: bool, tz: timezone
) -> tuple[list[Req], str, list[str], int, int]:
    """Returns (requests kept, source description, notes, total read, excluded)."""
    notes: list[str] = []
    reqs: list[Req]
    if source in ("auto", "db"):
        try:
            reqs = load_db(dsn, tz)
            desc = "request_log table (Postgres, read-only; created_at from the database)"
        except Exception as e:  # noqa: BLE001 - any connection or schema problem
            if source == "db":
                raise
            reqs = []
            desc = ""
            notes.append(f"Database unavailable ({type(e).__name__}); fell back to the JSONL file.")
    else:
        reqs, desc = [], ""
    if not desc:
        reqs, undated, mtime_day = load_jsonl(jsonl, tz)
        desc = f"JSONL fallback: {jsonl} (the database was not used)"
        if undated:
            notes.append(
                f"{undated} record(s) carry no timestamp, so they are dated by the file's modification day "
                f"({mtime_day}) and share one bucket: per-day trends and alerts are not meaningful. "
                "Use the database source for real per-day numbers."
            )
    total = len(reqs)
    kept = reqs if include_test else [r for r in reqs if not is_test_traffic(r.session_id)]
    if not include_test and total > len(kept):
        dropped = [r for r in reqs if is_test_traffic(r.session_id)]
        by_prefix = Counter(next(p for p in TEST_PREFIXES if r.session_id.startswith(p)) for r in dropped)
        span = f"{min(r.day for r in dropped)} to {max(r.day for r in dropped)}"
        notes.append(
            "Excluded test traffic by session prefix: "
            + ", ".join(f"{p} {n}" for p, n in sorted(by_prefix.items()))
            + f"; dates {span}."
        )
    return kept, desc, notes, total, total - len(kept)


def main(argv: list[str] | None = None) -> int:
    harden_console()
    p = argparse.ArgumentParser(
        prog="python -m harness.monitor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="reports/monitor.md")
    p.add_argument("--source", choices=["auto", "db", "jsonl"], default="auto")
    p.add_argument(
        "--dsn", default=DEFAULT_DSN, help="Postgres URL of project 1's database (default: its local one)"
    )
    p.add_argument("--jsonl", default=str(DEFAULT_REPO / "logs" / "requests.jsonl"))
    p.add_argument("--include-test-traffic", action="store_true")
    p.add_argument(
        "--utc-offset", type=float, default=5.5, help="hours east of UTC for day boundaries (default IST 5.5)"
    )
    p.add_argument(
        "--self-test", action="store_true", help="prove the alert logic on a seeded synthetic log and exit"
    )
    args = p.parse_args(argv)
    if args.self_test:
        from harness.monitor_synth import self_test

        return self_test(sys.stdout.write)
    tz = timezone(timedelta(hours=args.utc_offset))
    try:
        kept, desc, notes, total, excluded = collect(
            args.source, args.dsn, Path(args.jsonl), args.include_test_traffic, tz
        )
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    days = aggregate(kept)
    try:
        # the day that is still going on is not complete, so its volume must not be read as a collapse
        alerts = detect_alerts(days, partial_day=datetime.now(tz).date())
    except Exception as e:  # noqa: BLE001 - a bug in the alert logic must not cost the report
        alerts = []
        notes.append(
            f"Alert detection failed ({type(e).__name__}: {e}); the per-day numbers are still valid."
        )
    text = render_markdown(
        days,
        alerts,
        source=desc,
        notes=notes,
        total=total,
        excluded=excluded,
        test_traffic_included=args.include_test_traffic,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(
        f"wrote {out}: {len(days)} day(s), {sum(d.requests for d in days)} request(s) reported, "
        f"{excluded} test request(s) excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
