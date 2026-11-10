"""A seeded synthetic request log with planted drift, to prove the monitor's alert logic.

Everything here is made up on purpose: the point is to know in advance which alerts must fire.
Plan (21 days from 2026-10-01, about 120 real requests a day on Monday to Saturday and about a third of
that on Sundays, when support is closed, plus noisy test traffic):
  days 1-13   stable
  day 14      hand-off spike (35% instead of about 8%): must alert on hand-off rate, that day only
  days 15-16  stable again: must not alert (a median baseline shrugs off one bad day)
  day 17 on   latency step to 2.2x: must alert on p50 and p95 from day 17
The test traffic uses eval- and sim- session ids and is deliberately extreme: if exclusion failed,
alerts would fire on the stable days.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import date, datetime, timedelta

from harness.monitor import IST, aggregate, detect_alerts, is_test_traffic, parse_record

START = date(2026, 10, 1)
SPIKE_DAY, STEP_DAY, N_DAYS = 14, 17, 21  # 1-based day numbers
ARTICLES = [f"KB-{i:03d}" for i in range(1, 9)]


def make_log(seed: int = 7, per_day: int = 120, sunday_factor: float = 0.35) -> list[dict]:
    """Records in project 1's request_log shape, each with a ts. Deterministic for a seed."""
    rng = random.Random(seed)
    records: list[dict] = []
    for d in range(1, N_DAYS + 1):
        day = START + timedelta(days=d - 1)
        n = int(per_day * rng.uniform(0.9, 1.1) * (sunday_factor if day.weekday() == 6 else 1.0))
        handoff_p = 0.35 if d == SPIKE_DAY else 0.08
        lat_scale = 2.2 if d >= STEP_DAY else 1.0
        for i in range(n):
            ts = datetime(day.year, day.month, day.day, 9, 0, tzinfo=IST) + timedelta(
                seconds=rng.randint(0, 36000)
            )
            handoff = rng.random() < handoff_p
            calls = []
            if rng.random() < 0.4:
                err = rng.random() < 0.01
                calls.append(
                    {
                        "name": "lookup_order",
                        "args": {"order_id": "LC-1000"},
                        "result": {"error": "order not found"} if err else {"status": "shipped"},
                    }
                )
            records.append(
                {
                    "session_id": f"web-{d:02d}-{i:04d}",
                    "ts": ts.isoformat(),
                    "message": "synthetic",
                    "escalated": handoff,
                    "injection_suspected": False,
                    "output_flags": ["code"] if rng.random() < 0.02 else [],
                    "tool_calls": calls,
                    "citations": [] if handoff else rng.sample(ARTICLES, rng.choice([1, 2, 2, 3])),
                    "retrieved": [{"article_id": a} for a in rng.sample(ARTICLES, 3)],
                    "latency_ms": int(rng.lognormvariate(9.4, 0.35) * lat_scale),
                }
            )
        for i in range(max(per_day // 3, 5)):  # extreme test traffic that must be excluded
            ts = datetime(day.year, day.month, day.day, 2, 0, tzinfo=IST) + timedelta(
                seconds=rng.randint(0, 3600)
            )
            records.append(
                {
                    "session_id": f"{rng.choice(['eval-agent-', 'sim-angry-'])}{d:02d}-{i:03d}",
                    "ts": ts.isoformat(),
                    "escalated": True,
                    "output_flags": ["code"],
                    "injection_suspected": True,
                    "tool_calls": [{"name": "lookup_order", "args": {}, "result": {"error": "x"}}],
                    "citations": [],
                    "retrieved": [],
                    "latency_ms": 90_000,
                }
            )
    return records


def evaluate(seed: int = 7, per_day: int = 120) -> dict:
    """Run the alert logic on the planted log for one seed and say what fired, what was missed, what was stray."""
    reqs = [r for rec in make_log(seed, per_day) if (r := parse_record(rec, None, tz=IST)) is not None]
    kept = [r for r in reqs if not is_test_traffic(r.session_id)]
    days = aggregate(kept)
    got = {(a.day, a.metric) for a in detect_alerts(days)}
    spike_date = START + timedelta(days=SPIKE_DAY - 1)
    step_date = START + timedelta(days=STEP_DAY - 1)
    missed: list[str] = []
    if (spike_date, "handoff_rate") not in got:
        missed.append("hand-off spike")
    if (step_date, "p50_ms") not in got:
        missed.append("latency step (p50 on its first day)")
    planted = {(spike_date, "handoff_rate")} | {
        (step_date + timedelta(days=k), m)
        for k in range(0, N_DAYS - STEP_DAY + 1)
        for m in ("p50_ms", "p95_ms")
    }
    return {
        "requests": len(reqs),
        "excluded": len(reqs) - len(kept),
        "days": len(days),
        "fired": sorted((str(d), m) for d, m in got),
        "missed": missed,
        "stray": sorted((str(d), m) for d, m in got - planted),
    }


def volume_sweep(volumes=(10, 20, 40, 80, 120), seeds=range(1, 21)) -> list[dict]:
    """How the alert logic behaves as daily volume falls: for each volume (requests a day Monday to Saturday),
    over the seeds, the share of logs where the planted hand-off spike and latency step were detected, and
    the share of logs with a false alarm. This is where the monitor's minimum valid volume comes from."""
    rows = []
    for per_day in volumes:
        results = [evaluate(seed, per_day) for seed in seeds]
        n = len(results)
        rows.append(
            {
                "per_day": per_day,
                "logs": n,
                "spike_detected": sum(1 for r in results if "hand-off spike" not in r["missed"]) / n,
                "step_detected": sum(1 for r in results if not any("latency" in m for m in r["missed"])) / n,
                "false_alarm_logs": sum(1 for r in results if r["stray"]) / n,
            }
        )
    return rows


def low_volume_check(per_day: int = 12, seeds=range(1, 6), write: Callable[[str], object] = print) -> bool:
    """At a volume far below the valid minimum the monitor must stay quiet rather than guess: no false alarms.
    It is allowed (and expected) to miss the planted drift; the point is to say so."""
    results = [evaluate(seed, per_day) for seed in seeds]
    strays = [r["stray"] for r in results if r["stray"]]
    caught = sum(1 for r in results if "hand-off spike" not in r["missed"])
    write(
        f"low volume ({per_day} requests a day, {len(results)} logs): hand-off spike detected in {caught}, "
        f"false alarms in {len(strays)}\n"
    )
    return not strays


def self_test(write: Callable[[str], object] = print, seed: int = 7) -> int:
    """Returns 0 when exactly the planted drift fires: nothing missed, nothing stray, test traffic excluded."""
    r = evaluate(seed)
    problems = [f"missed: {', '.join(r['missed'])}"] if r["missed"] else []
    if r["stray"]:
        problems.append(f"alerts that were not planted (false alarms): {r['stray']}")
    if r["excluded"] == 0:
        problems.append("test traffic was not excluded")
    write(
        f"synthetic log: {r['requests']} requests, {r['excluded']} test requests excluded, {r['days']} days\n"
    )
    write(f"alerts fired: {r['fired']}\n")
    if not low_volume_check(write=write):
        problems.append("false alarms at low volume")
    write(
        "PASS: exactly the planted drift fired, and nothing at low volume\n"
        if not problems
        else "FAIL: " + "; ".join(problems) + "\n"
    )
    return 0 if not problems else 1
