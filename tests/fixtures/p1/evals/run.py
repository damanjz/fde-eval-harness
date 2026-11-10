"""Eval runner.

Usage:
  python -m evals.run --variant agent|keyword|no_retrieval|all [--limit N] [--ids A01,T03]
                      [--judge] [--resume] [--timeout SECONDS] [--report-only]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.scoring import (
    cost_inr,
    fmt_rate,
    judge_case,
    percentile,
    reply_to_dict,
    score_case,
)

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "cases.jsonl"
RESULTS_DIR = ROOT / "results"
REPORT_PATH = ROOT / "report.md"

VARIANTS = {"agent": "hybrid", "keyword": "keyword", "no_retrieval": "none"}
CATEGORIES = ["answer", "tool", "refuse"]
DEFAULT_TIMEOUT_S = 180.0
EVAL_NOW = "2026-11-10T09:00:00+05:30"  # dataset date; see settings.clock_now

RunTurn = Callable[..., Any]


def load_cases(path: Path = CASES_PATH, ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        wanted = set(ids)
        cases = [c for c in cases if c["id"] in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def _default_run_turn() -> RunTurn:
    from app.agent import run_turn  # lazy: scoring and tests work without the agent

    return run_turn


def _default_chat() -> Callable[[list[dict]], Any]:
    from app.llm import chat  # lazy

    return chat


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run fn in a daemon thread; raise TimeoutError if it does not finish in time."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - relayed to caller
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"no reply within {timeout_s:g}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def case_turns(case: dict) -> list[str]:
    return list(case["turns"]) if case.get("turns") else [case["question"]]


def merge_replies(replies: list[dict]) -> dict:
    """Combine turn replies: last text, union citations, all tool calls, last pending, summed usage."""
    merged: dict[str, Any] = {"text": "", "citations": [], "tool_calls": [], "pending_action": None,
                              "escalated": False, "usage": {}}
    for r in replies:
        merged["text"] = r.get("text", "")
        for c in r.get("citations") or []:
            if c not in merged["citations"]:
                merged["citations"].append(c)
        merged["tool_calls"].extend(r.get("tool_calls") or [])
        if r.get("pending_action"):
            merged["pending_action"] = r["pending_action"]
        merged["escalated"] = merged["escalated"] or bool(r.get("escalated"))
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                merged["usage"][k] = merged["usage"].get(k, 0) + v
    return merged


def run_case(case: dict, variant: str, run_turn: RunTurn, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Run one case on a fresh session. Never raises: failures are recorded in the result."""
    session_id = f"eval-{variant}-{case['id']}-{uuid.uuid4().hex[:8]}"
    mode = VARIANTS[variant]
    t0 = time.perf_counter()
    replies: list[dict] = []
    error: str | None = None
    try:
        for msg in case_turns(case):
            reply = _call_with_timeout(
                lambda m=msg: run_turn(session_id, m, confirm=None, retrieval_mode=mode), timeout_s
            )
            replies.append(reply_to_dict(reply))
    except TimeoutError as e:
        error = f"timeout: {e}"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    wall_ms = (time.perf_counter() - t0) * 1000
    merged = merge_replies(replies)
    rec: dict[str, Any] = {
        "id": case["id"], "category": case["category"], "variant": variant, "session_id": session_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "turns": case_turns(case), "reply": merged, "wall_ms": round(wall_ms),
        "error": error,
    }
    if error:
        rec.update(passed=False, checks={}, reasons=[error])
    else:
        rec.update(score_case(case, merged))
    return rec


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written line from an interrupted run
    return out


def latest_results_file(variant: str, results_dir: Path = RESULTS_DIR) -> Path | None:
    files = sorted(results_dir.glob(f"*-{variant}.jsonl"))
    return files[-1] if files else None


def run_variant(
    variant: str,
    cases: list[dict],
    out_path: Path,
    run_turn: RunTurn,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    judge_chat: Callable[[list[dict]], Any] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run cases in order, appending each result to out_path. Skips ids already in out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_results(out_path)}
    total = len(cases)
    for i, case in enumerate(cases, 1):
        if case["id"] in done:
            log(f"[{variant}] {i}/{total} {case['id']} skipped (already in results)")
            continue
        rec = run_case(case, variant, run_turn, timeout_s)
        if judge_chat and case["category"] == "answer" and not rec["error"]:
            rec["judge"] = judge_case(case, rec["reply"], judge_chat)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = "PASS" if rec["passed"] else "FAIL"
        extra = f" judge={'pass' if rec['judge']['pass'] else 'fail'}" if "judge" in rec else ""
        why = "" if rec["passed"] else f" - {rec['reasons'][0]}" if rec["reasons"] else ""
        log(f"[{variant}] {i}/{total} {case['id']} {status} {rec['wall_ms']}ms{extra}{why}")
    return read_results(out_path)


# ---- report ----

def _rec_tokens(rec: dict) -> tuple[int, int]:
    u = (rec.get("reply") or {}).get("usage") or {}
    return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


def _rec_latency(rec: dict) -> float:
    u = (rec.get("reply") or {}).get("usage") or {}
    return float(u.get("latency_ms") or rec.get("wall_ms") or 0)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def _count(recs: list[dict], pred: Callable[[dict], bool]) -> str:
    return fmt_rate(sum(1 for r in recs if pred(r)), len(recs))


def build_report(results: dict[str, list[dict]], cases: list[dict], prices: tuple[float, float, float]) -> str:
    """results: variant -> records. prices: (usd/Mtok in, usd/Mtok out, usd_to_inr)."""
    by_id = {c["id"]: c for c in cases}
    variants = [v for v in VARIANTS if v in results]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    L: list[str] = ["# Eval report", "", f"Generated {stamp}. Pass = every exact check on the case passed.", ""]

    L += ["## Pass rate by variant and category", ""]
    rows = []
    for v in variants:
        rs = results[v]
        rows.append([v] + [_count([r for r in rs if r["category"] == c], lambda r: r["passed"]) for c in CATEGORIES]
                    + [_count(rs, lambda r: r["passed"])])
    L += _table(["variant"] + CATEGORIES + ["overall"], rows) + [""]

    L += ["## Component accuracy", ""]
    rows = []
    for v in variants:
        rs = results[v]
        cit = [r for r in rs if "citations" in r.get("checks", {})]
        tool = [r for r in rs if "tool" in r.get("checks", {})]
        ref = [r for r in rs if "refusal" in r.get("checks", {})]
        fact = [r for r in rs if "facts" in r.get("checks", {})]
        rows.append([v, _count(fact, lambda r: r["checks"]["facts"]), _count(cit, lambda r: r["checks"]["citations"]),
                     _count(tool, lambda r: r["checks"]["tool"]), _count(ref, lambda r: r["checks"]["refusal"])])
    L += _table(["variant", "facts", "citations", "tool calls", "refusals"], rows) + [""]

    judged = {v: [r for r in results[v] if "judge" in r] for v in variants}
    if any(judged.values()):
        L += ["## LLM judge (answer cases, scored separately from exact checks)", ""]
        rows = []
        for v in variants:
            js = judged[v]
            agree = _count(js, lambda r: r["judge"]["pass"] == r["passed"])
            rows.append([v, _count(js, lambda r: r["judge"]["pass"]), _count(js, lambda r: r["passed"]), agree])
        L += _table(["variant", "judge pass", "exact pass (same cases)", "judge agrees with exact"], rows) + [""]

    L += ["## Latency, tokens, cost", ""]
    p_in, p_out, fx = prices
    rows = []
    for v in variants:
        rs = results[v]
        lat = [_rec_latency(r) for r in rs]
        toks = [_rec_tokens(r) for r in rs]
        n = max(len(rs), 1)
        cost = sum(cost_inr(a, b, p_in, p_out, fx) for a, b in toks) / n
        rows.append([v, f"{percentile(lat, 50) / 1000:.1f}s", f"{percentile(lat, 95) / 1000:.1f}s",
                     f"{sum(a for a, _ in toks) / n:.0f}", f"{sum(b for _, b in toks) / n:.0f}", f"INR {cost:.2f}"])
    L += _table(["variant", "p50", "p95", "prompt tok/query", "completion tok/query", "cost/query"], rows) + [""]

    L += ["## Failed cases", ""]
    any_fail = False
    for v in variants:
        fails = [r for r in results[v] if not r["passed"]]
        if not fails:
            continue
        any_fail = True
        L += [f"### {v} ({len(fails)} failed)", ""]
        for r in fails:
            q = (by_id.get(r["id"], {}).get("question") or " / ".join(r.get("turns", [])))[:70]
            why = "; ".join(r.get("reasons") or ["unknown"])[:160]
            L.append(f"- {r['id']} ({r['category']}) \"{q}\": {why}")
        L.append("")
    if not any_fail:
        L += ["None.", ""]
    return "\n".join(L)


def write_report(results_dir: Path, cases_path: Path, report_path: Path, variants: list[str]) -> Path:
    from app.config import settings

    results: dict[str, list[dict]] = {}
    for v in variants:
        f = latest_results_file(v, results_dir)
        if f:
            results[v] = read_results(f)
    cases = load_cases(cases_path)
    prices = (settings.price_in_per_mtok, settings.price_out_per_mtok, settings.usd_to_inr)
    report_path.write_text(build_report(results, cases, prices), encoding="utf-8")
    return report_path


# ---- CLI ----

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=[*VARIANTS, "all"], default="agent")
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", help="comma-separated case ids, e.g. A01,T03")
    p.add_argument("--judge", action="store_true", help="also score answer cases with the LLM judge")
    p.add_argument("--resume", action="store_true", help="append to the latest results file per variant, skipping done ids")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-case seconds (per turn)")
    p.add_argument("--report-only", action="store_true", help="rebuild the report from existing results files")
    p.add_argument("--cases", type=Path, default=CASES_PATH)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    return p.parse_args(argv)


def main(argv: list[str] | None = None, run_turn: RunTurn | None = None) -> int:
    args = parse_args(argv)
    # Freeze the agent's clock to the dataset date so return-window cases do not flip with the calendar
    # (found by the P5 review, 2026-11-10). CLOCK_NOW in the environment overrides.
    from app.config import settings

    settings.clock_now = settings.clock_now or EVAL_NOW
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if not args.report_only:
        cases = load_cases(args.cases, args.ids.split(",") if args.ids else None, args.limit)
        if not cases:
            print("No cases selected.", file=sys.stderr)
            return 2
        run_turn = run_turn or _default_run_turn()
        judge_chat = _default_chat() if args.judge else None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for v in variants:
            existing = latest_results_file(v, args.results_dir) if args.resume else None
            out = existing or args.results_dir / f"{stamp}-{v}.jsonl"
            print(f"== variant {v} -> {out}")
            run_variant(v, cases, out, run_turn, args.timeout, judge_chat)
    path = write_report(args.results_dir, args.cases, args.report, list(VARIANTS))
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        if "app.db" in sys.modules:  # close the connection pool cleanly if the agent opened one
            sys.modules["app.db"].reset_pool()
    raise SystemExit(code)
