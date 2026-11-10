"""Run one suite against one model and store every case result.

    python -m harness.run --suite support_agent --model fusion27b:iq3 [--limit N] [--resume]

The harness owns timing, token accounting, retries, per-case timeouts and storage. One bad case
(an exception, a timeout, a scorer bug) is recorded and the run carries on.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness import __version__, harden_console
from harness.adapters.base import Adapter, AdapterError, get_adapter, suite_names
from harness.models import (
    COMPLETE,
    ERROR,
    OK,
    PARTIAL,
    RUNNING,
    TIMEOUT,
    CaseResult,
    Run,
    now_iso,
    stable_hash,
)
from harness.stats import percentile
from harness.storage import DEFAULT_DB, RunNotFound, Store

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_RETRIES = 1
ORPHAN_WAIT_S = 60.0  # how long to wait for a timed-out case's thread before the next case starts
DEFAULT_ABORT_AFTER = 5  # consecutive errored or timed-out cases that stop a run (the service is down)
DEFAULT_USD_TO_INR = 88.0  # same default as project 1; only used when a price is set


class PreflightError(RuntimeError):
    """A live run was refused because the model or a backing service is not reachable."""


class ResumeMismatch(RuntimeError):
    """The unfinished run was made with different feature code, prompt or cases."""


@dataclass(frozen=True)
class Prices:
    """USD per million tokens, and the USD to INR rate. Local models are 0, 0."""

    usd_in_per_mtok: float = 0.0
    usd_out_per_mtok: float = 0.0
    usd_to_inr: float = DEFAULT_USD_TO_INR

    def cost_inr(self, prompt_tokens: int, completion_tokens: int) -> float:
        usd = (prompt_tokens * self.usd_in_per_mtok + completion_tokens * self.usd_out_per_mtok) / 1_000_000
        return usd * self.usd_to_inr


@dataclass
class RunSummary:
    run: Run
    results: list[CaseResult]
    ran: int = 0  # cases executed in this invocation
    skipped: int = 0  # cases already stored and kept (resume)
    resumed: bool = False
    notes: list[str] = field(default_factory=list)
    aborted: bool = False  # stopped early after repeated errors; resume later

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status != OK)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)


# ---- one case ----


class CaseTimeout(TimeoutError):
    """A call did not finish in time. `thread` is the worker still running it (it cannot be killed)."""

    def __init__(self, message: str, thread: threading.Thread | None = None):
        super().__init__(message)
        self.thread = thread


def call_with_timeout(fn: Callable[[], Any], timeout_s: float | None) -> Any:
    """Run fn in a daemon thread; raise CaseTimeout (a TimeoutError) if it does not finish in time.

    A thread cannot be killed: a timed-out call keeps running in the background until it ends on its
    own (the model client's own timeout). That is why timeouts are not retried automatically, and why the
    runner waits for the orphan, bounded, before the next case (see wait_for_orphans).
    """
    if not timeout_s or timeout_s <= 0:
        return fn()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - relayed to the caller
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise CaseTimeout(f"no result within {timeout_s:g}s", t)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _failed(
    case: dict, status: str, message: str, attempts: int, latency_ms: float, output: dict | None = None
) -> CaseResult:
    return CaseResult(
        case_id=case["id"],
        category=case.get("category", "uncategorised"),
        passed=False,
        status=status,
        reasons=[message],
        latency_ms=round(latency_ms, 1),
        attempts=attempts,
        output=output,
        error=message,
    )


def run_one(
    adapter: Adapter,
    case: dict,
    model: str,
    *,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    prices: Prices | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    orphans: list[threading.Thread] | None = None,
) -> CaseResult:
    """Run and score one case. Never raises. A timed-out worker thread is appended to `orphans`."""
    prices = prices or Prices()
    attempts = 0
    while True:
        attempts += 1
        t0 = clock()
        try:
            limit_s = adapter.case_timeout(case, timeout_s)
            output = call_with_timeout(lambda: adapter.run_case(case, model), limit_s)
        except TimeoutError as e:
            if orphans is not None and getattr(e, "thread", None) is not None:
                orphans.append(e.thread)
            return _failed(case, TIMEOUT, f"timeout: {e}", attempts, (clock() - t0) * 1000)
        except Exception as e:  # noqa: BLE001 - one bad case must not stop the run
            if attempts <= retries:
                sleep(min(2.0 * attempts, 10.0))
                continue
            return _failed(case, ERROR, f"{type(e).__name__}: {e}", attempts, (clock() - t0) * 1000)
        latency_ms = (clock() - t0) * 1000
        break

    if not isinstance(output, dict):
        return _failed(
            case, ERROR, f"adapter returned {type(output).__name__}, expected dict", attempts, latency_ms
        )
    usage = output.get("usage") or {}
    try:
        pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        pt = ct = 0
    try:
        scored = adapter.score(case, output)
        passed = bool(scored["passed"])
        checks = {str(k): bool(v) for k, v in (scored.get("checks") or {}).items()}
        reasons = [str(x) for x in (scored.get("reasons") or [])]
    except Exception as e:  # noqa: BLE001 - a scorer bug is a harness-side error, not a model failure
        return _failed(case, ERROR, f"scoring error: {type(e).__name__}: {e}", attempts, latency_ms, output)
    return CaseResult(
        case_id=case["id"],
        category=case.get("category", "uncategorised"),
        passed=passed,
        status=OK,
        checks=checks,
        reasons=reasons,
        latency_ms=round(latency_ms, 1),
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_inr=round(prices.cost_inr(pt, ct), 6),
        attempts=attempts,
        output=output,
    )


# ---- a whole run ----

HARNESS_KEY = "_harness"  # where the runner leaves flags inside a result's output (no schema change)


def flag_result(res: CaseResult, flag: str) -> None:
    """Mark a result, for example that it ran while a timed-out case's thread was still working."""
    out = dict(res.output) if isinstance(res.output, dict) else {}
    out[HARNESS_KEY] = {**out.get(HARNESS_KEY, {}), flag: True}
    res.output = out


def has_flag(res: CaseResult, flag: str) -> bool:
    return bool(isinstance(res.output, dict) and (res.output.get(HARNESS_KEY) or {}).get(flag))


def wait_for_orphans(
    threads: list[threading.Thread], limit_s: float, clock: Callable[[], float] = time.monotonic
) -> bool:
    """Wait up to limit_s in total for abandoned worker threads. True when none is still running."""
    deadline = clock() + max(0.0, limit_s)
    for t in threads:
        remaining = deadline - clock()
        if t.is_alive() and remaining > 0:
            t.join(remaining)
    threads[:] = [t for t in threads if t.is_alive()]
    return not threads


def validate_cases(cases: list[dict]) -> None:
    seen: set[str] = set()
    for c in cases:
        if "id" not in c:
            raise ValueError(f"case without an id: {str(c)[:80]}")
        if c["id"] in seen:
            raise ValueError(f"duplicate case id {c['id']}")
        seen.add(c["id"])


def select_cases(cases: list[dict], ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    if ids:
        wanted = set(ids)
        cases = [c for c in cases if c["id"] in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def fingerprint_of(prov: dict, all_cases: list[dict]) -> tuple:
    """What a run must share with another to be resumed or reused: feature version, prompt, cases, scorer, clock."""
    return (
        prov.get("git_hash"),
        prov.get("prompt_hash"),
        prov.get("code_hash"),
        stable_hash(all_cases),
        prov.get("scorer_hash"),
        (prov.get("config") or {}).get("clock_now"),
    )


def execute_run(
    adapter: Adapter,
    model: str,
    store: Store,
    *,
    suite: str | None = None,
    limit: int | None = None,
    ids: list[str] | None = None,
    resume: bool = False,
    run_id: str | None = None,
    label: str | None = None,
    timeout_s: float | None = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    prices: Prices | None = None,
    preflight: bool = True,
    warmup: bool = True,
    reuse_complete: bool = False,
    retry_errors: bool = True,
    abort_after: int | None = DEFAULT_ABORT_AFTER,
    orphan_wait_s: float = ORPHAN_WAIT_S,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> RunSummary:
    suite = suite or adapter.name
    prices = prices or Prices()
    all_cases = adapter.cases()
    validate_cases(all_cases)
    cases = select_cases(all_cases, ids, limit)
    if not cases:
        raise ValueError("no cases selected")
    prov = adapter.provenance()
    fingerprint = fingerprint_of(prov, all_cases)
    notes: list[str] = []

    if reuse_complete:  # matrix reruns: a finished run of this exact setup is not repeated
        done_run = store.find_complete(suite, model, fingerprint)
        if done_run and done_run.n_cases == len(cases):
            log(f"reusing complete run {done_run.run_id}")
            results = store.results(done_run.run_id)
            return RunSummary(
                run=done_run,
                results=results,
                skipped=len(results),
                resumed=True,
                notes=["reused a complete run with the same code, prompt and cases"],
            )

    run: Run | None = None
    if resume:
        if run_id:
            run = store.get_run(run_id)
            if run.suite != suite or run.model != model:
                raise ResumeMismatch(f"run {run.run_id} is {run.suite}/{run.model}, not {suite}/{model}")
            if run.fingerprint() != fingerprint:
                raise ResumeMismatch(
                    f"run {run.run_id} was made with different feature code, prompt or cases; start a new run"
                )
        else:
            run = store.find_resumable(suite, model, fingerprint)
            if run is None:
                stale = store.latest_run(suite, model, (RUNNING, PARTIAL))
                if stale:
                    notes.append(
                        f"unfinished run {stale.run_id} exists but the feature code, prompt or cases changed "
                        "since; starting a new run"
                    )
                else:
                    notes.append("nothing to resume; starting a new run")
    existing = {r.case_id: r for r in store.results(run.run_id)} if run else {}
    keep = {cid for cid, r in existing.items() if r.status == OK or not retry_errors}
    pending = [c for c in cases if c["id"] not in keep]
    pending_ids = {c["id"] for c in pending}

    adapter.configure(timeout_s=timeout_s)
    if preflight and pending:
        problems = adapter.preflight(model)
        if problems:
            raise PreflightError("; ".join(problems))
    if warmup and pending:
        try:
            adapter.warmup(model)  # untimed: model load must not land in the first case's latency
        except Exception as e:  # noqa: BLE001
            raise PreflightError(f"warm-up call failed: {type(e).__name__}: {e}") from e

    if run is None:
        started = now_iso()
        run = store.create_run(
            Run(
                run_id=run_id or store.new_run_id(suite, model, started),
                suite=suite,
                model=model,
                label=label,
                started_at=started,
                git_hash=prov.get("git_hash"),
                prompt_hash=prov.get("prompt_hash"),
                code_hash=prov.get("code_hash"),
                case_set_hash=stable_hash(all_cases),
                n_cases=len(cases),
                config={
                    "harness_version": __version__,
                    "timeout_s": timeout_s,
                    "retries": retries,
                    "limit": limit,
                    "ids": ids,
                    "n_cases_total": len(all_cases),
                    "scorer_hash": prov.get("scorer_hash"),
                    "git_dirty": prov.get("git_dirty"),
                    "synthetic": bool(adapter.synthetic),
                    **(prov.get("config") or {}),
                    "prices": {
                        "usd_in_per_mtok": prices.usd_in_per_mtok,
                        "usd_out_per_mtok": prices.usd_out_per_mtok,
                        "usd_to_inr": prices.usd_to_inr,
                    },
                },
            )
        )
    else:
        store.update_run(run.run_id, status=RUNNING, finished_at=None, n_cases=len(cases))
        run.status, run.finished_at, run.n_cases = RUNNING, None, len(cases)
    summary = RunSummary(
        run=run, results=[], skipped=len(cases) - len(pending), resumed=bool(existing), notes=notes
    )
    for n in notes:
        log(f"note: {n}")
    if summary.resumed:
        log(f"resuming {run.run_id}: {summary.skipped} cases kept, {len(pending)} to run")

    interrupted = False
    streak = 0
    orphans: list[threading.Thread] = []
    try:
        for i, case in enumerate(cases, 1):
            if case["id"] not in pending_ids:
                continue
            busy = False
            if orphans or adapter.orphan_threads():
                # a timed-out case may still be running in the background (a thread cannot be killed);
                # give it a bounded time to finish so it does not slow this case down
                threads = [*orphans, *adapter.orphan_threads()]
                if not wait_for_orphans(threads, orphan_wait_s):
                    busy = True
                    log(f"note: a timed-out case is still running in the background; {case['id']} is flagged")
                orphans[:] = threads
            res = run_one(
                adapter,
                case,
                model,
                timeout_s=timeout_s,
                retries=retries,
                prices=prices,
                clock=clock,
                sleep=sleep,
                orphans=orphans,
            )
            if busy:
                flag_result(res, "orphan_running")
            store.save_result(run.run_id, res)
            summary.ran += 1
            tag = "PASS" if res.passed else ("FAIL" if res.status == OK else res.status.upper())
            why = f" - {res.reasons[0][:100]}" if res.reasons and not res.passed else ""
            log(f"[{i}/{len(cases)}] {case['id']} {tag} {res.latency_ms / 1000:.1f}s{why}")
            streak = streak + 1 if res.status != OK else 0
            if abort_after and streak >= abort_after:
                summary.aborted = True
                log(
                    f"stopping: {streak} errored or timed-out cases in a row. The model or a service is probably "
                    "down. Fix it, then rerun with --resume."
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted: partial run saved; rerun with --resume to continue")
    finally:
        wanted = {c["id"] for c in cases}
        stored = {r.case_id: r for r in store.results(run.run_id) if r.case_id in wanted}
        done = len(stored) == len(wanted) and all(r.status == OK for r in stored.values())
        run.status = COMPLETE if done and not interrupted else PARTIAL
        run.finished_at = now_iso()
        store.update_run(run.run_id, status=run.status, finished_at=run.finished_at)
        summary.results = [stored[c["id"]] for c in cases if c["id"] in stored]
    if interrupted:
        raise KeyboardInterrupt
    return summary


def describe_summary(s: RunSummary) -> str:
    lat = [r.latency_ms for r in s.results if r.status == OK]
    parts = [
        f"run {s.run.run_id}: {s.passed}/{len(s.results)} passed",
        f"status {s.run.status}",
        f"{s.errors} errored or timed out" if s.errors else "no errors",
        f"p50 {percentile(lat, 50) / 1000:.1f}s p95 {percentile(lat, 95) / 1000:.1f}s"
        if lat
        else "no latency data",
    ]
    return ", ".join(parts)


# ---- CLI ----


def add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, help="run only the first N cases")
    p.add_argument("--ids", help="comma-separated case ids, e.g. A01,T03")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the latest unfinished run for this suite and model; errored cases are retried",
    )
    p.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per case (default %(default)s)"
    )
    p.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="retries after an exception, not after a timeout (default %(default)s)",
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite file (default %(default)s)")
    p.add_argument(
        "--price-in", type=float, default=0.0, help="USD per million prompt tokens (0 for local models)"
    )
    p.add_argument("--price-out", type=float, default=0.0, help="USD per million completion tokens")
    p.add_argument("--usd-to-inr", type=float, default=DEFAULT_USD_TO_INR)
    p.add_argument(
        "--abort-after",
        type=int,
        default=DEFAULT_ABORT_AFTER,
        help="stop after this many errored or timed-out cases in a row, 0 to never (default %(default)s)",
    )
    p.add_argument(
        "--no-preflight", action="store_true", help="skip the model and service reachability check"
    )
    p.add_argument(
        "--no-warmup", action="store_true", help="skip the untimed warm-up call after a model switch"
    )
    p.add_argument(
        "--reuse-complete",
        action="store_true",
        help="if a complete run with the same code, prompt and cases exists, use it and run nothing",
    )
    p.add_argument("--json-out", help="write {run_id, status, ...} to this file (used by harness.matrix)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m harness.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--suite", help=f"one of: {', '.join(suite_names())}")
    p.add_argument("--model", help="model name, e.g. fusion27b:iq3")
    p.add_argument("--run-id", help="use this run id (with --resume: the run to continue)")
    p.add_argument("--label", help="free-text label stored with the run")
    p.add_argument("--list-suites", action="store_true")
    add_run_options(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None, adapter: Adapter | None = None) -> int:
    harden_console()
    args = parse_args(argv)
    if args.list_suites:
        print("\n".join(suite_names()))
        return 0
    if not args.suite or not args.model:
        print("error: --suite and --model are required", file=sys.stderr)
        return 2
    try:
        adapter = adapter or get_adapter(args.suite)
    except (KeyError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    prices = Prices(args.price_in, args.price_out, args.usd_to_inr)
    try:
        with Store(args.db) as store:
            summary = execute_run(
                adapter,
                args.model,
                store,
                suite=args.suite,
                limit=args.limit,
                ids=args.ids.split(",") if args.ids else None,
                resume=args.resume,
                run_id=args.run_id,
                label=args.label,
                timeout_s=args.timeout,
                retries=args.retries,
                prices=prices,
                preflight=not args.no_preflight,
                warmup=not args.no_warmup,
                reuse_complete=args.reuse_complete,
                abort_after=args.abort_after,
            )
    except (PreflightError, ResumeMismatch, RunNotFound, AdapterError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        adapter.close()
    print(describe_summary(summary))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "run_id": summary.run.run_id,
                    "status": summary.run.status,
                    "suite": summary.run.suite,
                    "model": summary.run.model,
                    "passed": summary.passed,
                    "cases": len(summary.results),
                    "errors": summary.errors,
                }
            ),
            encoding="utf-8",
        )
    return 0 if summary.run.status == COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
