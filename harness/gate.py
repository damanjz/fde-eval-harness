"""Regression gate: exit non-zero when a candidate run is worse than the baseline.

    python -m harness.gate --baseline RUN --candidate RUN --max-drop 2 --must-pass refuse

Exit codes:
  0  candidate passes
  1  regression: overall passes drop by more than --max-drop cases, or a must-pass category loses a case
  2  not comparable or unusable: a different suite, case-id set, case file, scoring code or frozen clock; a run
     that does not cover the whole case file (a --limit run, or two partial runs); a run with no results or no
     hashes; an unknown must-pass category; an unreadable or malformed run. Any crash is exit 2 as well:
     exit 1 only ever means a real regression.

A must-pass category "loses a case" when any case that passed in the baseline fails in the candidate.
That is stricter than comparing counts, because a fix elsewhere in the category cannot offset a break;
use --count-only to compare counts instead. Errored and timed-out cases count as failures, never as
skipped. The McNemar line is context for a human reader and never changes the exit code.

Strict per-case must-pass blocks on noise too: a case that passed once can fail on a rerun of an unchanged
configuration. --rerun-lost K reruns each lost must-pass case K times on the candidate's model (live, local
only) and clears it when it passes in a strict majority of the reruns; a case that fails most reruns is a
confirmed loss. See docs/runbook.md for the expected false-block rate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from harness import harden_console
from harness.adapters.base import Adapter, AdapterError, get_adapter
from harness.compare import Comparison, compare_runs, fmt_p, not_comparable, pct
from harness.models import OK
from harness.run import DEFAULT_TIMEOUT_S, PreflightError, fingerprint_of, run_one
from harness.storage import DEFAULT_DB, RunNotFound, load_run

PASS, BLOCK, INVALID = 0, 1, 2
DEFAULT_MAX_DROP = 2  # the spec's example value; the measured noise floor recommends 4 (docs/runbook.md)

Rerun = Callable[[str], list[bool]]  # case id -> pass or fail of each rerun


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class GateResult:
    exit_code: int
    checks: list[Check] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # reasons for exit 2
    comparison: Comparison | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return {PASS: "PASS", BLOCK: "BLOCKED", INVALID: "NOT COMPARABLE"}[self.exit_code]


def evaluate_gate(
    baseline: tuple,
    candidate: tuple,
    *,
    max_drop: int = DEFAULT_MAX_DROP,
    must_pass: list[str] | None = None,
    count_only: bool = False,
    rerun: Rerun | None = None,
) -> GateResult:
    """baseline and candidate are (Run, results) pairs as returned by load_run.

    rerun(case_id) reruns one case on the candidate and returns the pass or fail of each rerun; it is used
    only for lost must-pass cases in per-case mode.
    """
    (a_run, a_res), (b_run, b_res) = baseline, candidate
    must_pass = [c.strip() for c in (must_pass or []) if c.strip()]
    if max_drop < 0:
        return GateResult(INVALID, problems=["--max-drop must be 0 or more"])
    problems = not_comparable(a_run, a_res, b_run, b_res, strict=True)
    known = {r.category for r in a_res}
    unknown = [c for c in must_pass if c not in known]
    if unknown:
        problems.append(
            f"must-pass category not in the runs: {', '.join(unknown)} (known: {', '.join(sorted(known)) or 'none'})"
        )
    if problems:
        return GateResult(INVALID, problems=problems)

    cmp = compare_runs(a_run, a_res, b_run, b_res)
    res = GateResult(PASS, comparison=cmp)

    drop = cmp.a_pass - cmp.b_pass
    res.checks.append(
        Check(
            "overall",
            drop <= max_drop,
            f"baseline {pct(cmp.a_pass, cmp.n)}, candidate {pct(cmp.b_pass, cmp.n)}: "
            f"{'drop of ' + str(drop) if drop > 0 else 'no drop'}, allowed {max_drop}",
        )
    )
    by_cat = {d.category: d for d in cmp.categories}
    for cat in must_pass:
        d = by_cat[cat]
        regressed = sorted(f.case_id for f in cmp.flips if f.category == cat and f.direction == "regressed")
        if count_only:
            ok = d.b_pass >= d.a_pass
            detail = f"baseline {pct(d.a_pass, d.n)}, candidate {pct(d.b_pass, d.n)} (counts compared)"
        else:
            detail = f"baseline {pct(d.a_pass, d.n)}, candidate {pct(d.b_pass, d.n)}"
            if regressed and rerun:
                confirmed = []
                bits = []
                for cid in regressed:
                    outcomes = rerun(cid)
                    cleared = sum(outcomes) * 2 > len(outcomes)
                    bits.append(
                        f"{cid} (reran {len(outcomes)}x: {sum(outcomes)} passed, "
                        f"{'cleared as noise' if cleared else 'confirmed lost'})"
                    )
                    if not cleared:
                        confirmed.append(cid)
                detail += "; lost in the candidate run: " + ", ".join(bits)
                ok = not confirmed
            else:
                ok = not regressed
                detail += f"; lost: {', '.join(regressed)}" if regressed else "; no case lost"
        res.checks.append(Check(f"must-pass: {cat}", ok, detail))
    if any(not c.ok for c in res.checks):
        res.exit_code = BLOCK

    m = cmp.mcnemar
    if m.flips:
        reading = "significant" if m.significant else "within noise"
        res.notes.append(
            f"Context only: {m.only_a} case(s) got worse, {m.only_b} got better; McNemar p = {fmt_p(m.p_value)} "
            f"({reading}). This does not change the exit code."
        )
    if cmp.b_errors or cmp.a_errors:
        res.notes.append(
            f"Errored or timed-out cases count as failures: baseline {cmp.a_errors}, candidate {cmp.b_errors}."
        )
    res.notes.extend(n for n in cmp.notes if n.startswith("Changed between"))
    return res


def describe(run) -> str:
    bits = [run.suite, run.model]
    if run.variant:
        bits.append(f"variant {run.variant}")
    if run.clock_now:
        bits.append(f"clock {run.clock_now}")
    return ", ".join(bits)


def render(res: GateResult, baseline: tuple, candidate: tuple) -> str:
    (a_run, _), (b_run, _) = baseline, candidate
    L = [
        f"# Gate: {res.verdict} (exit {res.exit_code})",
        "",
        f"- baseline: {a_run.run_id} ({describe(a_run)})",
        f"- candidate: {b_run.run_id} ({describe(b_run)})",
        "",
    ]
    if res.problems:
        L += ["The runs cannot be gated against each other:"] + [f"- {p}" for p in res.problems]
        L += [
            "",
            "Run both on the full, unchanged case set, with the same suite, scoring code and clock, then gate again.",
        ]
        return "\n".join(L)
    L += ["| check | result | detail |", "|---|---|---|"]
    L += [f"| {c.name} | {'ok' if c.ok else 'BLOCK'} | {c.detail} |" for c in res.checks]
    if res.notes:
        L += [""] + [f"- {n}" for n in res.notes]
    return "\n".join(L)


def make_rerunner(
    candidate_run,
    k: int,
    adapter: Adapter | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    preflight: bool = True,
) -> Rerun:
    """Build a rerun(case_id) for the candidate's suite and model. Live: it calls the model.

    Refuses when the feature has changed since the candidate run (the reruns would not test the candidate).
    """
    adapter = adapter or get_adapter(candidate_run.suite)
    cases = adapter.cases()
    fp = fingerprint_of(adapter.provenance(), cases)
    if fp != candidate_run.fingerprint():
        raise PreflightError(
            "the feature (code, prompt, cases, scorer or clock) changed since the candidate run, so reruns "
            "would not test the candidate; rerun the whole suite instead"
        )
    adapter.configure(timeout_s=timeout_s)
    if preflight:
        problems = adapter.preflight(candidate_run.model)
        if problems:
            raise PreflightError("; ".join(problems))
        adapter.warmup(candidate_run.model)
    by_id = {c["id"]: c for c in cases}

    def rerun(case_id: str) -> list[bool]:
        out = []
        for _ in range(k):
            r = run_one(adapter, by_id[case_id], candidate_run.model, timeout_s=timeout_s, retries=0)
            out.append(r.status == OK and r.passed)
        return out

    return rerun


def main(argv: list[str] | None = None, adapter: Adapter | None = None) -> int:
    harden_console()
    p = argparse.ArgumentParser(
        prog="python -m harness.gate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", required=True, metavar="RUN", help="run id or run file")
    p.add_argument("--candidate", required=True, metavar="RUN", help="run id or run file")
    p.add_argument(
        "--max-drop",
        type=int,
        default=DEFAULT_MAX_DROP,
        help="largest allowed overall drop in passed cases (default %(default)s, the spec's example; "
        "the measured noise floor recommends 4, see docs/runbook.md)",
    )
    p.add_argument("--must-pass", default="", help="comma-separated categories that may lose no case")
    p.add_argument(
        "--count-only", action="store_true", help="compare must-pass counts instead of individual cases"
    )
    p.add_argument(
        "--rerun-lost",
        type=int,
        default=0,
        metavar="K",
        help="rerun each lost must-pass case K times on the candidate's model (live) and clear it when it "
        "passes in a strict majority; 0 (default) blocks on the first loss",
    )
    p.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per rerun (default %(default)s)"
    )
    p.add_argument("--no-preflight", action="store_true", help="skip the reachability check before reruns")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--md", help="also write the result as markdown to this file (e.g. $GITHUB_STEP_SUMMARY)")
    args = p.parse_args(argv)
    try:
        return _run(args, adapter)
    except (RunNotFound, ValueError, OSError, PreflightError, AdapterError) as e:
        print(f"error: {e}", file=sys.stderr)
        return INVALID
    except Exception as e:  # noqa: BLE001 - a crash must never look like a regression (exit 1)
        print(f"error: internal error ({type(e).__name__}: {e})", file=sys.stderr)
        return INVALID


def _run(args: argparse.Namespace, adapter: Adapter | None) -> int:
    baseline = load_run(args.baseline, args.db)
    candidate = load_run(args.candidate, args.db)
    must_pass = args.must_pass.split(",") if args.must_pass else []
    rerun = None
    if args.rerun_lost > 0 and not args.count_only:
        built: dict[str, Rerun] = {}

        def rerun(case_id: str) -> list[bool]:
            if (
                "fn" not in built
            ):  # built on first use: no live check unless a must-pass case was actually lost
                built["fn"] = make_rerunner(
                    candidate[0], args.rerun_lost, adapter, args.timeout, preflight=not args.no_preflight
                )
            return built["fn"](case_id)

    res = evaluate_gate(
        baseline,
        candidate,
        max_drop=args.max_drop,
        must_pass=must_pass,
        count_only=args.count_only,
        rerun=rerun,
    )
    text = render(res, baseline, candidate)
    print(text)
    if args.md:
        with Path(args.md).open("a", encoding="utf-8") as f:  # append: $GITHUB_STEP_SUMMARY collects several
            f.write(text + "\n\n")
    return res.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
