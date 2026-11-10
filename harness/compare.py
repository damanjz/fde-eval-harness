"""Compare two runs case by case, with statistics a non-statistician can act on.

    python -m harness.compare RUN_A RUN_B [--md FILE] [--html FILE]

RUN is a run id (or unique prefix) in the database, or a path to an exported run file (.json).
B is read as the newer or candidate run: positive deltas mean B is better than A.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from harness import harden_console
from harness.models import OK, CaseResult, Run
from harness.stats import (
    ALPHA,
    Bootstrap,
    McNemar,
    bootstrap_diff_ci,
    mcnemar_exact,
    mean,
    min_significant_net,
    percentile,
    wilson_interval,
)
from harness.storage import DEFAULT_DB, RunNotFound, load_run


@dataclass
class CategoryDelta:
    category: str
    n: int
    a_pass: int
    b_pass: int
    mcnemar: McNemar

    @property
    def delta(self) -> int:
        return self.b_pass - self.a_pass


@dataclass
class Flip:
    case_id: str
    category: str
    direction: str  # "regressed" (A passed, B failed) or "fixed" (A failed, B passed)
    a_reasons: list[str]
    b_reasons: list[str]
    a_status: str
    b_status: str


@dataclass
class Perf:
    p50_ms: float
    p95_ms: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_cost_inr: float
    total_cost_inr: float
    completed: int  # cases with status ok


@dataclass
class Comparison:
    a: Run
    b: Run
    n: int  # shared cases
    only_in_a: list[str]
    only_in_b: list[str]
    a_pass: int
    b_pass: int
    a_ci: tuple[float, float]
    b_ci: tuple[float, float]
    mcnemar: McNemar
    boot: Bootstrap
    min_net: int | None
    categories: list[CategoryDelta]
    flips: list[Flip]
    a_perf: Perf
    b_perf: Perf
    a_errors: int
    b_errors: int
    notes: list[str] = field(default_factory=list)
    synthetic: bool = False

    @property
    def n_same(self) -> int:
        return self.n - self.mcnemar.flips

    @property
    def net_cases(self) -> int:
        return self.b_pass - self.a_pass

    @property
    def verdict(self) -> str:
        m = self.mcnemar
        if m.flips == 0:
            return "identical"
        sig_p, sig_ci = m.significant, self.boot.excludes_zero
        if sig_p and sig_ci:
            return "better" if m.net > 0 else "worse"
        if sig_p != sig_ci:
            return "borderline"
        return "noise"


def perf_of(results: list[CaseResult]) -> Perf:
    ok = [r for r in results if r.status == OK]
    lat = [r.latency_ms for r in ok]
    return Perf(
        p50_ms=percentile(lat, 50),
        p95_ms=percentile(lat, 95),
        mean_prompt_tokens=mean([r.prompt_tokens for r in ok]),
        mean_completion_tokens=mean([r.completion_tokens for r in ok]),
        mean_cost_inr=mean([r.cost_inr for r in ok]),
        total_cost_inr=sum(r.cost_inr for r in ok),
        completed=len(ok),
    )


def provenance_notes(a: Run, b: Run) -> list[str]:
    """What differs between the two runs, so a delta can be attributed."""
    notes: list[str] = []
    if a.suite != b.suite:
        notes.append(f"Different suites ({a.suite} vs {b.suite}): case ids may not mean the same thing.")
    diffs = []
    if a.model != b.model:
        diffs.append(f"model ({a.model} vs {b.model})")
    for label, x, y in (
        ("prompt", a.prompt_hash, b.prompt_hash),
        ("feature code", a.code_hash, b.code_hash),
        ("git commit", a.git_hash, b.git_hash),
    ):
        if x and y and x != y:
            diffs.append(f"{label} ({x} vs {y})")
    for label, x, y in (
        ("variant", a.variant, b.variant),
        ("scoring code", a.scorer_hash, b.scorer_hash),
        ("frozen clock", a.clock_now, b.clock_now),
    ):
        if x != y and (x or y):
            diffs.append(f"{label} ({x or 'none'} vs {y or 'none'})")
    if a.case_set_hash and b.case_set_hash and a.case_set_hash != b.case_set_hash:
        notes.append(
            "The case definitions differ between the runs (case set hash changed); "
            "only shared case ids are compared."
        )
    if diffs:
        notes.append("Changed between the runs: " + "; ".join(diffs) + ".")
    elif a.model == b.model:
        notes.append(
            "Same model, prompt, code, scorer and clock: flips between these runs are pure run-to-run noise."
        )
    if a.config.get("source") or b.config.get("source"):
        notes.append("At least one run was imported from recorded results; see its source in the run file.")
    return notes


def coverage_problems(run: Run, results: list[CaseResult], who: str) -> list[str]:
    """A gate-grade run covered the whole case file. Partial runs, and runs that do not say, do not qualify."""
    problems: list[str] = []
    if not results:
        return [f"the {who} run has no results"]
    total = run.n_cases_total
    if total is None:
        problems.append(f"the {who} run does not record how many cases the case file has")
    elif len(results) != total:
        problems.append(f"the {who} run covers {len(results)} of {total} cases (a partial run)")
    if run.config.get("limit") or run.config.get("ids"):
        problems.append(f"the {who} run was made with --limit or --ids")
    if not run.case_set_hash:
        problems.append(f"the {who} run has no case-file hash")
    if not run.scorer_hash:
        problems.append(f"the {who} run has no scoring-code hash")
    return problems


def not_comparable(
    a_run: Run,
    a_results: list[CaseResult],
    b_run: Run,
    b_results: list[CaseResult],
    strict: bool = False,
) -> list[str]:
    """Reasons two runs cannot be compared (empty list = comparable).

    Always: the same suite, the same set of case ids, the same case file when both record it, and the same
    scoring code when both record it. With strict=True (the gate) also: each run covers the whole case file
    (so two partial runs cannot pass), each records its case-file and scoring-code hashes, and both ran under
    the same frozen clock (or both under the wall clock). A --limit run or a run against edited cases cannot
    pass a gate by luck of what was left out.
    """
    problems: list[str] = []
    if a_run.suite != b_run.suite:
        problems.append(f"different suites: {a_run.suite} vs {b_run.suite}")
    if strict:
        problems += coverage_problems(a_run, a_results, "baseline")
        problems += coverage_problems(b_run, b_results, "candidate")
    a_ids, b_ids = {r.case_id for r in a_results}, {r.case_id for r in b_results}
    if a_ids != b_ids:
        missing, extra = sorted(a_ids - b_ids), sorted(b_ids - a_ids)
        bits = []
        if missing:
            bits.append(
                f"{len(missing)} case(s) in the baseline but not the candidate ({', '.join(missing[:6])}"
                f"{'...' if len(missing) > 6 else ''})"
            )
        if extra:
            bits.append(
                f"{len(extra)} case(s) in the candidate but not the baseline ({', '.join(extra[:6])}"
                f"{'...' if len(extra) > 6 else ''})"
            )
        problems.append("different case-id sets: " + "; ".join(bits))
    if a_run.case_set_hash and b_run.case_set_hash and a_run.case_set_hash != b_run.case_set_hash:
        problems.append(f"different case files (hash {a_run.case_set_hash} vs {b_run.case_set_hash})")
    if a_run.scorer_hash and b_run.scorer_hash and a_run.scorer_hash != b_run.scorer_hash:
        problems.append(
            f"different scoring code (hash {a_run.scorer_hash} vs {b_run.scorer_hash}): "
            "the same reply could score differently"
        )
    if strict and a_run.clock_now != b_run.clock_now:
        problems.append(
            f"different clocks (baseline {a_run.clock_now or 'wall clock'}, candidate "
            f"{b_run.clock_now or 'wall clock'}): date arithmetic in tools gives different answers"
        )
    return problems


def compare_runs(
    a_run: Run,
    a_results: list[CaseResult],
    b_run: Run,
    b_results: list[CaseResult],
    *,
    seed: int = 0,
    n_boot: int = 100_000,
) -> Comparison:
    a_by = {r.case_id: r for r in a_results}
    b_by = {r.case_id: r for r in b_results}
    shared = [r.case_id for r in a_results if r.case_id in b_by]  # A's order
    only_a = [cid for cid in a_by if cid not in b_by]
    only_b = [cid for cid in b_by if cid not in a_by]
    a_pass_list = [a_by[c].passed for c in shared]
    b_pass_list = [b_by[c].passed for c in shared]

    flips: list[Flip] = []
    for cid in shared:
        x, y = a_by[cid], b_by[cid]
        if x.passed != y.passed:
            flips.append(
                Flip(
                    cid,
                    y.category or x.category,
                    "regressed" if x.passed else "fixed",
                    x.reasons,
                    y.reasons,
                    x.status,
                    y.status,
                )
            )
    only_a_n = sum(1 for f in flips if f.direction == "regressed")
    only_b_n = sum(1 for f in flips if f.direction == "fixed")
    m = mcnemar_exact(only_a_n, only_b_n)

    cats: dict[str, list[str]] = {}
    for cid in shared:
        cats.setdefault(b_by[cid].category or a_by[cid].category, []).append(cid)
    categories = []
    for cat, ids in sorted(cats.items()):
        ra = sum(1 for c in ids if a_by[c].passed and not b_by[c].passed)
        rb = sum(1 for c in ids if b_by[c].passed and not a_by[c].passed)
        categories.append(
            CategoryDelta(
                cat,
                len(ids),
                sum(a_by[c].passed for c in ids),
                sum(b_by[c].passed for c in ids),
                mcnemar_exact(ra, rb),
            )
        )

    a_shared = [a_by[c] for c in shared]
    b_shared = [b_by[c] for c in shared]
    notes = provenance_notes(a_run, b_run)
    if only_a:
        notes.append(
            f"{len(only_a)} case(s) only in A (ignored): {', '.join(only_a[:8])}{'...' if len(only_a) > 8 else ''}"
        )
    if only_b:
        notes.append(
            f"{len(only_b)} case(s) only in B (ignored): {', '.join(only_b[:8])}{'...' if len(only_b) > 8 else ''}"
        )
    a_err = sum(1 for r in a_shared if r.status != OK)
    b_err = sum(1 for r in b_shared if r.status != OK)
    if a_err or b_err:
        notes.append(
            f"Errored or timed-out cases count as failures: A has {a_err}, B has {b_err}. "
            "Re-run them with --resume before trusting the comparison."
        )
    n = len(shared)
    return Comparison(
        a=a_run,
        b=b_run,
        n=n,
        only_in_a=only_a,
        only_in_b=only_b,
        a_pass=sum(a_pass_list),
        b_pass=sum(b_pass_list),
        a_ci=wilson_interval(sum(a_pass_list), n),
        b_ci=wilson_interval(sum(b_pass_list), n),
        mcnemar=m,
        boot=bootstrap_diff_ci(a_pass_list, b_pass_list, n_boot=n_boot, seed=seed),
        min_net=min_significant_net(m.flips),
        categories=categories,
        flips=flips,
        a_perf=perf_of(a_shared),
        b_perf=perf_of(b_shared),
        a_errors=a_err,
        b_errors=b_err,
        notes=notes,
        synthetic=bool(a_run.config.get("synthetic") or b_run.config.get("synthetic")),
    )


# ---- text ----


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def points(x: float) -> str:
    return f"{x * 100:+.1f}"


def _signed(x: float, unit: str = "", digits: int = 1) -> str:
    if round(x, digits) == 0:
        return f"{0:.{digits}f}{unit}"
    return f"{x:+.{digits}f}{unit}"


def _rel(a: float, b: float) -> str:
    return f"{(b - a) / a * 100:+.0f}%" if a else "-"


VERDICT_TEXT = {
    "identical": "IDENTICAL: no case flipped.",
    "better": "B IS BETTER, beyond noise: both tests agree.",
    "worse": "B IS WORSE, beyond noise: both tests agree.",
    "borderline": "BORDERLINE: the two tests disagree. Treat the difference as weak evidence.",
    "noise": "WITHIN NOISE: the difference is too small to tell from chance.",
}


def explain(c: Comparison) -> list[str]:
    """The comparison in plain words. Each item is one short paragraph."""
    m, b = c.mcnemar, c.boot
    out = [
        f"Both runs were scored on the same {c.n} cases. {c.n_same} cases got the same result in both runs, "
        f"so they say nothing about which run is better. The other {m.flips} cases flipped."
    ]
    if m.flips == 0:
        out.append("Nothing flipped, so there is nothing to test: the runs agree on every case.")
        return out
    out.append(
        f"{m.only_b} case(s) went from fail to pass (B better) and {m.only_a} went from pass to fail (B worse): "
        f"net {_signed(m.net, ' case(s)', 0)}. If the runs were equally good the flips would split about evenly "
        f"({m.flips / 2:g} each way expected); the net {abs(m.net)} is how far this split is from even."
    )
    if m.p_value >= 0.999:
        chance = "essentially always"
    elif m.p_value < 0.001:
        chance = "less than 0.1% of the time"
    else:
        chance = (
            f"about {m.p_value * 100:.0f}% of the time"
            if m.p_value >= 0.01
            else f"about {m.p_value * 100:.1f}% of the time"
        )
    out.append(
        f"McNemar exact test, p = {fmt_p(m.p_value)}. If the two runs were really equally good, each flip would be a "
        f"coin toss, and a split at least this uneven ({m.only_a} vs {m.only_b}) would happen {chance} by luck. "
        f"Below {ALPHA} is conventionally called significant."
    )
    if c.min_net is None:
        out.append(
            f"With only {m.flips} flips, no split can reach p < {ALPHA}: this much data cannot settle it."
        )
    else:
        out.append(
            f"With {m.flips} flips, the net change has to be at least {c.min_net} case(s) before it is "
            f"unlikely to be luck. Here it is {abs(m.net)}."
        )
    how = (
        f"every possible resample of the {c.n} cases was enumerated exactly (no random draws)"
        if b.exact
        else f"resampling the {c.n} cases {b.n_boot:,} times (seed {b.seed})"
    )
    out.append(
        f"Bootstrap check: {how}. Accuracy of B minus A is {points(b.diff)} points with a 95% interval of "
        f"{points(b.low)} to {points(b.high)}. "
        + (
            "The interval excludes 0."
            if b.excludes_zero
            else "The interval includes 0, so the true difference could be nothing."
        )
    )
    out.append(
        "Limits: one run per model. LLM sampling can flip a case between two runs of the same model, so compare "
        "a model with itself once to see the noise floor. Per-category rows are small samples; read their "
        "p-values as hints, and expect a few false alarms when many categories are checked."
    )
    return out


def noise_floor_lines(c: Comparison) -> list[str]:
    """For two runs of one configuration, run in this order: how much flipping and net swing is pure noise."""
    m = c.mcnemar
    same_config = c.a.model == c.b.model and not any(
        x and y and x != y
        for x, y in (
            (c.a.prompt_hash, c.b.prompt_hash),
            (c.a.code_hash, c.b.code_hash),
            (c.a.git_hash, c.b.git_hash),
            (c.a.scorer_hash, c.b.scorer_hash),
            (c.a.clock_now, c.b.clock_now),
        )
    )
    lines: list[str] = []
    if not same_config:
        lines.append(
            "WARNING: these runs differ in model, prompt, code, scorer or clock, so this is not a noise floor."
        )
    allowance = m.flips if c.min_net is None else c.min_net - 1
    drop = max(0, c.a_pass - c.b_pass)
    lines += [
        f"Same configuration, run twice: {m.flips} of {c.n} cases flipped ({m.only_a} pass to fail, "
        f"{m.only_b} fail to pass), a net swing of {abs(m.net)} case(s). Nothing changed between the runs, "
        "so all of it is noise.",
        f"Seen from a gate (first run as baseline, second as candidate), the overall drop was {drop} case(s): "
        f"a --max-drop of {drop} or more would have let this pair through. Noise can swing the other way too.",
        f"Statistically, with {m.flips} flips a net change of up to {allowance} case(s) cannot be told from "
        f"noise (p >= {ALPHA}). Take the larger of {drop} and {allowance} ({max(drop, allowance)}) as a "
        "starting point for --max-drop. It is one pair of runs, a small sample of the noise: repeat the rerun "
        "to firm it up.",
        "Per category (the cost of a strict per-case must-pass; compare with --rerun-lost K in harness.gate):",
    ]
    for d in c.categories:
        lost = d.mcnemar.only_a
        flips = d.mcnemar.flips
        if d.a_pass:
            rate = lost / d.a_pass
            block = 1 - (1 - rate) ** d.a_pass
            if lost:
                est = (
                    f"{lost} of {d.a_pass} cases that passed in the first run failed in the second "
                    f"({100 * rate:.0f}%), so a strict per-case gate on this category would have blocked this "
                    f"pair; if that rate holds, about {100 * block:.0f}% of reruns of an unchanged "
                    "configuration would be blocked"
                )
            else:
                est = (
                    f"none of the {d.a_pass} cases that passed in the first run failed in the second; one pair "
                    f"only bounds the per-case rate below about {100 * min(1.0, 3 / d.a_pass):.0f}% (rule of "
                    "three), so the false-block rate is bounded, not measured"
                )
        else:
            est = "no case passed in the first run, so nothing could be lost"
        lines.append(f"{d.category}: {flips} of {d.n} cases flipped; {est}.")
    return lines


def render_markdown(c: Comparison, noise_floor: bool = False) -> str:
    L: list[str] = [f"# Compare: {c.a.run_id}  vs  {c.b.run_id}", ""]
    if c.synthetic:
        L += [
            "> SYNTHETIC: at least one run comes from the fake adapter. These numbers are not a real model's.",
            "",
        ]
    for tag, r in (("A", c.a), ("B", c.b)):
        bits = [r.suite, r.model, r.started_at[:16].replace("T", " ")]
        for k, v in (
            ("variant", r.variant),
            ("git", r.git_hash),
            ("prompt", r.prompt_hash),
            ("code", r.code_hash),
            ("scorer", r.scorer_hash),
            ("clock", r.clock_now),
        ):
            if v:
                bits.append(f"{k} {v}")
        if r.label:
            bits.append(f'"{r.label}"')
        L.append(f"- {tag}: " + ", ".join(bits))
    L += ["", "## Verdict", "", VERDICT_TEXT[c.verdict], ""]
    L += ["| | A | B | change (B - A) |", "|---|---|---|---|"]
    L.append(
        f"| accuracy | {pct(c.a_pass, c.n)} | {pct(c.b_pass, c.n)} | {_signed(c.net_cases, ' case(s)', 0)}, "
        f"{points(c.boot.diff)} points |"
    )
    L.append(
        f"| 95% interval (Wilson) | {100 * c.a_ci[0]:.0f}% to {100 * c.a_ci[1]:.0f}% | "
        f"{100 * c.b_ci[0]:.0f}% to {100 * c.b_ci[1]:.0f}% | bootstrap {points(c.boot.low)} to {points(c.boot.high)} points |"
    )
    L.append(
        f"| flips | {c.mcnemar.only_a} worse | {c.mcnemar.only_b} better | McNemar p = {fmt_p(c.mcnemar.p_value)} |"
    )
    L.append(f"| errored or timed out | {c.a_errors} | {c.b_errors} | |")
    L += ["", "## In plain words", ""]
    L += [f"- {line}" for line in explain(c)]

    L += [
        "",
        "## By category",
        "",
        "| category | cases | A | B | change | flips (better / worse) | p | reading |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in c.categories:
        m = d.mcnemar
        reading = "no flips" if m.flips == 0 else ("significant" if m.significant else "weak: within noise")
        L.append(
            f"| {d.category} | {d.n} | {pct(d.a_pass, d.n)} | {pct(d.b_pass, d.n)} | {_signed(d.delta, '', 0)} | "
            f"{m.only_b} / {m.only_a} | {fmt_p(m.p_value)} | {reading} |"
        )

    L += ["", "## Flipped cases", ""]
    if not c.flips:
        L += ["None.", ""]
    else:
        L += ["| case | category | direction | why it failed |", "|---|---|---|---|"]
        for f in sorted(c.flips, key=lambda f: (f.direction != "regressed", f.case_id)):
            arrow = "pass -> fail" if f.direction == "regressed" else "fail -> pass"
            why = "; ".join(f.b_reasons if f.direction == "regressed" else f.a_reasons) or "-"
            L.append(f"| {f.case_id} | {f.category} | {arrow} | {why[:140].replace('|', '/')} |")
        L.append("")

    pa, pb = c.a_perf, c.b_perf
    L += [
        "## Latency, tokens, cost",
        "",
        "| | A | B | change |",
        "|---|---|---|---|",
        f"| p50 latency | {pa.p50_ms / 1000:.1f}s | {pb.p50_ms / 1000:.1f}s | {_rel(pa.p50_ms, pb.p50_ms)} |",
        f"| p95 latency | {pa.p95_ms / 1000:.1f}s | {pb.p95_ms / 1000:.1f}s | {_rel(pa.p95_ms, pb.p95_ms)} |",
        f"| prompt tokens / case | {pa.mean_prompt_tokens:.0f} | {pb.mean_prompt_tokens:.0f} | "
        f"{_rel(pa.mean_prompt_tokens, pb.mean_prompt_tokens)} |",
        f"| completion tokens / case | {pa.mean_completion_tokens:.0f} | {pb.mean_completion_tokens:.0f} | "
        f"{_rel(pa.mean_completion_tokens, pb.mean_completion_tokens)} |",
        f"| cost / case | INR {pa.mean_cost_inr:.2f} | INR {pb.mean_cost_inr:.2f} | "
        f"{_signed(pb.mean_cost_inr - pa.mean_cost_inr, '', 2)} |",
        "",
        "Latency is wall time per case measured by the harness, over cases that completed.",
        "",
    ]
    if noise_floor:
        L += ["## Noise floor", ""] + [f"- {n}" for n in noise_floor_lines(c)] + [""]
    if c.notes:
        L += ["## Notes", ""] + [f"- {n}" for n in c.notes] + [""]
    return "\n".join(L)


# ---- CLI ----


def main(argv: list[str] | None = None) -> int:
    harden_console()
    p = argparse.ArgumentParser(
        prog="python -m harness.compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_a", metavar="RUN_A")
    p.add_argument("run_b", metavar="RUN_B")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--md", help="also write the markdown to this file")
    p.add_argument("--html", help="also write a standalone HTML report to this file")
    p.add_argument(
        "--noise-floor",
        action="store_true",
        help="RUN_A and RUN_B are the same configuration run twice: report the noise floor",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="seed for the Monte Carlo fallback (the exact bootstrap has none)"
    )
    p.add_argument(
        "--boot",
        type=int,
        default=100_000,
        help="Monte Carlo resamples, used only above 1500 cases (default %(default)s)",
    )
    args = p.parse_args(argv)
    try:
        ra, res_a = load_run(args.run_a, args.db)
        rb, res_b = load_run(args.run_b, args.db)
    except (RunNotFound, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cmp = compare_runs(ra, res_a, rb, res_b, seed=args.seed, n_boot=args.boot)
    text = render_markdown(cmp, noise_floor=args.noise_floor)
    print(text)
    if args.md:
        from pathlib import Path

        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(text + "\n", encoding="utf-8")
    if args.html:
        from pathlib import Path

        from harness.report import compare_html

        Path(args.html).parent.mkdir(parents=True, exist_ok=True)
        Path(args.html).write_text(compare_html(cmp, noise_floor=args.noise_floor), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
