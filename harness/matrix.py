"""Model matrix: run one suite against several models and write reports/matrix.md and reports/matrix.html.

    python -m harness.matrix --suite support_agent --models fusion27b:iq3,fusion27b:q4ks,qwen3:14b

Each (suite, model) runs in its own subprocess (`python -m harness.run`): feature repos such as
projects 1 and 4 both have top-level `app` and `evals` packages and must never share an interpreter,
and a fresh process per model gives the untimed warm-up call a clean start. Models run one after
another because the GPU is shared. --resume continues unfinished runs and reuses finished ones whose
code, prompt and cases are unchanged. --report-only rebuilds the reports from the database.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from harness import harden_console
from harness.compare import Comparison, Perf, compare_runs, fmt_p, not_comparable, pct, perf_of, points
from harness.models import COMPLETE, OK, CaseResult, Run
from harness.run import DEFAULT_RETRIES, DEFAULT_TIMEOUT_S, has_flag
from harness.stats import wilson_interval
from harness.storage import DEFAULT_DB, RunNotFound, Store

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


@dataclass
class ModelStats:
    model: str
    run: Run
    results: list[CaseResult]
    passed: int
    n: int
    ci: tuple[float, float]
    by_cat: dict[str, tuple[int, int]]  # category -> (passed, cases)
    perf: Perf
    errors: int

    @property
    def accuracy(self) -> float:
        return self.passed / self.n if self.n else 0.0


@dataclass
class MatrixData:
    suite: str
    models: list[ModelStats]
    categories: list[str]
    case_ids: list[str]  # shared by every model, in run order
    case_category: dict[str, str]
    pairs: list[Comparison]
    generated: str
    notes: list[str] = field(default_factory=list)
    synthetic: bool = False


def build_matrix(
    suite: str, runs: list[tuple[Run, list[CaseResult]]], generated: str | None = None
) -> MatrixData:
    """runs in the order the models were given. Only cases present in every run are compared."""
    if len(runs) < 1:
        raise ValueError("no runs to compare")
    notes: list[str] = []
    shared = [r.case_id for r in runs[0][1]]
    for _, res in runs[1:]:
        ids = {r.case_id for r in res}
        shared = [c for c in shared if c in ids]
    for run, res in runs:
        problems = not_comparable(runs[0][0], runs[0][1], run, res)
        if problems:
            notes.append(f"{run.model}: " + "; ".join(problems) + ". Only shared cases are used.")
    for label, attr in (("git commit", "git_hash"), ("prompt", "prompt_hash"), ("feature code", "code_hash")):
        values = {getattr(r, attr) for r, _ in runs if getattr(r, attr)}
        if len(values) > 1:
            notes.append(
                f"The models ran against different {label} ({', '.join(sorted(values))}): "
                "differences between them mix the model with a change to the feature. Rerun them on one version."
            )
    clocks = {r.clock_now for r, _ in runs}
    if len(clocks) > 1:
        notes.append(
            "The models ran under different frozen clocks ("
            + ", ".join(sorted(c or "wall clock" for c in clocks))
            + "): date-based cases can differ for that reason alone."
        )
    for run, res in runs:
        busy = sum(1 for r in res if has_flag(r, "orphan_running"))
        if busy:
            notes.append(
                f"{run.model}: {busy} case(s) ran while a timed-out case was still working in the background; "
                "their latency may be inflated."
            )
    shared_set = set(shared)
    stats: list[ModelStats] = []
    cat_of: dict[str, str] = {}
    for run, res in runs:
        kept = [r for r in res if r.case_id in shared_set]
        for r in kept:
            cat_of.setdefault(r.case_id, r.category)
        by_cat: dict[str, list[int]] = {}
        for r in kept:
            tally = by_cat.setdefault(r.category, [0, 0])
            tally[0] += int(r.passed)
            tally[1] += 1
        passed = sum(1 for r in kept if r.passed)
        stats.append(
            ModelStats(
                model=run.model,
                run=run,
                results=kept,
                passed=passed,
                n=len(kept),
                ci=wilson_interval(passed, len(kept)),
                by_cat={k: (v[0], v[1]) for k, v in by_cat.items()},
                perf=perf_of(kept),
                errors=sum(1 for r in kept if r.status != OK),
            )
        )
        if run.status != COMPLETE:
            notes.append(
                f"{run.model}: run {run.run_id} is {run.status}; errored or missing cases count as failures."
            )
    categories = sorted({c for s in stats for c in s.by_cat})
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairs.append(
                compare_runs(
                    runs[i][0],
                    [r for r in runs[i][1] if r.case_id in shared_set],
                    runs[j][0],
                    [r for r in runs[j][1] if r.case_id in shared_set],
                )
            )
    return MatrixData(
        suite=suite,
        models=stats,
        categories=categories,
        case_ids=shared,
        case_category=cat_of,
        pairs=pairs,
        generated=generated or datetime.now().strftime("%Y-%m-%d %H:%M"),
        notes=notes,
        synthetic=any(r.config.get("synthetic") for r, _ in runs),
    )


# ---- markdown ----


def render_markdown(d: MatrixData) -> str:
    n = d.models[0].n if d.models else 0
    L = [
        f"# Model matrix: {d.suite}",
        "",
        f"Generated {d.generated}. {n} cases per model, one run each. Pass = the feature's own exact scoring.",
        "",
    ]
    if d.synthetic:
        L += ["> SYNTHETIC: this matrix comes from the fake adapter. The numbers are not a real model's.", ""]
    L += [
        "## Accuracy",
        "",
        "| model | passed | 95% interval (Wilson) | " + " | ".join(d.categories) + " | errored |",
        "|---|---|---|" + "---|" * len(d.categories) + "---|",
    ]
    for s in d.models:
        cats = " | ".join(pct(*s.by_cat.get(c, (0, 0))) for c in d.categories)
        L.append(
            f"| {s.model} | {pct(s.passed, s.n)} | {100 * s.ci[0]:.0f}% to {100 * s.ci[1]:.0f}% | {cats} | {s.errors} |"
        )
    L += [
        "",
        "## Latency, tokens, cost",
        "",
        "| model | p50 | p95 | prompt tok/case | completion tok/case | cost/case | total cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in d.models:
        p = s.perf
        L.append(
            f"| {s.model} | {p.p50_ms / 1000:.1f}s | {p.p95_ms / 1000:.1f}s | {p.mean_prompt_tokens:.0f} | "
            f"{p.mean_completion_tokens:.0f} | INR {p.mean_cost_inr:.2f} | INR {p.total_cost_inr:.2f} |"
        )
    L += [
        "",
        "Latency is wall time per case measured by the harness, over cases that completed. "
        "Local models cost INR 0; set --price-in and --price-out for paid models.",
        "",
    ]
    if d.pairs:
        L += [
            "## Are the differences real?",
            "",
            "Paired by case. A flip is a case one model passes and the other fails. McNemar is an exact test on the "
            "flips; the bootstrap interval is for accuracy(B) minus accuracy(A). With 60 cases, a few points of "
            "difference is usually noise.",
            "",
            "| A | B | A passed | B passed | flips (B better / B worse) | McNemar p | bootstrap 95% (points) | reading |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in d.pairs:
            m = c.mcnemar
            reading = {
                "identical": "identical",
                "better": "B better",
                "worse": "B worse",
                "borderline": "borderline",
                "noise": "within noise",
            }[c.verdict]
            L.append(
                f"| {c.a.model} | {c.b.model} | {c.a_pass} | {c.b_pass} | {m.only_b} / {m.only_a} | "
                f"{fmt_p(m.p_value)} | {points(c.boot.low)} to {points(c.boot.high)} | {reading} |"
            )
        L.append("")
    L += [
        "## Runs",
        "",
        "| model | run | status | git | prompt | code | started |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in d.models:
        r = s.run
        L.append(
            f"| {s.model} | {r.run_id} | {r.status} | {r.git_hash or '-'} | {r.prompt_hash or '-'} | "
            f"{r.code_hash or '-'} | {r.started_at[:16].replace('T', ' ')} |"
        )
    if d.notes:
        L += ["", "## Notes", ""] + [f"- {x}" for x in d.notes]
    L.append("")
    return "\n".join(L)


def write_reports(d: MatrixData, out_dir: Path) -> tuple[Path, Path]:
    from harness.report import matrix_html

    out_dir.mkdir(parents=True, exist_ok=True)
    md, html = out_dir / "matrix.md", out_dir / "matrix.html"
    md.write_text(render_markdown(d), encoding="utf-8")
    html.write_text(matrix_html(d), encoding="utf-8")
    return md, html


# ---- driver ----


def run_model(
    suite: str, model: str, db: str, passthrough: list[str], resume: bool, log=print
) -> tuple[int, dict]:
    """One subprocess per (suite, model). Returns (exit code, the run's json summary or {})."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            "-m",
            "harness.run",
            "--suite",
            suite,
            "--model",
            model,
            "--db",
            db,
            "--json-out",
            str(out),
            *passthrough,
        ]
        if resume:
            cmd += ["--resume", "--reuse-complete"]
        log(f"== {suite} / {model}")
        code = subprocess.run(cmd, cwd=ROOT, check=False).returncode
        info = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return code, info


def latest_complete(store: Store, suite: str, model: str, full_suite: bool = True) -> Run | None:
    """The latest complete run for a model. By default only full-suite runs (no --limit, no --ids, every case
    of the case file), so a smoke test or a rerun of a few cases is never reported as the model's result."""
    runs = [r for r in store.list_runs(suite, model) if r.status == COMPLETE]
    if full_suite:
        runs = [r for r in runs if r.is_full_suite]
    return runs[-1] if runs else None


def main(argv: list[str] | None = None) -> int:
    harden_console()
    p = argparse.ArgumentParser(
        prog="python -m harness.matrix",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--suite", required=True)
    p.add_argument("--models", required=True, help="comma-separated, run in this order")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--out", default=str(REPORTS), help="report directory (default reports/)")
    p.add_argument("--limit", type=int, help="run only the first N cases (smoke test; not gate-grade)")
    p.add_argument("--resume", action="store_true", help="continue unfinished runs, reuse finished ones")
    p.add_argument(
        "--report-only", action="store_true", help="do not run; report the latest complete run per model"
    )
    p.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="MODEL=RUN_ID",
        help="report this run for the model instead of the latest full-suite one (repeatable)",
    )
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="MODEL=IN,OUT",
        help="USD per million tokens for a paid model, e.g. claude-x=3,15 (repeatable)",
    )
    p.add_argument("--usd-to-inr", type=float, default=88.0)
    p.add_argument("--no-preflight", action="store_true")
    args = p.parse_args(argv)
    args.db = str(Path(args.db).resolve())  # the children run in another working directory
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    pins: dict[str, str] = {}
    for spec in args.pin:
        name, _, rid = spec.partition("=")
        if not name or not rid:
            print(f"error: bad --pin '{spec}', want MODEL=RUN_ID", file=sys.stderr)
            return 2
        pins[name] = rid
    stray_pins = [m for m in pins if m not in models]
    if stray_pins:
        print(f"error: --pin names a model that is not in --models: {', '.join(stray_pins)}", file=sys.stderr)
        return 2
    prices: dict[str, tuple[float, float]] = {}
    for spec in args.price:
        name, _, rest = spec.partition("=")
        try:
            a, b = rest.split(",")
            prices[name] = (float(a), float(b))
        except ValueError:
            print(f"error: bad --price '{spec}', want MODEL=IN,OUT", file=sys.stderr)
            return 2

    failed: list[str] = []
    run_ids: dict[str, str] = {}
    if not args.report_only:
        for model in [m for m in models if m not in pins]:
            pt = [
                "--timeout",
                str(args.timeout),
                "--retries",
                str(args.retries),
                "--usd-to-inr",
                str(args.usd_to_inr),
            ]
            if args.limit:
                pt += ["--limit", str(args.limit)]
            if args.no_preflight:
                pt.append("--no-preflight")
            if model in prices:
                pt += ["--price-in", str(prices[model][0]), "--price-out", str(prices[model][1])]
            code, info = run_model(args.suite, model, args.db, pt, args.resume)
            if info.get("run_id"):
                run_ids[model] = info["run_id"]
            if code not in (0, 1):
                failed.append(f"{model} (exit {code})")
            elif code == 1:
                failed.append(f"{model} (run incomplete; rerun with --resume)")

    runs: list[tuple[Run, list[CaseResult]]] = []
    try:
        with Store(args.db) as store:
            for model in models:
                if model in pins:
                    run = store.get_run(pins[model])
                    if (run.suite, run.model) != (args.suite, model):
                        print(
                            f"error: pinned run {run.run_id} is {run.suite}/{run.model}, not {args.suite}/{model}",
                            file=sys.stderr,
                        )
                        return 2
                elif model in run_ids:
                    run = store.get_run(run_ids[model])
                else:
                    run = latest_complete(store, args.suite, model, full_suite=not args.limit)
                if run is None:
                    failed.append(f"{model} (no complete full-suite run in {args.db})")
                    continue
                runs.append((run, store.results(run.run_id)))
    except RunNotFound as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not runs:
        print("error: no runs to report", file=sys.stderr)
        return 2
    data = build_matrix(args.suite, runs)
    for f in failed:
        data.notes.append(f"Not in the matrix or incomplete: {f}")
    md, html = write_reports(data, Path(args.out))
    print(f"wrote {md} and {html}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
