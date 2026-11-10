"""Build the committed fixtures from project 1's recorded eval results. Read-only on project 1.

    python scripts/make_fixtures.py [--p1 ../fde-support-agent] [--model fusion27b:iq3]

Writes:
  fixtures/runs/*.run.json          compact runs for the CI gate (no reply text)
  tests/fixtures/p1/                a frozen copy of project 1's evals/scoring.py, evals/run.py, cases.jsonl
                                    and one full results file, for the parity test (so it keeps passing
                                    when project 1 reruns or changes its scoring)

Every fixture is a real recorded run, not invented numbers. Project 1 stored neither the model name
nor the case file per run, so the model is as passed here and case_set_hash is the current cases.jsonl;
both facts are written into each run's config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.adapters.support_agent import import_results_file  # noqa: E402
from harness.storage import write_run_file  # noqa: E402

# fixture name -> (project 1 results file, optional first-N cases)
RUNS = {
    "support_agent-p1-run2-agent": ("20261110-003445-agent.jsonl", None),
    "support_agent-p1-run3-agent": ("20261110-011511-agent.jsonl", None),
    "support_agent-p1-run4-agent": ("20261110-015524-agent.jsonl", None),
    "support_agent-p1-run4-no-retrieval": ("20261110-015524-no_retrieval.jsonl", None),
    "support_agent-p1-run4-agent-first20": ("20261110-015524-agent.jsonl", 20),
}
PARITY_RESULTS = "20261110-023435-agent.jsonl"  # latest complete agent run at build time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p1", default=str(Path(__file__).resolve().parents[2] / "fde-support-agent"))
    p.add_argument("--model", default="fusion27b:iq3")
    p.add_argument(
        "--parity-results", default=PARITY_RESULTS, help="project 1 results file for the parity test"
    )
    args = p.parse_args()
    p1 = Path(args.p1)
    results_dir, cases = p1 / "evals" / "results", p1 / "evals" / "cases.jsonl"
    out = ROOT / "fixtures" / "runs"
    out.mkdir(parents=True, exist_ok=True)

    for name, (fname, first_n) in RUNS.items():
        src = results_dir / fname
        tmp = src
        if first_n:  # a partial run, like `--limit 20`: must not pass the gate against a full run
            tmp = ROOT / "fixtures" / "_tmp_first_n.jsonl"
            tmp.write_text(
                "".join(src.read_text(encoding="utf-8").splitlines(keepends=True)[:first_n]), encoding="utf-8"
            )
        run, res = import_results_file(tmp, model=args.model, cases_path=cases)
        run.run_id = name
        run.config["source"] = f"project 1 evals/results/{fname}" + (
            f" (first {first_n} cases)" if first_n else ""
        )
        run.label = name
        if first_n:  # a --limit style partial run: not gate-grade, whatever its hashes say
            run.config["limit"] = first_n
        write_run_file(out / f"{name}.run.json", run, res, with_output=False)
        if first_n:
            tmp.unlink()
        print(f"{name}: {sum(r.passed for r in res)}/{len(res)} passed")

    snap = ROOT / "tests" / "fixtures" / "p1"
    (snap / "evals").mkdir(parents=True, exist_ok=True)
    for f in ("__init__.py", "scoring.py", "run.py"):
        shutil.copyfile(p1 / "evals" / f, snap / "evals" / f)
    shutil.copyfile(cases, snap / "evals" / "cases.jsonl")
    shutil.copyfile(results_dir / args.parity_results, snap / "agent-results.jsonl")
    hashes = {
        str(f.relative_to(snap).as_posix()): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(snap.rglob("*"))
        if f.is_file() and f.name != "SNAPSHOT.json" and "__pycache__" not in f.parts
    }
    (snap / "SNAPSHOT.json").write_text(
        json.dumps(
            {
                "taken_from": str(p1),
                "results_file": args.parity_results,
                "sha256": hashes,
                "note": "Frozen copy of project 1 files for the parity test. Do not edit; rerun scripts/make_fixtures.py.",
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"snapshot written to {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
