"""Inspect and move runs.

    python -m harness.runs list [--suite S]
    python -m harness.runs show RUN
    python -m harness.runs export RUN OUT.json [--with-output]
    python -m harness.runs import FILE.json
    python -m harness.runs import-p1 RESULTS.jsonl --model fusion27b:iq3 [--out FILE.json | --to-db]

Exported run files are what the CI gate compares (fixtures/runs/*.run.json). `import-p1` converts a
project 1 results file (evals/results/*.jsonl) into a harness run, so its history can be compared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness import harden_console
from harness.compare import pct
from harness.models import OK
from harness.storage import (
    DEFAULT_DB,
    RunNotFound,
    Store,
    import_run,
    load_run,
    read_run_file,
    write_run_file,
)


def main(argv: list[str] | None = None) -> int:
    harden_console()
    p = argparse.ArgumentParser(
        prog="python -m harness.runs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--suite")
    ls.add_argument("--model")
    sh = sub.add_parser("show")
    sh.add_argument("run")
    ex = sub.add_parser("export")
    ex.add_argument("run")
    ex.add_argument("out")
    ex.add_argument("--with-output", action="store_true", help="keep each case's full output (large)")
    im = sub.add_parser("import")
    im.add_argument("file")
    ip = sub.add_parser("import-p1")
    ip.add_argument("results", help="project 1 evals/results/*.jsonl")
    ip.add_argument("--model", required=True, help="model name to record (project 1 did not store it)")
    ip.add_argument("--cases", help="project 1 evals/cases.jsonl (default: its repo path)")
    ip.add_argument("--label")
    ip.add_argument("--out", help="write a run file instead of touching the database")
    ip.add_argument("--to-db", action="store_true", help="import into the database")
    args = p.parse_args(argv)

    try:
        if args.cmd == "list":
            with Store(args.db) as store:
                runs = store.list_runs(args.suite, args.model)
                if not runs:
                    print("no runs")
                for r in runs:
                    res = store.results(r.run_id)
                    print(
                        f"{r.run_id}  {r.suite}  {r.model}  {r.status}  "
                        f"{sum(1 for x in res if x.passed)}/{len(res)} passed"
                        f"{'  (synthetic)' if r.config.get('synthetic') else ''}"
                    )
        elif args.cmd == "show":
            run, res = load_run(args.run, args.db)
            print(f"{run.run_id}: {run.suite} / {run.model}, {run.status}, started {run.started_at}")
            print(
                f"git {run.git_hash or '-'}  prompt {run.prompt_hash or '-'}  code {run.code_hash or '-'}  "
                f"cases {run.case_set_hash or '-'}"
            )
            print(
                f"passed {pct(sum(1 for r in res if r.passed), len(res))}; "
                f"errored or timed out {sum(1 for r in res if r.status != OK)}"
            )
            for k, v in run.config.items():
                print(f"  {k}: {v}")
        elif args.cmd == "export":
            run, res = load_run(args.run, args.db)
            print(f"wrote {write_run_file(args.out, run, res, with_output=args.with_output)}")
        elif args.cmd == "import":
            run, res = read_run_file(args.file)
            with Store(args.db) as store:
                import_run(store, run, res)
            print(f"imported {run.run_id} ({len(res)} results)")
        elif args.cmd == "import-p1":
            from harness.adapters.support_agent import import_results_file

            run, res = import_results_file(
                Path(args.results),
                model=args.model,
                cases_path=Path(args.cases) if args.cases else None,
                label=args.label,
            )
            if args.out:
                print(f"wrote {write_run_file(args.out, run, res, with_output=False)}")
            if args.to_db or not args.out:
                with Store(args.db) as store:
                    import_run(store, run, res)
                print(f"imported {run.run_id} ({len(res)} results) into {args.db}")
    except (RunNotFound, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - e.g. a duplicate run id on import
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
