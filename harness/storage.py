"""SQLite storage for runs and per-case results, plus run export/import as JSON files."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from harness.models import COMPLETE, PARTIAL, RUN_FORMAT, RUNNING, CaseResult, Run, slug

DEFAULT_DB = Path("runs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  suite TEXT NOT NULL,
  model TEXT NOT NULL,
  label TEXT,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  git_hash TEXT,
  prompt_hash TEXT,
  code_hash TEXT,
  case_set_hash TEXT,
  n_cases INTEGER NOT NULL DEFAULT 0,
  config TEXT NOT NULL DEFAULT '{}',
  notes TEXT
);
CREATE TABLE IF NOT EXISTS results (
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  case_id TEXT NOT NULL,
  category TEXT NOT NULL,
  passed INTEGER NOT NULL,
  status TEXT NOT NULL,
  checks TEXT NOT NULL DEFAULT '{}',
  reasons TEXT NOT NULL DEFAULT '[]',
  latency_ms REAL NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  cost_inr REAL NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 1,
  output TEXT,
  error TEXT,
  ts TEXT NOT NULL,
  PRIMARY KEY (run_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_suite_model ON runs(suite, model, started_at);
"""

RUN_COLUMNS = [
    "run_id",
    "suite",
    "model",
    "label",
    "status",
    "started_at",
    "finished_at",
    "git_hash",
    "prompt_hash",
    "code_hash",
    "case_set_hash",
    "n_cases",
    "config",
    "notes",
]


class RunNotFound(LookupError):
    pass


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Store:
    """One SQLite file. Every write commits at once so an interrupted run loses nothing."""

    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- runs ----

    def new_run_id(self, suite: str, model: str, started_at: str) -> str:
        stamp = started_at.replace("-", "").replace(":", "").replace("T", "-")
        base = f"{stamp}-{slug(suite)}-{slug(model)}"
        rid, n = base, 1
        while self.conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (rid,)).fetchone():
            n += 1
            rid = f"{base}-{n}"
        return rid

    def create_run(self, run: Run) -> Run:
        row = run.to_dict()
        row["config"] = json.dumps(row["config"], ensure_ascii=False, default=str)
        self.conn.execute(
            f"INSERT INTO runs ({', '.join(RUN_COLUMNS)}) VALUES ({', '.join('?' for _ in RUN_COLUMNS)})",
            [row[c] for c in RUN_COLUMNS],
        )
        self.conn.commit()
        return run

    def update_run(self, run_id: str, **fields) -> None:
        if not fields:
            return
        if "config" in fields:
            fields["config"] = json.dumps(fields["config"], ensure_ascii=False, default=str)
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", [*fields.values(), run_id])
        self.conn.commit()

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> Run:
        d = dict(row)
        d["config"] = json.loads(d.get("config") or "{}")
        return Run.from_dict(d)

    def get_run(self, ref: str) -> Run:
        """Exact run id, or a unique prefix of one."""
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (ref,)).fetchone()
        if row:
            return self._run_from_row(row)
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE run_id LIKE ? ESCAPE '\\' ORDER BY run_id", (_like_escape(ref) + "%",)
        ).fetchall()
        if len(rows) == 1:
            return self._run_from_row(rows[0])
        if not rows:
            raise RunNotFound(f"no run '{ref}' in {self.path}")
        raise RunNotFound(f"'{ref}' matches {len(rows)} runs: {', '.join(r['run_id'] for r in rows[:5])}")

    def list_runs(self, suite: str | None = None, model: str | None = None) -> list[Run]:
        q, args = "SELECT * FROM runs WHERE 1=1", []
        if suite:
            q += " AND suite = ?"
            args.append(suite)
        if model:
            q += " AND model = ?"
            args.append(model)
        rows = self.conn.execute(q + " ORDER BY started_at, run_id", args).fetchall()
        return [self._run_from_row(r) for r in rows]

    def latest_run(self, suite: str, model: str, statuses: tuple[str, ...] | None = None) -> Run | None:
        runs = [r for r in self.list_runs(suite, model) if statuses is None or r.status in statuses]
        return runs[-1] if runs else None

    def find_resumable(self, suite: str, model: str, fingerprint: tuple) -> Run | None:
        """Latest unfinished run (running or partial) for this suite and model with the same fingerprint."""
        run = self.latest_run(suite, model, (RUNNING, PARTIAL))
        return run if run and run.fingerprint() == fingerprint else None

    def find_complete(self, suite: str, model: str, fingerprint: tuple) -> Run | None:
        run = self.latest_run(suite, model)
        return run if run and run.status == COMPLETE and run.fingerprint() == fingerprint else None

    # ---- results ----

    def save_result(self, run_id: str, r: CaseResult) -> None:
        self.conn.execute(
            "INSERT INTO results (run_id, case_id, category, passed, status, checks, reasons,"
            " latency_ms, prompt_tokens, completion_tokens, cost_inr, attempts, output, error, ts)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(run_id, case_id) DO UPDATE SET category=excluded.category, passed=excluded.passed,"
            " status=excluded.status, checks=excluded.checks, reasons=excluded.reasons,"
            " latency_ms=excluded.latency_ms, prompt_tokens=excluded.prompt_tokens,"
            " completion_tokens=excluded.completion_tokens, cost_inr=excluded.cost_inr,"
            " attempts=excluded.attempts, output=excluded.output, error=excluded.error, ts=excluded.ts",
            (
                run_id,
                r.case_id,
                r.category,
                int(r.passed),
                r.status,
                json.dumps(r.checks),
                json.dumps(r.reasons, ensure_ascii=False),
                r.latency_ms,
                r.prompt_tokens,
                r.completion_tokens,
                r.cost_inr,
                r.attempts,
                json.dumps(r.output, ensure_ascii=False, default=str) if r.output is not None else None,
                r.error,
                r.ts,
            ),
        )
        self.conn.commit()

    def results(self, run_id: str) -> list[CaseResult]:
        rows = self.conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY rowid", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d.pop("run_id")
            d["passed"] = bool(d["passed"])
            d["checks"] = json.loads(d["checks"] or "{}")
            d["reasons"] = json.loads(d["reasons"] or "[]")
            d["output"] = json.loads(d["output"]) if d["output"] else None
            out.append(CaseResult.from_dict(d))
        return out

    def delete_run(self, run_id: str) -> None:
        self.conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self.conn.commit()


# ---- run files (export, import, fixtures) ----


def run_to_file_dict(run: Run, results: list[CaseResult], with_output: bool = True) -> dict:
    rows = []
    for r in results:
        d = r.to_dict()
        if not with_output:
            d["output"] = None
        rows.append(d)
    return {"format": RUN_FORMAT, "run": run.to_dict(), "results": rows}


def write_run_file(path: str | Path, run: Run, results: list[CaseResult], with_output: bool = True) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(run_to_file_dict(run, results, with_output), indent=1, ensure_ascii=False)
    p.write_text(body + "\n", encoding="utf-8")
    return p


def read_run_file(path: str | Path) -> tuple[Run, list[CaseResult]]:
    """Read a run file. Anything wrong with it is a ValueError (so the CLIs exit 2), never a crash.

    Checked: the format tag, the required fields of the run and of every result, duplicate case ids.
    An errored or timed-out result is never a pass, whatever the file says (see CaseResult).
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not a readable run file ({e})") from e
    if not isinstance(data, dict) or data.get("format") != RUN_FORMAT:
        found = data.get("format") if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"{path}: not a harness run file (format {found!r}, want {RUN_FORMAT!r})")
    try:
        run = Run.from_dict(data["run"])
        results = [CaseResult.from_dict(r) for r in data["results"]]
        for r in results:
            if not isinstance(r.case_id, str) or not r.case_id or not isinstance(r.category, str):
                raise ValueError("a result has no usable case_id or category")
        if not isinstance(run.config, dict):
            raise ValueError("the run's config is not an object")
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise ValueError(f"{path}: malformed run file ({type(e).__name__}: {e})") from e
    seen: set[str] = set()
    for r in results:
        if r.case_id in seen:
            raise ValueError(f"{path}: duplicate case id {r.case_id}")
        seen.add(r.case_id)
    return run, results


def load_run(ref: str, db_path: str | Path = DEFAULT_DB) -> tuple[Run, list[CaseResult]]:
    """Resolve RUN: a path to an exported run file, else a run id (or unique prefix) in the database."""
    p = Path(ref)
    if p.suffix == ".json" and p.is_file():
        return read_run_file(p)
    if not Path(db_path).exists():
        raise RunNotFound(f"'{ref}' is not a run file and the database {db_path} does not exist")
    with Store(db_path) as store:
        run = store.get_run(ref)
        return run, store.results(run.run_id)


def import_run(store: Store, run: Run, results: list[CaseResult]) -> Run:
    """Copy a run (for example from a run file) into the database. Keeps the run id; errors if it exists."""
    store.create_run(run)
    for r in results:
        store.save_result(run.run_id, r)
    return run
