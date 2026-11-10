"""Adapter for project 4, the ask-your-data text-to-SQL agent (../fde-ask-your-data).

Written against project 4's documented contract (its docs/spec.md, "Contract for project 5"):
  app.agent.ask(question, ...) -> dict   keys sql, columns, rows, summary, refused, refused_reason,
                                         usage {prompt_tokens, completion_tokens, latency_ms}; never raises
  evals.scoring.score_case(case, result, conn=None) -> {"passed", "checks", "reasons"}
  evals/gold.jsonl                       40 gold questions; category is exactly simple, complex or refuse
Model switch: `app.config.settings.ollama_model`, read on every LLM call. Only the agent variant runs;
project 4's table-names-only baseline is not rerun by the harness.

score_case compares result sets by running the gold SQL and the agent's SQL on a database connection
passed as its third argument. This adapter opens that connection lazily as the read-only analyst role
(settings.analyst_dsn; override with FDE_ASK_YOUR_DATA_SCORE_DSN) and rolls back after every case.

Status (checked 2026-11-10 against project 4's code): matches its own evals/run.py, which calls
agent.ask(question, variant=..., log_requests=..., log_path=...) and scoring.score_case(case, result, conn)
with conn = psycopg.connect(settings.analyst_dsn). Its 40 gold cases load and validate through this adapter
(tests/test_ask_adapter.py, local only). Not yet run live: project 4's warehouse Postgres (port 5435) was not
up while this was built, so the full ask-and-score path has only been proved on a stand-in repo that follows
the same contract. TODO for the orchestrator: one live case through this adapter once the warehouse is up
(`python -m harness.run --suite ask_your_data --model fusion27b:iq3 --limit 3`).
Needs project 4's Postgres on 5435 (warehouse loaded) and the chosen model in Ollama.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

from harness.adapters._feature import FeatureRepoAdapter, file_hash, import_feature_repo
from harness.adapters.base import AdapterError, InfraError, git_info, tree_hash

DEFAULT_REPO = Path(os.environ.get("FDE_ASK_YOUR_DATA_REPO", str(Path(__file__).resolve().parents[3] / "fde-ask-your-data")))  # sibling checkout by default
CATEGORIES = ("simple", "complex", "refuse")
# Only an unreachable model is infrastructure. "internal error" and the like are the agent's own failures
# and are scored as failures, so a bug in the feature cannot hide as an excluded case.
INFRA_ERROR_PREFIXES = ("model unavailable",)


class AskYourDataAdapter(FeatureRepoAdapter):
    name = "ask_your_data"
    description = "Project 4: ask-your-data text-to-SQL agent, execution accuracy against gold SQL."
    repo_env = "FDE_ASK_YOUR_DATA_REPO"
    default_repo = DEFAULT_REPO

    def __init__(self, repo: Path | str | None = None, gold_path: Path | str | None = None):
        super().__init__(repo)
        self.gold_path = Path(gold_path) if gold_path else self.repo / "evals" / "gold.jsonl"
        self._conn = None

    def _load(self) -> SimpleNamespace:
        import_feature_repo(self.repo)
        agent = importlib.import_module("app.agent")
        scoring = importlib.import_module("evals.scoring")
        if not os.environ.get("HARNESS_KEEP_FEATURE_LOGS"):
            logging_ = importlib.import_module("app.logging_")
            logging_.log_request = lambda *a, **k: None
        return SimpleNamespace(
            agent=agent,
            ask=agent.ask,
            score_case=scoring.score_case,
            scoring=scoring,
            scoring_file=Path(scoring.__file__),
            settings=importlib.import_module("app.config").settings,
            llm=importlib.import_module("app.llm"),
        )

    # ---- adapter interface ----

    def cases(self) -> list[dict]:
        if not self.gold_path.is_file():
            raise AdapterError(f"gold file not found: {self.gold_path} (project 4 writes evals/gold.jsonl)")
        out = [
            json.loads(line)
            for line in self.gold_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for c in out:
            if c.get("category") not in CATEGORIES:
                raise AdapterError(
                    f"case {c.get('id')}: category must be one of {CATEGORIES}, got {c.get('category')!r}"
                )
        return out

    def configure(self, timeout_s: float | None = None) -> None:
        if timeout_s:
            self.turn_timeout_s = float(timeout_s)

    def run_case(self, case: dict, model: str) -> dict:
        self.set_model(model)
        kwargs = {}
        params = inspect.signature(self.m.ask).parameters
        if "variant" in params:
            kwargs["variant"] = "agent"
        if "log_requests" in params:
            kwargs["log_requests"] = False  # harness runs stay out of the feature's request log
        result = self.m.ask(case["question"], **kwargs)
        err = str(result.get("error") or "")
        if result.get("status") == "error" and err.startswith(INFRA_ERROR_PREFIXES):
            raise InfraError(err)
        usage = result.get("usage") or {}
        return {
            "result": result,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
            "p4_latency_ms": usage.get("latency_ms"),
        }

    def _scoring_conn(self):
        if self._conn is None or getattr(self._conn, "closed", False):
            import psycopg

            dsn = os.environ.get("FDE_ASK_YOUR_DATA_SCORE_DSN") or self.m.settings.analyst_dsn
            self._conn = psycopg.connect(dsn, connect_timeout=5)
        return self._conn

    def score(self, case: dict, output: dict) -> dict:
        needs_conn = "conn" in inspect.signature(self.m.score_case).parameters
        if case["category"] != "refuse" and needs_conn:
            conn = self._scoring_conn()
            try:
                return self.m.score_case(case, output["result"], conn=conn)
            finally:
                conn.rollback()
        return self.m.score_case(case, output["result"])

    # ---- provenance, reachability ----

    def provenance(self) -> dict:
        git_hash, dirty = git_info(self.repo)
        return {
            "git_hash": git_hash,
            "git_dirty": dirty,
            "scorer_hash": file_hash([self.m.scoring_file]),
            "config": {"variant": "agent", "judge": None, "clock_now": None},
            "prompt_hash": file_hash(
                [self.repo / "app" / "prompts.py", self.repo / "warehouse" / "schema_docs.yaml"]
            ),
            "code_hash": tree_hash(self.repo, ("*.py", "*.yaml"), ("app", "warehouse")),
        }

    def preflight(self, model: str) -> list[str]:
        problems = self.check_ollama(model)
        try:
            import psycopg

            with psycopg.connect(self.m.settings.analyst_dsn, connect_timeout=3) as conn:
                conn.execute("SELECT 1")
        except Exception as e:  # noqa: BLE001
            problems.append(
                f"project 4 warehouse is not reachable ({type(e).__name__}); "
                "run `docker compose up -d` in its repo"
            )
        return problems

    def warmup(self, model: str) -> None:
        self.warm_up_llm(model, embed=False)

    def close(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
        super().close()
