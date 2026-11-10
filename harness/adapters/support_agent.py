"""Adapter for project 1, the Larkfield support agent (../fde-support-agent).

Calls project 1 in-process and measures exactly what project 1 measures:
  cases()     project 1's own `evals.run.load_cases()` (evals/cases.jsonl, 60 cases)
  run_case()  project 1's own `evals.run.run_case(case, "agent", run_turn, timeout_s)`: multi-turn
              cases, a fresh session per case, reply merging and timeouts are not re-implemented here
  score()     project 1's own `evals.scoring.score_case(case, merged_reply)`: exact scoring
Model switch: `app.config.settings.ollama_model`, which project 1 reads on every LLM call.

The gated metric is the exact scoring only. Project 1's LLM judge uses the same settings.ollama_model,
so it would change with the model under test; this adapter does not run it (recorded in the run config
as judge: none). If a judge is ever added, pin one judge model for every candidate.

Infrastructure failure versus model failure. Project 1 catches every exception in run_turn and answers with
its hand-off text, logging "turn failed, handing off" with the exception. On a refuse case that text can
pass, so a dead Ollama or a slow model whose 120 s request timed out would score as a good refusal. The
adapter hooks that logger message (per thread, per case) and classifies the exception:
  LLMError("empty answer")  the model returned nothing: a model failure, scored as project 1 scores it
  anything else             a failed request or timeout, a database error: an infrastructure error, recorded as
                            an error (retried, counted as a failure in compare and gate), never as a pass
This deliberately deviates from project 1's own score on such a reply, which would count the hand-off.

Clock. Orders have fixed dates, so a wall clock moves return-window cases between runs. The adapter sets
`app.config.settings.clock_now` to the dataset's NOW (DATASET_NOW) and records it in the run config; runs made
under different clocks are not comparable (gate exit 2).

Project 1 files are never modified. Its request logging is switched off in this process (monkeypatch)
so harness runs do not write to its logs or request_log table; set HARNESS_KEEP_FEATURE_LOGS=1 to keep it.
Needs project 1's Postgres on 5433 (ingested) and the `nomic-embed-text` model in Ollama.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace

from harness.adapters._feature import FeatureRepoAdapter, file_hash, import_feature_repo
from harness.adapters.base import InfraError, git_info, tree_hash
from harness.models import COMPLETE, ERROR, OK, TIMEOUT, CaseResult, Run, stable_hash

VARIANT = "agent"  # project 1's baselines (keyword, no_retrieval) are not rerun by the harness
DEFAULT_REPO = Path(os.environ.get("FDE_SUPPORT_AGENT_REPO", str(Path(__file__).resolve().parents[3] / "fde-support-agent")))  # sibling checkout by default
DATASET_NOW = "2026-11-10T09:00:00+05:30"  # the dataset's "now": orders and policy windows are relative to it
TURN_FAILED_MESSAGE = "turn failed, handing off"  # project 1's log message on its exception path
MODEL_FAILURE_MESSAGES = ("empty answer",)  # LLMError texts that mean the model answered badly, not infra


def is_model_failure(exc: BaseException | None) -> bool:
    """An empty reply from the model is the model's failure; every other exception is infrastructure."""
    return type(exc).__name__ == "LLMError" and str(exc).startswith(MODEL_FAILURE_MESSAGES)


class _TurnFailureCapture(logging.Handler):
    """Collects the exceptions behind project 1's "turn failed, handing off" log lines, for one thread only,
    so a timed-out turn still running in the background cannot leak into the next case."""

    def __init__(self, sink: list[BaseException | None], thread_id: int):
        super().__init__(level=logging.ERROR)
        self.sink = sink
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread_id and record.getMessage().startswith(TURN_FAILED_MESSAGE):
            self.sink.append(record.exc_info[1] if record.exc_info else None)


class SupportAgentAdapter(FeatureRepoAdapter):
    name = "support_agent"
    description = "Project 1: Larkfield support agent (hybrid RAG, tools, guardrails), exact scoring."
    repo_env = "FDE_SUPPORT_AGENT_REPO"
    default_repo = DEFAULT_REPO

    def __init__(self, repo: Path | str | None = None, clock_now: str | None = DATASET_NOW):
        super().__init__(repo)
        self.clock_now = clock_now
        self._failures: dict[str, list[BaseException | None]] = {}
        self._live_turns: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

    def _load(self) -> SimpleNamespace:
        import_feature_repo(self.repo)
        agent = importlib.import_module("app.agent")
        ev_run = importlib.import_module("evals.run")
        ev_scoring = importlib.import_module("evals.scoring")
        if not os.environ.get("HARNESS_KEEP_FEATURE_LOGS"):
            logging_ = importlib.import_module("app.logging_")
            logging_.log_request = lambda record: None  # the agent calls logging_.log_request at request time
        return SimpleNamespace(
            agent=agent,
            run_turn=self._guarded_run_turn,
            raw_run_turn=agent.run_turn,
            settings=importlib.import_module("app.config").settings,
            llm=importlib.import_module("app.llm"),
            p1_run_case=ev_run.run_case,
            load_cases=ev_run.load_cases,
            case_turns=ev_run.case_turns,
            score_case=ev_scoring.score_case,
            scoring_file=Path(ev_scoring.__file__),
        )

    # ---- the clock ----

    @property
    def clock_supported(self) -> bool:
        return hasattr(self.m.settings, "clock_now")

    def apply_clock(self) -> None:
        if self.clock_now and self.clock_supported:
            self.m.settings.clock_now = self.clock_now

    def set_model(self, model: str) -> None:
        super().set_model(model)
        self.apply_clock()

    # ---- hooking project 1's exception path ----

    def _guarded_run_turn(self, session_id, message, *args, **kwargs):
        """project 1's run_turn, plus a note of every exception it turned into a hand-off."""
        captured: list[BaseException | None] = []
        thread = threading.current_thread()
        handler = _TurnFailureCapture(captured, threading.get_ident())
        log = logging.getLogger(self.m.agent.__name__)
        log.addHandler(handler)
        with self._lock:
            self._live_turns[threading.get_ident()] = thread
        try:
            return self.m.raw_run_turn(session_id, message, *args, **kwargs)
        finally:
            log.removeHandler(handler)
            with self._lock:
                self._live_turns.pop(threading.get_ident(), None)
                if captured:
                    self._failures.setdefault(session_id, []).extend(captured)

    def orphan_threads(self) -> list[threading.Thread]:
        """Project 1 abandons a turn that exceeded its timeout but the thread keeps running (and keeps the GPU)."""
        with self._lock:
            return [t for t in self._live_turns.values() if t.is_alive()]

    # ---- adapter interface ----

    def cases(self) -> list[dict]:
        return self.m.load_cases()

    def configure(self, timeout_s: float | None = None) -> None:
        if timeout_s:
            self.turn_timeout_s = float(timeout_s)

    def case_timeout(self, case: dict, timeout_s: float | None) -> float | None:
        """Project 1 times out each turn itself; the outer limit covers every turn plus a little slack."""
        turns = len(self.m.case_turns(case))
        return (timeout_s or self.turn_timeout_s) * turns + 15

    def run_case(self, case: dict, model: str) -> dict:
        self.set_model(model)
        rec = self.m.p1_run_case(case, VARIANT, self.m.run_turn, self.turn_timeout_s)
        with self._lock:
            failures = self._failures.pop(rec.get("session_id"), [])
        err = rec.get("error")
        if err:  # project 1 records timeouts and exceptions as failures; the harness records them as errors
            if str(err).startswith("timeout"):
                raise TimeoutError(err)
            raise InfraError(str(err))
        infra = [f for f in failures if not is_model_failure(f)]
        if infra:
            first = infra[0]
            raise InfraError(
                "project 1 handed off after an exception, not after a decision "
                f"({type(first).__name__}: {str(first)[:200]}). Recorded as an error, not as a reply."
            )
        reply = rec["reply"]
        usage = reply.get("usage") or {}
        return {
            "reply": reply,
            "turns": rec.get("turns"),
            "variant": VARIANT,
            "clock_now": self.clock_now if self.clock_supported else None,
            "session_id": rec.get("session_id"),
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
            "model_failures": [str(f) for f in failures],
            "p1_latency_ms": usage.get("latency_ms"),
            "p1_wall_ms": rec.get("wall_ms"),
        }

    def score(self, case: dict, output: dict) -> dict:
        return self.m.score_case(case, output["reply"])

    # ---- provenance, reachability ----

    def provenance(self) -> dict:
        _ = self.m  # make sure the repo is importable before reading its prompt
        git_hash, dirty = git_info(self.repo)
        prompts, tools = importlib.import_module("app.prompts"), importlib.import_module("app.tools")
        return {
            "git_hash": git_hash,
            "git_dirty": dirty,
            "scorer_hash": file_hash([self.m.scoring_file]),
            "config": {
                "variant": VARIANT,
                "judge": None,
                "clock_now": self.clock_now if self.clock_supported else None,
            },
            "prompt_hash": stable_hash(
                {"system": prompts.SYSTEM_PROMPT, "tools": tools.TOOL_SPECS, "judge": None}
            ),
            "code_hash": tree_hash(self.repo, ("*.py", "*.md", "*.json"), ("app", "data")),
        }

    def db_ready(self) -> list[str]:
        try:
            import psycopg

            with psycopg.connect(self.m.settings.database_url, connect_timeout=3) as conn:
                n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        except Exception as e:  # noqa: BLE001
            return [
                f"project 1 Postgres is not reachable ({type(e).__name__}); run `docker compose up -d db` in its repo"
            ]
        return [] if n else ["project 1 database has no chunks; run `python -m app.ingest` in its repo"]

    def clock_ready(self) -> list[str]:
        if self.clock_now and not self.clock_supported:
            return [
                "project 1 has no `clock_now` setting, so its tools use the wall clock and return-window cases "
                "move with the calendar; update project 1 (settings.clock_now) or pass --no-preflight to accept that"
            ]
        return []

    def preflight(self, model: str) -> list[str]:
        return self.check_ollama(model) + self.db_ready() + self.clock_ready()

    def warmup(self, model: str) -> None:
        self.warm_up_llm(model, embed=True)


# ---- import project 1's recorded results (evals/results/*.jsonl) as harness runs ----


def read_cases_file(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_results_file(
    path: Path,
    *,
    model: str,
    cases_path: Path | None = None,
    scorer_path: Path | None = None,
    label: str | None = None,
    with_output: bool = False,
) -> tuple[Run, list[CaseResult]]:
    """Convert one project 1 results file into a harness run, without importing project 1.

    Project 1 did not store the model name, the case file, the scoring code or the clock that each past run
    used. So the caller names the model; case_set_hash and scorer_hash are the hashes of the cases file and of
    evals/scoring.py as they are now (the parity test shows that the current scorer reproduces the recorded
    verdicts); clock_now is None, because those runs used the wall clock. All of this is written into the
    run's config so nobody reads more into the fixture than it says.
    """
    path = Path(path)
    cases_path = cases_path or DEFAULT_REPO / "evals" / "cases.jsonl"
    scorer_path = scorer_path or cases_path.parent / "scoring.py"
    all_cases = read_cases_file(cases_path)
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[CaseResult] = []
    for rec in recs:
        err = rec.get("error")
        reply = rec.get("reply") or {}
        usage = reply.get("usage") or {}
        status = OK if not err else (TIMEOUT if str(err).startswith("timeout") else ERROR)
        results.append(
            CaseResult(
                case_id=rec["id"],
                category=rec["category"],
                passed=bool(rec.get("passed")) and status == OK,
                status=status,
                checks=dict(rec.get("checks") or {}),
                reasons=list(rec.get("reasons") or []),
                latency_ms=float(rec.get("wall_ms") or usage.get("latency_ms") or 0),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                attempts=1,
                output={"reply": reply} if with_output else None,
                error=err,
                ts=rec.get("ts", ""),
            )
        )
    stamp = path.name.split("-agent")[0]
    started = recs[0].get("ts", "") if recs else ""
    run = Run(
        run_id=f"p1-{path.stem}",
        suite="support_agent",
        model=model,
        status=COMPLETE if len(results) else "partial",
        label=label or f"recorded project 1 run {path.name}",
        started_at=started or stamp,
        finished_at=recs[-1].get("ts") if recs else None,
        case_set_hash=stable_hash(all_cases),
        n_cases=len(results),
        config={
            "source": f"project 1 evals/results/{path.name}",
            "variant": recs[0].get("variant") if recs else None,
            "imported": True,
            "judge": None,
            "limit": None,
            "ids": None,
            "n_cases_total": len(all_cases),
            "scorer_hash": file_hash([scorer_path]),
            "clock_now": None,
            "note": "Imported from recorded results. Project 1 did not store the model name, the case file, the "
            "scoring code or the clock per run: the model is as named at import, case_set_hash and scorer_hash "
            "are those of the current cases.jsonl and scoring.py, and the runs used the wall clock.",
        },
    )
    return run, results
