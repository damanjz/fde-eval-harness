"""The support_agent adapter, proved three ways:
1. against a stub of project 1's `app` package plus a frozen copy of its real `evals` code (runs in CI),
2. parity: scoring project 1's recorded replies through the adapter must reproduce its recorded pass/fail,
3. against the real project 1 repo and Postgres with its LLM mocked (local only; skipped when unavailable).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import sys
from pathlib import Path

import pytest

from harness.adapters.base import AdapterError, InfraError
from harness.adapters.support_agent import SupportAgentAdapter, import_results_file
from harness.models import COMPLETE, OK, TIMEOUT
from harness.run import execute_run, run_one
from tests.conftest import P1_SNAPSHOT, REAL_P1

STUB_FILES = {
    "app/__init__.py": "",
    "app/config.py": """
class Settings:
    def __init__(self):
        self.ollama_model = "fusion27b:iq3"
        self.llm_provider = "ollama"
        self.anthropic_model = "claude-x"
        self.anthropic_api_key = None
        self.ollama_url = "http://localhost:1"
        self.llm_timeout_s = 120.0
        self.database_url = "postgresql://stub"
        self.clock_now = None

settings = Settings()
""",
    "app/prompts.py": 'SYSTEM_PROMPT = "stub system prompt v1"\n',
    "app/tools.py": 'TOOL_SPECS = [{"function": {"name": "lookup_order"}}]\n',
    "app/logging_.py": "LOGGED = []\n\ndef log_request(record):\n    LOGGED.append(record)\n",
    "app/db.py": "RESETS = []\n\ndef reset_pool():\n    RESETS.append(1)\n",
    "app/llm.py": """
from dataclasses import dataclass
from app.config import settings

CALLS = []

class LLMError(RuntimeError):
    pass

@dataclass
class Resp:
    content: str = "ready"

def chat(messages, tools=None):
    CALLS.append(("chat", settings.ollama_model, settings.llm_timeout_s))
    return Resp()

def embed(texts, kind="document"):
    CALLS.append(("embed", settings.ollama_model, kind))
    return [[0.0] * 3 for _ in texts]
""",
    "app/agent.py": """
import logging
import time
from dataclasses import dataclass, field
from app import logging_
from app.config import settings
from app.llm import LLMError

log = logging.getLogger(__name__)
HANDOFF_TEXT = ("I'm sorry, I can't answer that reliably right now, so I'm passing this to a person on our team. "
                "They will get back to you during support hours (9:00-19:00 IST, Mon-Sat).")
SCRIPT = {}
MODELS_SEEN = []
CLOCKS_SEEN = []

@dataclass
class AgentReply:
    text: str
    citations: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    pending_action: dict | None = None
    escalated: bool = False
    usage: dict = field(default_factory=dict)

def run_turn(session_id, user_message, confirm=None, retrieval_mode="hybrid"):
    MODELS_SEEN.append((settings.llm_provider, settings.ollama_model, settings.anthropic_model))
    CLOCKS_SEEN.append(settings.clock_now)
    spec = SCRIPT.get(user_message, {"text": HANDOFF_TEXT, "escalated": True, "llm_calls": 1})
    try:
        if spec.get("sleep"):
            time.sleep(spec["sleep"])
        if spec.get("raise"):
            raise RuntimeError(spec["raise"])
        if spec.get("llm_error"):  # a failed or timed-out request, as llm._post raises it after its retry
            raise LLMError(spec["llm_error"])
        if spec.get("empty"):  # the model answered with nothing: project 1 raises this in run_turn itself
            raise LLMError("empty answer")
    except Exception:
        log.exception("turn failed, handing off")  # the line project 1 writes on its exception path
        return AgentReply(text=HANDOFF_TEXT, escalated=True, usage={"prompt_tokens": 10, "completion_tokens": 0,
                                                                    "latency_ms": 5, "llm_calls": 1})
    logging_.log_request({"session_id": session_id, "message": user_message})
    return AgentReply(
        text=spec["text"], citations=spec.get("citations", []), tool_calls=spec.get("tool_calls", []),
        pending_action=spec.get("pending_action"), escalated=spec.get("escalated", False),
        usage={"prompt_tokens": spec.get("pt", 1500), "completion_tokens": spec.get("ct", 40),
               "latency_ms": 5, "llm_calls": spec.get("llm_calls", 1)},
    )
""",
}

A01 = "How long does standard delivery usually take?"
GOOD_A01 = {"text": "Standard delivery takes 3 to 6 business days [KB-001].", "citations": ["KB-001"]}
BAD_A01 = {"text": "Delivery is quick.", "citations": []}


@pytest.fixture
def stub_repo(tmp_path):
    repo = tmp_path / "p1"
    for rel, body in STUB_FILES.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    shutil.copytree(P1_SNAPSHOT / "evals", repo / "evals")  # project 1's real evals code, frozen
    (repo / "data").mkdir()
    (repo / "data" / "KB-001.md").write_text("stub article", encoding="utf-8")
    return repo


@pytest.fixture
def adapter(stub_repo, monkeypatch):
    monkeypatch.delenv("HARNESS_KEEP_FEATURE_LOGS", raising=False)
    ad = SupportAgentAdapter(repo=stub_repo)
    yield ad
    ad.close()


def case(adapter, cid):
    return next(c for c in adapter.cases() if c["id"] == cid)


def script(adapter, question, **spec):
    adapter.m.agent.SCRIPT[question] = spec


# ---- the adapter against the stub ----


def test_cases_come_from_project_1s_own_loader(adapter):
    cases = adapter.cases()
    assert len(cases) == 60
    cats = {}
    for c in cases:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    assert cats == {"answer": 35, "tool": 15, "refuse": 10}
    assert adapter.name == "support_agent"


def test_run_case_goes_through_project_1s_run_case_and_scores_with_its_scorer(adapter):
    c = case(adapter, "A01")
    script(adapter, A01, **GOOD_A01)
    out = adapter.run_case(c, "fusion27b:iq3")
    assert out["reply"]["text"].startswith("Standard delivery") and out["variant"] == "agent"
    assert out["usage"] == {"prompt_tokens": 1500, "completion_tokens": 40}
    assert out["session_id"].startswith("eval-agent-A01-")  # project 1's own session naming
    scored = adapter.score(c, out)
    assert (
        scored["passed"] is True
        and scored["checks"] == {"facts": True, "citations": True}
        and scored["reasons"] == []
    )
    script(adapter, A01, **BAD_A01)
    bad = adapter.score(c, adapter.run_case(c, "fusion27b:iq3"))
    assert bad["passed"] is False and any("missing fact" in r for r in bad["reasons"])
    assert any("missing citation KB-001" in r for r in bad["reasons"])


def test_model_is_switched_through_project_1s_settings(adapter):
    c = case(adapter, "A01")
    script(adapter, A01, **GOOD_A01)
    for model in ("qwen3:14b", "fusion27b:q4ks", "fusion27b:iq3"):
        adapter.run_case(c, model)
    seen = adapter.m.agent.MODELS_SEEN
    assert [m[1] for m in seen] == ["qwen3:14b", "fusion27b:q4ks", "fusion27b:iq3"]
    assert {m[0] for m in seen} == {"ollama"} and os.environ["OLLAMA_MODEL"] == "fusion27b:iq3"
    adapter.run_case(c, "claude-sonnet-5-5")
    assert seen[-1][0] == "anthropic" and seen[-1][2] == "claude-sonnet-5-5"
    adapter.run_case(c, "qwen3:14b")
    assert seen[-1][0] == "ollama"  # switching back restores the local provider


def test_harness_runs_do_not_write_project_1s_request_log(adapter, stub_repo, monkeypatch):
    script(adapter, A01, **GOOD_A01)
    adapter.run_case(case(adapter, "A01"), "m")
    assert adapter.m.agent.logging_.LOGGED == []
    adapter.close()
    for name in [n for n in sys.modules if n == "app" or n.startswith(("app.", "evals"))]:
        del sys.modules[name]
    monkeypatch.setenv("HARNESS_KEEP_FEATURE_LOGS", "1")
    keep = SupportAgentAdapter(repo=stub_repo)
    script(keep, A01, **GOOD_A01)
    keep.run_case(case(keep, "A01"), "m")
    assert len(keep.m.agent.logging_.LOGGED) == 1


def test_a_slow_turn_becomes_a_timeout_not_a_failure(adapter):
    script(adapter, A01, **GOOD_A01, sleep=1.5)
    adapter.configure(timeout_s=0.1)
    c = case(adapter, "A01")
    with pytest.raises(TimeoutError, match="timeout"):
        adapter.run_case(c, "m")
    r = run_one(adapter, c, "m", timeout_s=0.1, retries=2, sleep=lambda _s: None)
    assert r.status == TIMEOUT and r.passed is False and r.attempts == 1


def test_case_timeout_scales_with_the_number_of_turns(adapter):
    adapter.configure(timeout_s=100)
    assert adapter.case_timeout({"question": "q"}, 100) == 115
    assert adapter.case_timeout({"turns": ["a", "b", "c"]}, 100) == 315


def test_an_llm_request_failure_is_an_error_even_though_project_1_says_it_handed_off(adapter):
    """Project 1 turns every exception into the hand-off text, and on a refuse case that text passes. The adapter
    hooks project 1's own exception path, so a dead Ollama or a slow model's timed-out request cannot score."""
    c = case(adapter, "R06")  # angry customer: expected to escalate
    out = adapter.run_case(c, "m")  # unscripted: a genuine hand-off decision, no exception
    assert adapter.score(c, out)["passed"] is True
    msg = "request to http://localhost:11434/api/chat failed: ReadTimeout('timed out')"
    script(adapter, c["question"], llm_error=msg)
    with pytest.raises(InfraError, match=r"handed off after an exception.*LLMError: request to http"):
        adapter.run_case(c, "m")
    r = run_one(adapter, c, "m", retries=0, sleep=lambda _s: None)
    assert r.status == "error" and r.passed is False and "LLMError" in r.reasons[0]


def test_what_project_1_would_have_scored_is_a_pass_so_the_deviation_is_deliberate(adapter):
    """The same reply, scored by project 1's own scorer, passes the refuse case. The harness records an error."""
    c = case(adapter, "R06")
    script(adapter, c["question"], llm_error="request to http://x failed: ReadTimeout")
    p1_record = adapter.m.p1_run_case(
        c, "agent", adapter.m.run_turn, 5.0
    )  # what project 1's own eval records
    assert p1_record["passed"] is True and p1_record["reply"]["escalated"] is True
    with pytest.raises(InfraError):
        adapter.run_case(c, "m")  # the harness refuses to count it


def test_an_empty_model_answer_is_the_models_failure_and_scored_as_project_1_scores_it(adapter):
    c = case(adapter, "R06")
    script(adapter, c["question"], empty=True)
    out = adapter.run_case(c, "m")  # no InfraError: the model returned nothing
    assert out["model_failures"] == ["empty answer"]
    assert (
        adapter.score(c, out)["passed"] is True
    )  # the hand-off escalates, which is what a refuse case wants
    a = case(adapter, "A01")
    script(adapter, a["question"], empty=True)
    assert adapter.score(a, adapter.run_case(a, "m"))["passed"] is False  # and an answer case fails


def test_a_database_or_other_exception_in_the_agent_is_an_error(adapter):
    script(adapter, A01, text="x", **{"raise": "database is gone"})
    with pytest.raises(InfraError, match="RuntimeError: database is gone"):
        adapter.run_case(case(adapter, "A01"), "m")


def test_a_slow_turns_late_failure_cannot_leak_into_the_next_case(adapter):
    """A timed-out turn keeps running; when it fails later, that must not make the next case an error."""
    import time

    slow = case(adapter, "A02")
    script(adapter, slow["question"], sleep=0.4, llm_error="request to http://x failed: ReadTimeout")
    adapter.configure(timeout_s=0.1)
    with pytest.raises(TimeoutError):
        adapter.run_case(slow, "m")
    assert len(adapter.orphan_threads()) == 1  # the abandoned turn is still working
    script(adapter, A01, **GOOD_A01)
    assert adapter.score(case(adapter, "A01"), adapter.run_case(case(adapter, "A01"), "m"))["passed"] is True
    deadline = time.time() + 3
    while adapter.orphan_threads() and time.time() < deadline:
        time.sleep(0.05)
    assert adapter.orphan_threads() == []  # the orphan finished (and logged its failure to its own session)
    script(adapter, A01, **GOOD_A01)
    assert adapter.run_case(case(adapter, "A01"), "m")["reply"]["text"]  # still clean


def test_the_clock_is_frozen_to_the_dataset_now_and_recorded(adapter):
    from harness.adapters.support_agent import DATASET_NOW

    assert DATASET_NOW == "2026-11-10T09:00:00+05:30"
    script(adapter, A01, **GOOD_A01)
    out = adapter.run_case(case(adapter, "A01"), "m")
    assert adapter.m.agent.CLOCKS_SEEN == [DATASET_NOW] and adapter.m.settings.clock_now == DATASET_NOW
    assert out["clock_now"] == DATASET_NOW
    prov = adapter.provenance()
    assert prov["config"]["clock_now"] == DATASET_NOW
    assert prov["config"]["variant"] == "agent" and prov["config"]["judge"] is None


def test_a_different_clock_changes_the_fingerprint_so_runs_are_not_mixed(stub_repo, tmp_path):
    from harness.run import fingerprint_of

    a = SupportAgentAdapter(repo=stub_repo)
    fp_a = fingerprint_of(a.provenance(), a.cases())
    a.close()
    for name in [n for n in sys.modules if n == "app" or n.startswith(("app.", "evals"))]:
        del sys.modules[name]
    b = SupportAgentAdapter(repo=stub_repo, clock_now="2026-11-20T09:00:00+05:30")
    fp_b = fingerprint_of(b.provenance(), b.cases())
    b.close()
    assert (
        fp_a[:5] == fp_b[:5]
        and fp_a[5] == "2026-11-10T09:00:00+05:30"
        and fp_b[5] == "2026-11-20T09:00:00+05:30"
    )


def test_a_feature_without_a_clock_setting_is_a_preflight_problem_and_records_no_clock(
    stub_repo, monkeypatch
):
    cfg = stub_repo / "app" / "config.py"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("        self.clock_now = None\n", ""), encoding="utf-8"
    )
    ad = SupportAgentAdapter(repo=stub_repo)
    try:
        monkeypatch.setattr(ad, "ollama_models", lambda: ["fusion27b:iq3"])
        monkeypatch.setattr(ad, "db_ready", lambda: [])
        problems = ad.preflight("fusion27b:iq3")
        assert len(problems) == 1 and "no `clock_now` setting" in problems[0]
        assert ad.provenance()["config"]["clock_now"] is None  # honest: it ran on the wall clock
        assert not hasattr(ad.m.settings, "clock_now")
    finally:
        ad.close()
    off = SupportAgentAdapter(
        repo=stub_repo, clock_now=None
    )  # explicitly unpinned: nothing to complain about
    try:
        monkeypatch.setattr(off, "ollama_models", lambda: ["fusion27b:iq3"])
        monkeypatch.setattr(off, "db_ready", lambda: [])
        assert off.preflight("fusion27b:iq3") == []
    finally:
        off.close()


def test_the_scorer_is_part_of_the_provenance(adapter, stub_repo):
    import hashlib

    prov = adapter.provenance()
    assert re.fullmatch(r"[0-9a-f]{12}", prov["scorer_hash"])
    scoring = stub_repo / "evals" / "scoring.py"
    assert prov["scorer_hash"] == hashlib.sha256(b"scoring.py" + scoring.read_bytes()).hexdigest()[:12]
    scoring.write_text(scoring.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert adapter.provenance()["scorer_hash"] != prov["scorer_hash"]


def test_provenance_changes_with_the_prompt_and_code(adapter, stub_repo):
    p1 = adapter.provenance()
    assert p1["git_hash"] is None and p1["git_dirty"] is None  # the temp repo has no commits
    assert re.fullmatch(r"[0-9a-f]{12}", p1["prompt_hash"]) and re.fullmatch(r"[0-9a-f]{12}", p1["code_hash"])
    assert adapter.provenance() == p1 and p1["scorer_hash"]  # stable
    (stub_repo / "app" / "extra.py").write_text("X = 1\n", encoding="utf-8")
    assert adapter.provenance()["code_hash"] != p1["code_hash"]
    (stub_repo / "data" / "KB-001.md").write_text("changed article", encoding="utf-8")
    assert adapter.provenance()["code_hash"] != p1["code_hash"]  # help-centre data is part of the fingerprint
    adapter.m.settings  # noqa: B018 - loaded modules are reused
    sys.modules["app.prompts"].SYSTEM_PROMPT = "a different prompt"
    assert adapter.provenance()["prompt_hash"] != p1["prompt_hash"]


def test_preflight_names_each_problem(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "ollama_models", lambda: ["qwen3:14b", "nomic-embed-text:latest"])
    monkeypatch.setattr(adapter, "db_ready", lambda: [])
    assert adapter.preflight("qwen3:14b") == []
    assert "is not in Ollama" in adapter.preflight("fusion27b:iq3")[0]
    monkeypatch.setattr(adapter, "db_ready", lambda: ["project 1 Postgres is not reachable"])
    assert len(adapter.preflight("fusion27b:iq3")) == 2

    def down():
        raise OSError("refused")

    monkeypatch.setattr(adapter, "ollama_models", down)
    assert "Ollama is not reachable" in adapter.check_ollama("qwen3:14b")[0]
    assert "ANTHROPIC_API_KEY" in adapter.check_ollama("claude-sonnet-5-5")[0]


def test_warmup_loads_the_model_with_a_long_timeout_and_restores_it(adapter):
    adapter.warmup("qwen3:14b")
    calls = adapter.m.llm.CALLS
    assert calls[0] == ("chat", "qwen3:14b", 900.0)  # long timeout during the first load
    assert calls[1][:2] == ("embed", "qwen3:14b") and calls[1][2] == "query"
    assert adapter.m.settings.llm_timeout_s == 120.0


def test_harness_end_to_end_on_the_stub(adapter, store):
    script(adapter, A01, **GOOD_A01)
    script(
        adapter,
        case(adapter, "A02")["question"],
        text="Orders below INR 999 pay INR 79 [KB-001].",
        citations=["KB-001"],
        pt=2000,
        ct=60,
    )
    s = execute_run(adapter, "fusion27b:iq3", store, ids=["A01", "A02"], preflight=False, log=lambda _m: None)
    assert s.run.status == COMPLETE and s.passed == 2 and s.run.suite == "support_agent"
    assert [r.prompt_tokens for r in s.results] == [1500, 2000]
    assert s.run.prompt_hash and s.run.code_hash and s.run.case_set_hash
    assert s.run.scorer_hash and s.run.clock_now == "2026-11-10T09:00:00+05:30"
    assert s.run.config["n_cases_total"] == 60 and s.run.config["limit"] is None and not s.run.is_full_suite
    assert s.run.config["variant"] == "agent" and s.run.config["judge"] is None
    assert adapter.m.llm.CALLS[0][0] == "chat"  # the warm-up ran
    assert all(r.status == OK for r in s.results)


def test_one_suite_per_process_is_enforced(adapter, tmp_path):
    adapter.cases()  # loads the stub repo's app and evals
    other = tmp_path / "p4"
    (other / "app").mkdir(parents=True)
    (other / "app" / "__init__.py").write_text("", encoding="utf-8")
    (other / "app" / "agent.py").write_text("def run_turn(*a, **k): pass\n", encoding="utf-8")
    (other / "evals").mkdir()
    (other / "evals" / "__init__.py").write_text("", encoding="utf-8")
    second = SupportAgentAdapter(repo=other)
    with pytest.raises(AdapterError, match="One suite per process"):
        second.cases()


def test_missing_repo_is_a_clear_error(tmp_path):
    with pytest.raises(AdapterError, match="feature repo not found"):
        SupportAgentAdapter(repo=tmp_path / "nope").cases()


# ---- parity: the harness measures what project 1 measures ----


def recorded():
    return [
        json.loads(line)
        for line in (P1_SNAPSHOT / "agent-results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_parity_with_project_1_on_its_recorded_replies(adapter):
    """Feed every recorded reply through the adapter's score(); pass/fail must equal project 1's own record."""
    by_id = {c["id"]: c for c in adapter.cases()}
    recs = recorded()
    assert len(recs) == 60 and {r["id"] for r in recs} == set(by_id)
    assert (
        0 < sum(1 for r in recs if not r["passed"]) < 60
    )  # both outcomes occur, so the test means something
    mismatches = []
    for r in recs:
        got = adapter.score(by_id[r["id"]], {"reply": r["reply"]})
        if (got["passed"], got["checks"], got["reasons"]) != (r["passed"], r["checks"], r["reasons"]):
            mismatches.append(r["id"])
    assert mismatches == []


@pytest.mark.p1
@pytest.mark.skipif(
    not (REAL_P1 / "evals" / "scoring.py").exists() or os.environ.get("CI"),
    reason="project 1 repo not available",
)
def test_parity_with_the_current_project_1_scoring_code():
    ad = SupportAgentAdapter(repo=REAL_P1)
    try:
        by_id = {c["id"]: c for c in ad.cases()}
        recs = [r for r in recorded() if r["id"] in by_id]
        assert len(recs) == 60
        assert [
            r["id"] for r in recs if ad.score(by_id[r["id"]], {"reply": r["reply"]})["passed"] != r["passed"]
        ] == []
    finally:
        ad.close()


# ---- importing recorded results ----


def test_import_results_file_keeps_project_1s_verdicts():
    run, results = import_results_file(
        P1_SNAPSHOT / "agent-results.jsonl",
        model="fusion27b:iq3",
        cases_path=P1_SNAPSHOT / "evals" / "cases.jsonl",
    )
    recs = recorded()
    assert len(results) == 60 and run.suite == "support_agent" and run.status == COMPLETE
    assert [r.passed for r in results] == [bool(x["passed"]) for x in recs]
    assert results[0].prompt_tokens == recs[0]["reply"]["usage"]["prompt_tokens"]
    assert results[0].latency_ms == recs[0]["wall_ms"]
    assert run.config["imported"] is True and "did not store the model name" in run.config["note"]
    assert run.is_full_suite and run.n_cases_total == 60 and run.clock_now is None and run.scorer_hash
    assert run.case_set_hash and run.model == "fusion27b:iq3"


# ---- the real project 1 with its LLM mocked (local only) ----


def _db_up() -> bool:
    try:
        with socket.create_connection(("localhost", 5433), timeout=1):
            return True
    except OSError:
        return False


def _fake_embed(dims: int):
    def embed(texts, kind="document"):
        out = []
        for t in texts:
            v = [0.0] * dims
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dims] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    return embed


@pytest.mark.p1
@pytest.mark.skipif(
    not (REAL_P1 / "app" / "agent.py").exists() or not _db_up() or os.environ.get("CI"),
    reason="project 1 repo or its Postgres is not available",
)
def test_real_project_1_end_to_end_with_a_mocked_llm(monkeypatch):
    """Project 1's real run_turn, retrieval and tools on its real database; only the LLM is scripted."""
    ad = SupportAgentAdapter(repo=REAL_P1)
    try:
        llm, settings = ad.m.llm, ad.m.settings
        seen_models = []

        def scripted_chat(messages, tools=None):
            seen_models.append(settings.ollama_model)
            user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            if "LC-1192" in user and not any(m["role"] == "tool" for m in messages):
                return llm.LLMResponse(
                    content="", tool_calls=[llm.ToolCall("lookup_order", {"order_id": "LC-1192"})]
                )
            if "LC-1192" in user:
                return llm.LLMResponse(
                    content="Order LC-1192 has shipped with BlueDart.",
                    prompt_tokens=900,
                    completion_tokens=12,
                )
            return llm.LLMResponse(
                content="Standard delivery takes 3 to 6 business days [KB-001].",
                prompt_tokens=1200,
                completion_tokens=20,
            )

        def no_http(*a, **k):
            raise AssertionError("this test must never call Ollama")

        monkeypatch.setattr(llm, "_post", no_http)  # any path to the real model fails the test
        monkeypatch.setattr(llm, "chat", scripted_chat)
        monkeypatch.setattr(llm, "embed", _fake_embed(settings.embed_dims))
        assert ad.db_ready() == []
        cases = {c["id"]: c for c in ad.cases()}
        out_a = ad.run_case(cases["A01"], "qwen3:14b")
        assert ad.score(cases["A01"], out_a)["passed"] is True
        assert out_a["usage"]["prompt_tokens"] == 1200 and "KB-001" in out_a["reply"]["citations"]
        out_t = ad.run_case(cases["T01"], "qwen3:14b")
        scored = ad.score(cases["T01"], out_t)
        assert [c["name"] for c in out_t["reply"]["tool_calls"]] == ["lookup_order"]
        assert scored["passed"] is True, scored
        assert (
            out_t["usage"]["prompt_tokens"] == 900 and out_t["reply"]["usage"]["llm_calls"] == 2
        )  # tool call, then answer
        assert set(seen_models) == {"qwen3:14b"}
        prov = ad.provenance()
        assert prov["prompt_hash"] and prov["code_hash"]
    finally:
        ad.close()


def test_stub_files_are_plain_python():
    for rel, body in STUB_FILES.items():
        compile(body, rel, "exec")
    assert Path(P1_SNAPSHOT / "evals" / "scoring.py").exists()


@pytest.mark.p1
@pytest.mark.skipif(
    not (REAL_P1 / "app" / "agent.py").exists() or not _db_up() or os.environ.get("CI"),
    reason="project 1 repo or its Postgres is not available",
)
def test_a_full_harness_run_on_real_project_1_is_comparable_with_the_committed_fixtures(monkeypatch, store):
    """All 60 real cases through harness.run on project 1's real retrieval, tools and database, LLM mocked.

    The answers are canned, so the pass rate means nothing. What this proves: every case flows through
    without errors, the run carries project 1's provenance, and its case-file hash equals the committed
    fixtures', so a live run can be gated against the recorded baseline.
    """
    from harness.compare import not_comparable
    from harness.storage import read_run_file
    from tests.conftest import FIXTURES

    ad = SupportAgentAdapter(repo=REAL_P1)
    try:
        llm, settings = ad.m.llm, ad.m.settings

        def no_http(*a, **k):
            raise AssertionError("this test must never call Ollama")

        monkeypatch.setattr(llm, "_post", no_http)
        monkeypatch.setattr(
            llm,
            "chat",
            lambda messages, tools=None: llm.LLMResponse(content="Please contact support.", prompt_tokens=10),
        )
        monkeypatch.setattr(llm, "embed", _fake_embed(settings.embed_dims))
        s = execute_run(ad, "fusion27b:iq3", store, preflight=False, warmup=False, log=lambda _m: None)
        assert s.run.status == COMPLETE and len(s.results) == 60 and s.errors == 0
        assert s.run.prompt_hash and s.run.code_hash and s.run.config["judge"] is None
        baseline_run, baseline_results = read_run_file(FIXTURES / "support_agent-p1-run3-agent.run.json")
        assert not_comparable(baseline_run, baseline_results, s.run, s.results) == []
        assert all(r.output["reply"]["text"] for r in s.results)
    finally:
        ad.close()


def test_the_frozen_project_1_snapshot_is_byte_for_byte_unchanged():
    """A formatter or an editor must not touch the vendored copy: parity is only meaningful against the original."""
    meta = json.loads((P1_SNAPSHOT / "SNAPSHOT.json").read_text(encoding="utf-8"))
    assert {"evals/scoring.py", "evals/run.py", "evals/cases.jsonl", "agent-results.jsonl"} <= set(
        meta["sha256"]
    )
    for rel, digest in meta["sha256"].items():
        assert hashlib.sha256((P1_SNAPSHOT / rel).read_bytes()).hexdigest() == digest, rel


# ---- real project 1: the exception path and the clock (local only; its LLM is mocked) ----

real_p1_available = pytest.mark.skipif(
    not (REAL_P1 / "app" / "agent.py").exists() or not _db_up() or os.environ.get("CI"),
    reason="project 1 repo or its Postgres is not available",
)


def _mock_llm(ad, monkeypatch, chat):
    llm, settings = ad.m.llm, ad.m.settings

    def no_http(*a, **k):
        raise AssertionError("this test must never call Ollama")

    monkeypatch.setattr(llm, "_post", no_http)
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", _fake_embed(settings.embed_dims))
    return llm


@pytest.mark.p1
@real_p1_available
def test_real_project_1_llm_failures_are_errors_but_its_own_scorer_would_have_passed_them(monkeypatch):
    """The refuse case R06 expects an escalation. A failed request makes project 1 hand off, which passes."""
    ad = SupportAgentAdapter(repo=REAL_P1)
    try:
        llm = ad.m.llm
        calls = {"n": 0}

        def failing_chat(messages, tools=None):
            calls["n"] += 1
            raise llm.LLMError("request to http://localhost:11434/api/chat failed: ReadTimeout('timed out')")

        _mock_llm(ad, monkeypatch, failing_chat)
        r06 = next(c for c in ad.cases() if c["id"] == "R06")
        p1_record = ad.m.p1_run_case(r06, "agent", ad.m.run_turn, 30.0)  # project 1's own eval record
        assert (
            p1_record["passed"] is True and p1_record["reply"]["escalated"] is True
        )  # it would have scored a PASS
        with pytest.raises(InfraError, match=r"handed off after an exception.*LLMError: request to http"):
            ad.run_case(r06, "fusion27b:iq3")  # the harness records an error instead
        r = run_one(ad, r06, "fusion27b:iq3", retries=0, sleep=lambda _s: None)
        assert r.status == "error" and r.passed is False
        assert calls["n"] >= 2
    finally:
        ad.close()


@pytest.mark.p1
@real_p1_available
def test_real_project_1_a_timeout_on_the_second_llm_call_is_an_error_not_a_refusal_pass(monkeypatch):
    """A slow model: the first call (a tool call) works, the second one hits the 120 s httpx timeout."""
    ad = SupportAgentAdapter(repo=REAL_P1)
    try:
        llm = ad.m.llm
        calls = {"n": 0}

        def chat(messages, tools=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return llm.LLMResponse(
                    content="", tool_calls=[llm.ToolCall("lookup_order", {"order_id": "LC-1192"})]
                )
            raise llm.LLMError("request to http://localhost:11434/api/chat failed: ReadTimeout('timed out')")

        _mock_llm(ad, monkeypatch, chat)
        t01 = next(c for c in ad.cases() if c["id"] == "T01")
        with pytest.raises(InfraError, match="ReadTimeout"):
            ad.run_case(t01, "qwen3:14b")
        assert calls["n"] == 2
    finally:
        ad.close()


@pytest.mark.p1
@real_p1_available
def test_real_project_1_an_empty_answer_is_the_models_failure(monkeypatch):
    ad = SupportAgentAdapter(repo=REAL_P1)
    try:
        llm = ad.m.llm
        _mock_llm(ad, monkeypatch, lambda messages, tools=None: llm.LLMResponse(content="", prompt_tokens=3))
        r06 = next(c for c in ad.cases() if c["id"] == "R06")
        out = ad.run_case(r06, "fusion27b:iq3")  # no InfraError: project 1 raised LLMError("empty answer")
        assert out["model_failures"] == ["empty answer"] and out["reply"]["escalated"] is True
        assert ad.score(r06, out)["passed"] is True  # scored exactly as project 1 scores it
    finally:
        ad.close()


@pytest.mark.p1
@real_p1_available
def test_real_project_1_runs_under_the_frozen_dataset_clock(monkeypatch):
    """Orders have fixed dates: the tool's 'today' must be the dataset's, not the wall clock's."""
    from harness.adapters.support_agent import DATASET_NOW

    ad = SupportAgentAdapter(repo=REAL_P1)
    try:
        llm = ad.m.llm
        state = {"n": 0}

        def chat(messages, tools=None):
            state["n"] += 1
            if state["n"] == 1:
                return llm.LLMResponse(
                    content="", tool_calls=[llm.ToolCall("lookup_order", {"order_id": "LC-1151"})]
                )
            return llm.LLMResponse(content="Your return window is open.", prompt_tokens=5)

        _mock_llm(ad, monkeypatch, chat)
        t09 = next(c for c in ad.cases() if c["id"] == "T09")
        out = ad.run_case(t09, "fusion27b:iq3")
        assert ad.m.settings.clock_now == DATASET_NOW and out["clock_now"] == DATASET_NOW
        result = out["reply"]["tool_calls"][0]["result"]
        assert result["today"].startswith("2026-11-10") and result["days_since_delivery"] == 15
        assert result["return_window_open"] is True and result["return_window_ends"].startswith("2026-11-25")
        assert ad.provenance()["config"]["clock_now"] == DATASET_NOW
        assert ad.clock_ready() == []
    finally:
        ad.close()
