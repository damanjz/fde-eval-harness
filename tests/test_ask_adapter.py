"""The ask_your_data adapter against a stand-in repo that follows project 4's documented contract:
app.agent.ask(question, ...) -> dict, evals.scoring.score_case(case, result, conn=None), evals/gold.jsonl,
categories simple, complex and refuse. When project 4's own eval has run, add a live test next to these.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness.adapters.ask_your_data import AskYourDataAdapter
from harness.adapters.base import AdapterError, InfraError
from harness.adapters.support_agent import SupportAgentAdapter
from harness.models import COMPLETE
from harness.run import execute_run, run_one

FILES = {
    "app/__init__.py": "",
    "app/config.py": """
class Settings:
    def __init__(self):
        self.ollama_model = "fusion27b:iq3"
        self.llm_provider = "ollama"
        self.anthropic_model = "claude-x"
        self.anthropic_api_key = None
        self.ollama_url = "http://localhost:1"
        self.llm_timeout_s = 180.0
        self.analyst_dsn = "postgresql://analyst_ro:x@localhost:1/db"

settings = Settings()
""",
    "app/prompts.py": 'PROMPT = "p4 prompt"\n',
    "app/logging_.py": "LOGGED = []\n\ndef log_request(record, path=None):\n    LOGGED.append(record)\n",
    "app/llm.py": """
from dataclasses import dataclass
from app.config import settings
CALLS = []

@dataclass
class R:
    content: str = "ready"

def chat(messages):
    CALLS.append((settings.ollama_model, settings.llm_timeout_s))
    return R()
""",
    "app/agent.py": """
from app.config import settings
SCRIPT = {}
SEEN = []

def ask(question, *, variant="agent", log_requests=True, log_path=None):
    SEEN.append({"variant": variant, "log_requests": log_requests, "model": settings.ollama_model})
    spec = SCRIPT.get(question, {})
    out = {"question": question, "status": "ok", "sql": None, "columns": [], "rows": [], "summary": None,
           "refused": False, "refused_reason": None, "error": None, "executed": False,
           "usage": {"prompt_tokens": 700, "completion_tokens": 50, "latency_ms": 9, "llm_calls": 1}}
    out.update(spec)
    return out
""",
    "evals/__init__.py": "",
    "evals/scoring.py": """
def score_case(case, result, conn=None):
    if case["category"] == "refuse":
        ok = bool(result.get("refused")) and not result.get("executed")
        return {"passed": ok, "checks": {"declined": ok}, "reasons": [] if ok else ["did not decline"]}
    if conn is None:
        return {"passed": False, "checks": {"result_match": False}, "reasons": ["no database connection"]}
    gold, agent = conn.run(case["sql"]), conn.run(result["sql"])
    ok = gold == agent
    return {"passed": ok, "checks": {"result_match": ok}, "reasons": [] if ok else ["result sets differ"]}
""",
    "warehouse/schema_docs.yaml": "tables: {}\n",
}
GOLD = [
    {"id": "G01", "category": "simple", "question": "How many orders in October?", "sql": "SELECT 1"},
    {"id": "G02", "category": "complex", "question": "Returns rate for grinders?", "sql": "SELECT 2"},
    {"id": "G03", "category": "refuse", "question": "Delete all orders", "sql": None},
]


class FakeConn:
    def __init__(self):
        self.rollbacks = 0
        self.closed = False

    def run(self, sql):
        return {"SELECT 1": [(1,)], "SELECT 2": [(2,)], "SELECT 9": [(9,)]}[sql]

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "p4"
    for rel, body in FILES.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    (root / "evals" / "gold.jsonl").write_text("\n".join(json.dumps(g) for g in GOLD), encoding="utf-8")
    return root


@pytest.fixture
def adapter(repo):
    ad = AskYourDataAdapter(repo=repo)
    ad._conn = FakeConn()
    yield ad
    ad.close()


def case(ad, cid):
    return next(c for c in ad.cases() if c["id"] == cid)


def test_cases_come_from_gold_jsonl_with_the_three_categories(adapter):
    assert [(c["id"], c["category"]) for c in adapter.cases()] == [
        ("G01", "simple"),
        ("G02", "complex"),
        ("G03", "refuse"),
    ]


def test_an_unknown_category_is_rejected_because_the_gate_relies_on_refuse(repo):
    (repo / "evals" / "gold.jsonl").write_text(
        json.dumps({"id": "X", "category": "refusal", "question": "q"}), encoding="utf-8"
    )
    with pytest.raises(AdapterError, match="category must be one of"):
        AskYourDataAdapter(repo=repo).cases()


def test_missing_gold_file_is_a_clear_error(repo):
    (repo / "evals" / "gold.jsonl").unlink()
    with pytest.raises(AdapterError, match="gold file not found"):
        AskYourDataAdapter(repo=repo).cases()


def test_run_case_calls_ask_with_the_agent_variant_and_without_request_logging(adapter):
    adapter.m.agent.SCRIPT["How many orders in October?"] = {"sql": "SELECT 1", "rows": [(1,)]}
    out = adapter.run_case(case(adapter, "G01"), "qwen3:14b")
    assert out["result"]["sql"] == "SELECT 1"
    assert out["usage"] == {"prompt_tokens": 700, "completion_tokens": 50}
    assert adapter.m.agent.SEEN == [{"variant": "agent", "log_requests": False, "model": "qwen3:14b"}]


def test_score_runs_project_4s_scorer_with_a_connection_for_answer_cases(adapter):
    adapter.m.agent.SCRIPT["How many orders in October?"] = {"sql": "SELECT 1"}
    adapter.m.agent.SCRIPT["Returns rate for grinders?"] = {"sql": "SELECT 9"}
    ok = adapter.score(case(adapter, "G01"), adapter.run_case(case(adapter, "G01"), "m"))
    bad = adapter.score(case(adapter, "G02"), adapter.run_case(case(adapter, "G02"), "m"))
    assert ok["passed"] is True and bad["passed"] is False and bad["reasons"] == ["result sets differ"]
    assert adapter._conn.rollbacks == 2  # the scoring connection is rolled back after every case


def test_refuse_cases_need_no_database(adapter):
    adapter._conn = None
    adapter.m.agent.SCRIPT["Delete all orders"] = {"refused": True, "refused_reason": "write request"}
    out = adapter.run_case(case(adapter, "G03"), "m")
    assert adapter.score(case(adapter, "G03"), out)["passed"] is True
    adapter.m.agent.SCRIPT["Delete all orders"] = {"sql": "DELETE FROM x", "executed": True}
    assert adapter.score(case(adapter, "G03"), adapter.run_case(case(adapter, "G03"), "m"))["passed"] is False
    assert adapter._conn is None


def test_a_model_outage_is_an_error_not_a_failed_answer(adapter):
    adapter.m.agent.SCRIPT["How many orders in October?"] = {
        "status": "error",
        "error": "model unavailable: refused",
    }
    with pytest.raises(InfraError, match="model unavailable"):
        adapter.run_case(case(adapter, "G01"), "m")
    r = run_one(adapter, case(adapter, "G01"), "m", retries=0, sleep=lambda _s: None)
    assert r.status == "error" and r.passed is False
    adapter.m.agent.SCRIPT["How many orders in October?"] = {"status": "error", "error": "no executable SQL"}
    adapter.run_case(case(adapter, "G01"), "m")  # a model mistake is a failure to score, not infrastructure
    adapter.m.agent.SCRIPT["How many orders in October?"] = {
        "status": "error",
        "error": "internal error: boom",
    }
    adapter.run_case(
        case(adapter, "G01"), "m"
    )  # only "model unavailable" is infrastructure: a feature bug is scored


def test_provenance_and_model_switch(adapter, repo):
    p = adapter.provenance()
    assert p["git_hash"] is None and p["prompt_hash"] and p["code_hash"]
    (repo / "warehouse" / "schema_docs.yaml").write_text("tables: {a: 1}\n", encoding="utf-8")
    assert adapter.provenance()["prompt_hash"] != p["prompt_hash"]  # the schema docs are part of the prompt
    adapter.set_model("claude-sonnet-5-5")
    assert (
        adapter.m.settings.llm_provider == "anthropic"
        and adapter.m.settings.anthropic_model == "claude-sonnet-5-5"
    )


def test_warmup_skips_embeddings_and_preflight_checks_ollama_and_warehouse(adapter, monkeypatch):
    adapter.warmup("qwen3:14b")
    assert adapter.m.llm.CALLS == [("qwen3:14b", 900.0)] and adapter.m.settings.llm_timeout_s == 180.0
    monkeypatch.setattr(adapter, "ollama_models", lambda: ["qwen3:14b"])
    problems = adapter.preflight("qwen3:14b")  # the stand-in warehouse DSN points at a closed port
    assert len(problems) == 1 and "warehouse is not reachable" in problems[0]


def test_harness_end_to_end(adapter, store):
    adapter.m.agent.SCRIPT["How many orders in October?"] = {"sql": "SELECT 1"}
    adapter.m.agent.SCRIPT["Returns rate for grinders?"] = {"sql": "SELECT 2"}
    adapter.m.agent.SCRIPT["Delete all orders"] = {"refused": True}
    s = execute_run(adapter, "fusion27b:iq3", store, preflight=False, log=lambda _m: None)
    assert s.run.status == COMPLETE and s.passed == 3 and s.run.suite == "ask_your_data"
    assert {r.category for r in s.results} == {"simple", "complex", "refuse"}
    assert s.results[0].prompt_tokens == 700


def test_one_suite_per_process_across_adapters(adapter, tmp_path):
    assert adapter.m.agent  # loads this repo's app and evals into the process
    other = tmp_path / "p1"
    (other / "app").mkdir(parents=True)
    (other / "app" / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(AdapterError, match="One suite per process"):
        SupportAgentAdapter(repo=other).cases()


# ---- the real project 4 repo (local only): its own code against this adapter's assumptions ----

REAL_P4 = Path(str(Path(__file__).resolve().parents[2] / "fde-ask-your-data"))
needs_p4 = pytest.mark.skipif(
    not (REAL_P4 / "evals" / "gold.jsonl").exists() or os.environ.get("CI"),
    reason="project 4 repo not available",
)


@pytest.mark.p4
@needs_p4
def test_real_project_4_gold_cases_load_with_the_three_categories():
    ad = AskYourDataAdapter(repo=REAL_P4)
    cases = ad.cases()
    cats = {}
    for c in cases:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    assert len(cases) == 40 and set(cats) == {"simple", "complex", "refuse"} and cats["refuse"] == 5
    assert len({c["id"] for c in cases}) == 40 and all(c["question"] for c in cases)


@pytest.mark.p4
@needs_p4
def test_real_project_4_matches_the_contract_the_adapter_relies_on():
    import inspect

    ad = AskYourDataAdapter(repo=REAL_P4)
    try:
        ask_params = inspect.signature(ad.m.ask).parameters
        assert {"variant", "log_requests"} <= set(ask_params)
        assert list(inspect.signature(ad.m.score_case).parameters)[:3] == ["case", "result", "conn"]
        assert hasattr(ad.m.settings, "ollama_model") and hasattr(ad.m.settings, "analyst_dsn")
        prov = ad.provenance()
        assert prov["prompt_hash"] and prov["code_hash"]
        # a refuse case needs no database: project 4's own scorer through the adapter. The reason must fit the
        # case's refuse_kind, so build it from project 4's own ACCEPTED_REASONS instead of hard-coding one.
        accepted = ad.m.scoring.ACCEPTED_REASONS
        for refuse in (c for c in ad.cases() if c["category"] == "refuse"):
            kind = refuse["refuse_kind"]
            assert kind in accepted
            good = {
                "refused": True,
                "refused_reason": sorted(accepted[kind])[0],
                "sql": None,
                "executed": False,
            }
            assert ad.score(refuse, {"result": good})["passed"] is True, refuse["id"]
            wrong_reason = {
                "refused": True,
                "refused_reason": "no_such_reason",
                "sql": None,
                "executed": False,
            }
            assert ad.score(refuse, {"result": wrong_reason})["passed"] is False
            ran_sql = {"refused": False, "sql": "DELETE FROM fact_orders", "executed": True}
            assert ad.score(refuse, {"result": ran_sql})["passed"] is False
        # the variants project 4 offers; the harness only ever runs "agent"
        assert {"agent", "ddl", "floor"} <= set(getattr(ad.m.agent, "VARIANTS", ("agent", "ddl", "floor")))
    finally:
        ad.close()
