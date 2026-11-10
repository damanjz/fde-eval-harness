"""Shared helpers: result builders, a scripted adapter, and isolation for feature-repo imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from harness.adapters.base import Adapter
from harness.models import COMPLETE, ERROR, OK, CaseResult, Run
from harness.storage import Store

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures" / "runs"
P1_SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "p1"
REAL_P1 = Path(str(Path(__file__).resolve().parents[2] / "fde-support-agent"))


@pytest.fixture(autouse=True)
def clean_feature_modules():
    """Feature repos share top-level names (app, evals). Each test starts and ends with none loaded."""

    def purge() -> None:
        for name in [n for n in sys.modules if n in ("app", "evals") or n.startswith(("app.", "evals."))]:
            del sys.modules[name]

    before = list(sys.path)
    purge()
    yield
    purge()
    sys.path[:] = before


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "runs.db") as s:
        yield s


def res(
    case_id: str,
    category: str,
    passed: bool,
    *,
    status: str = OK,
    latency_ms: float = 1000.0,
    pt: int = 100,
    ct: int = 10,
    reasons: list[str] | None = None,
) -> CaseResult:
    if status != OK:
        passed = False
    return CaseResult(
        case_id=case_id,
        category=category,
        passed=passed,
        status=status,
        reasons=reasons if reasons is not None else ([] if passed else [f"{case_id} failed"]),
        latency_ms=latency_ms,
        prompt_tokens=pt,
        completion_tokens=ct,
    )


def make_run(run_id: str = "r", suite: str = "s", model: str = "m", **kw) -> Run:
    kw.setdefault("status", COMPLETE)
    return Run(run_id=run_id, suite=suite, model=model, started_at="2026-11-10T10:00:00", **kw)


def full_run(run_id: str = "r", results: list[CaseResult] | None = None, **kw) -> Run:
    """A run that covered its whole case file and carries the hashes a gate needs."""
    config = {
        "n_cases_total": len(results or []),
        "limit": None,
        "ids": None,
        "scorer_hash": "s1",
        **kw.pop("config", {}),
    }
    kw.setdefault("case_set_hash", "h1")
    return make_run(run_id, config=config, **kw)


def make_results(spec: str, categories: dict[str, str] | None = None) -> list[CaseResult]:
    """spec like 'A1:a+ A2:a- T1:t+ R1:r-' : id, category letter, + pass or - fail, e for error."""
    cats = categories or {"a": "answer", "t": "tool", "r": "refuse"}
    out = []
    for tok in spec.split():
        cid, rest = tok.split(":")
        cat, mark = cats[rest[0]], rest[1]
        out.append(res(cid, cat, mark == "+", status=ERROR if mark == "e" else OK))
    return out


class ScriptedAdapter(Adapter):
    """A fake feature whose behaviour is scripted per case id, for runner tests."""

    name = "scripted"
    synthetic = True

    def __init__(
        self,
        n: int = 5,
        fail: set[str] | None = None,
        raises: dict[str, Exception] | None = None,
        score_raises: set[str] | None = None,
        prompt_hash: str = "p1",
    ):
        self.n = n
        self.fail = fail or set()
        self.raises = raises or {}
        self.score_raises = score_raises or set()
        self.prompt_hash = prompt_hash
        self.calls: list[str] = []
        self.warmups: list[str] = []
        self.preflight_problems: list[str] = []
        self.configured: float | None = None
        self.interrupt_at: str | None = None
        self.closed = False

    def cases(self):
        return [{"id": f"C{i}", "category": "answer" if i % 2 else "refuse"} for i in range(1, self.n + 1)]

    def configure(self, timeout_s=None):
        self.configured = timeout_s

    def preflight(self, model):
        return list(self.preflight_problems)

    def warmup(self, model):
        self.warmups.append(model)

    def provenance(self):
        return {
            "git_hash": "abc1234",
            "git_dirty": False,
            "prompt_hash": self.prompt_hash,
            "code_hash": "c1",
            "scorer_hash": "s1",
        }

    def run_case(self, case, model):
        self.calls.append(case["id"])
        if case["id"] == self.interrupt_at:
            raise KeyboardInterrupt
        exc = self.raises.get(case["id"])
        if exc:
            if isinstance(exc, list):
                raise exc.pop(0) if exc else RuntimeError("exhausted")
            raise exc
        return {"ok": case["id"] not in self.fail, "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    def score(self, case, output):
        if case["id"] in self.score_raises:
            raise ValueError("scorer bug")
        return {
            "passed": output["ok"],
            "checks": {"ok": output["ok"]},
            "reasons": [] if output["ok"] else ["bad"],
        }

    def close(self):
        self.closed = True
