"""A synthetic suite for tests and the CI smoke run. Its numbers are made up and never describe a real model.

Each (case, model) pair passes or fails deterministically from a hash, with a pass probability chosen
from the model name, so repeated runs of the same model agree and different models differ.
"""

from __future__ import annotations

import hashlib

from harness.adapters.base import Adapter

CATEGORIES = ("answer", "tool", "refuse")
# pass probability by model name; anything else uses DEFAULT_RATE
RATES = {"fake-strong": 0.92, "fake-mid": 0.78, "fake-weak": 0.5}
DEFAULT_RATE = 0.8


def _unit(*parts: str) -> float:
    """Deterministic float in [0, 1) from the parts."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class FakeAdapter(Adapter):
    name = "fake"
    description = "Synthetic suite for tests and CI. Numbers are not real."
    synthetic = True

    def __init__(self, n_cases: int = 30, rates: dict[str, float] | None = None):
        self.n_cases = n_cases
        self.rates = {**RATES, **(rates or {})}

    def cases(self) -> list[dict]:
        out = []
        for i in range(self.n_cases):
            cat = CATEGORIES[i % len(CATEGORIES)]
            out.append({"id": f"F{i + 1:03d}", "category": cat, "question": f"synthetic question {i + 1}"})
        return out

    def run_case(self, case: dict, model: str) -> dict:
        rate = self.rates.get(model, DEFAULT_RATE)
        good = _unit(case["id"], model) < rate
        words = 20 + int(_unit(case["id"], "len") * 60)
        return {
            "answer": "ok" if good else "wrong",
            "usage": {"prompt_tokens": 800 + words * 5, "completion_tokens": words},
        }

    def score(self, case: dict, output: dict) -> dict:
        ok = output["answer"] == "ok"
        return {
            "passed": ok,
            "checks": {"answer": ok},
            "reasons": [] if ok else [f"synthetic failure on {case['id']}"],
        }

    def provenance(self) -> dict:
        return {
            "prompt_hash": "fake-prompt-v1",
            "code_hash": "fake-code-v1",
            "scorer_hash": "fake-scorer-v1",
        }
