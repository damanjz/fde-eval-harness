"""Plain data types shared by every harness module."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

RUN_FORMAT = "fde-harness-run/1"

# status of one case result
OK, ERROR, TIMEOUT = "ok", "error", "timeout"
# status of a run
RUNNING, COMPLETE, PARTIAL = "running", "complete", "partial"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "x"


def stable_hash(obj: Any) -> str:
    """Short hash of any JSON-able object (sorted keys), for fingerprints."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    status: str = OK  # ok | error | timeout; error and timeout always have passed=False
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0  # wall time of the final attempt, measured by the harness
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_inr: float = 0.0
    attempts: int = 1
    output: dict | None = None
    error: str | None = None
    ts: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        # an errored or timed-out case is never a pass, whatever a file or an adapter says
        if self.status != OK:
            self.passed = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CaseResult:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Run:
    run_id: str
    suite: str
    model: str
    status: str = RUNNING
    label: str | None = None
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    git_hash: str | None = None  # of the feature repo, when it has commits
    prompt_hash: str | None = None
    code_hash: str | None = None  # fingerprint of the feature source, for repos without commits
    case_set_hash: str | None = None
    n_cases: int = 0  # cases selected for this run
    config: dict = field(default_factory=dict)  # timeout, retries, prices, git_dirty, import source, ...
    notes: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Run:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # facts kept in config, so the database schema never has to change
    @property
    def scorer_hash(self) -> str | None:
        """Hash of the feature's scoring code (evals/scoring.py): two runs scored differently are not comparable."""
        return self.config.get("scorer_hash")

    @property
    def clock_now(self) -> str | None:
        """The frozen 'now' the feature ran under (ISO datetime), or None when it used the wall clock."""
        return self.config.get("clock_now")

    @property
    def variant(self) -> str | None:
        return self.config.get("variant")

    @property
    def n_cases_total(self) -> int | None:
        """How many cases the whole case file has. Older runs did not record it: a run with no limit and no
        ids covered the whole file, so its own case count is the total."""
        total = self.config.get("n_cases_total")
        if total is not None:
            return int(total)
        if not self.config.get("limit") and not self.config.get("ids") and "limit" in self.config:
            return self.n_cases
        return None

    @property
    def is_full_suite(self) -> bool:
        """Selected every case of the case file (no --limit, no --ids)."""
        total = self.n_cases_total
        return (
            not self.config.get("limit")
            and not self.config.get("ids")
            and total is not None
            and self.n_cases == total
        )

    def fingerprint(self) -> tuple:
        """What must match for a run to be resumed or reused: same feature code, prompt, cases, scorer and clock."""
        return (
            self.git_hash,
            self.prompt_hash,
            self.code_hash,
            self.case_set_hash,
            self.scorer_hash,
            self.clock_now,
        )
