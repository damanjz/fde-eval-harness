"""The adapter interface, shared helpers, and the suite registry.

An adapter wraps one AI feature. The harness owns timing, token accounting, retries, timeouts and
storage; the adapter owns the domain logic: which cases exist, how to run one, how to score it.

Contract (see docs/runbook.md, "Add a suite"):

    cases() -> list[dict]            each has at least "id" and "category"
    run_case(case, model) -> dict    the output; put token usage in output["usage"]
                                     {"prompt_tokens": int, "completion_tokens": int}
    score(case, output) -> dict      {"passed": bool, "checks": {name: bool}, "reasons": [str]}

Raise InfraError from run_case when the model or a backing service is unreachable: the case is then
recorded as an error (not a model failure) and is re-run by --resume. Errors and timeouts still count
as failures in compare and gate, never as skipped.

One suite per process: feature repos such as projects 1 and 4 both have top-level `app` and `evals`
packages, so two of them must never be imported into one interpreter. `harness.run` loads one adapter;
`harness.matrix` starts a subprocess per (suite, model).
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import threading
from pathlib import Path
from typing import Any


class AdapterError(Exception):
    """Base class for adapter problems."""


class InfraError(AdapterError):
    """The model or a backing service failed; this says nothing about the model's quality."""


class Adapter:
    """Base class with defaults. Subclasses set `name` and implement cases, run_case and score."""

    name = "base"
    description = ""
    synthetic = False  # True only for fake/test adapters; reports then carry a SYNTHETIC banner

    def cases(self) -> list[dict]:
        raise NotImplementedError

    def run_case(self, case: dict, model: str) -> dict:
        raise NotImplementedError

    def score(self, case: dict, output: dict) -> dict:
        raise NotImplementedError

    def provenance(self) -> dict:
        """Fingerprints of what is under test: git_hash, git_dirty, prompt_hash, code_hash (all optional)."""
        return {}

    def configure(self, timeout_s: float | None = None) -> None:
        """Receive run settings from the harness before the first case (for example the per-case timeout)."""

    def case_timeout(self, case: dict, timeout_s: float | None) -> float | None:
        """Outer time limit for one case. Multi-turn adapters scale it by the number of turns."""
        return timeout_s

    def preflight(self, model: str) -> list[str]:
        """Return human-readable problems that make a live run pointless (service down, model missing)."""
        return []

    def warmup(self, model: str) -> None:
        """One untimed call after a model switch, so model load time does not land in the first case.

        Raise InfraError if the model cannot answer at all. Called once per run, before the first case.
        """

    def orphan_threads(self) -> list[threading.Thread]:
        """Threads still running work for a case that already timed out. The runner waits for them (bounded)
        before the next case so they do not distort its latency. Adapters that start their own threads
        override this; the harness's own per-case thread is tracked by the runner."""
        return []

    def close(self) -> None:
        """Release connections. Called once after the run."""


# ---- helpers for adapters ----


def git_info(repo: Path) -> tuple[str | None, bool | None]:
    """(short commit hash, dirty flag) of a repo, or (None, None) when git is missing or there is no commit."""

    def run(*args: str) -> str | None:
        try:
            p = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    head = run("rev-parse", "--short", "HEAD")
    if not head:
        return None, None
    status = run("status", "--porcelain")
    return head, (bool(status) if status is not None else None)


def tree_hash(
    root: Path, patterns: tuple[str, ...] = ("*.py",), subdirs: tuple[str, ...] = ("",)
) -> str | None:
    """Fingerprint of source files (paths and contents), for repos that have no commit to name."""
    h = hashlib.sha256()
    found = False
    for sub in subdirs:
        base = root / sub
        if not base.is_dir():
            continue
        for pattern in patterns:
            for f in sorted(base.rglob(pattern)):
                if "__pycache__" in f.parts or not f.is_file():
                    continue
                found = True
                h.update(f.relative_to(root).as_posix().encode())
                h.update(f.read_bytes())
    return h.hexdigest()[:12] if found else None


def load_dotenv(path: Path, environ: dict[str, str]) -> int:
    """Read KEY=VALUE lines from a .env file into environ without overriding existing keys."""
    if not path.is_file():
        return 0
    n = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in environ:
            environ[key] = value
            n += 1
    return n


# ---- registry ----

# name -> "module:Class". Imported lazily so a suite's dependencies are needed only when it is used.
REGISTRY: dict[str, str] = {
    "support_agent": "harness.adapters.support_agent:SupportAgentAdapter",
    "ask_your_data": "harness.adapters.ask_your_data:AskYourDataAdapter",
    "fake": "harness.adapters.fake:FakeAdapter",
}


def suite_names() -> list[str]:
    return sorted(REGISTRY)


def get_adapter(name: str, **kwargs: Any) -> Adapter:
    if name not in REGISTRY:
        raise KeyError(f"unknown suite '{name}'. Known suites: {', '.join(suite_names())}")
    module, _, cls = REGISTRY[name].partition(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)
