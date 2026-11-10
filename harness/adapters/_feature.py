"""Shared plumbing for adapters that call a feature repo in-process (projects 1 and 4).

Both repos have top-level `app` and `evals` packages, so only one of them may be imported per
process. `import_feature_repo` enforces that: it refuses to import a repo when `app` or `evals` is
already loaded from somewhere else. `harness.run` loads one adapter per process and `harness.matrix`
starts one subprocess per (suite, model), so the rule holds in normal use.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from harness.adapters.base import Adapter, AdapterError, InfraError, load_dotenv

FEATURE_PACKAGES = ("app", "evals")
WARMUP_TIMEOUT_S = 900.0  # first load of a 27B model from disk can be slow


def import_feature_repo(repo: Path, packages: tuple[str, ...] = FEATURE_PACKAGES) -> None:
    """Make `repo`'s top-level packages importable, or fail if another repo already owns those names."""
    repo = repo.resolve()
    if not repo.is_dir():
        raise AdapterError(f"feature repo not found: {repo} (set the path with the adapter's repo option)")
    for pkg in packages:
        mod = sys.modules.get(pkg)
        origin = getattr(mod, "__file__", None) if mod else None
        if origin and repo not in Path(origin).resolve().parents:
            raise AdapterError(
                f"'{pkg}' is already imported from {Path(origin).resolve().parent}, not {repo}. "
                "One suite per process: two feature repos must never share an interpreter. "
                "Run each suite in its own process (harness.matrix does this)."
            )
    load_dotenv(repo / ".env", os.environ)  # the repo's settings read .env from the working directory
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def file_hash(paths: list[Path]) -> str | None:
    """Short hash over the contents of existing files, in the order given."""
    h = hashlib.sha256()
    found = False
    for p in paths:
        if p.is_file():
            found = True
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12] if found else None


class FeatureRepoAdapter(Adapter):
    """Base for in-process feature adapters. Subclasses implement _load, cases, run_case and score."""

    repo_env = ""  # environment variable that overrides the repo path
    default_repo = Path(".")
    turn_timeout_s = 180.0

    def __init__(self, repo: Path | str | None = None):
        self.repo = Path(repo or os.environ.get(self.repo_env) or self.default_repo)
        self._m: Any = None  # loaded modules, set by _load

    # ---- loading ----

    def _load(self) -> Any:
        raise NotImplementedError

    @property
    def m(self) -> Any:
        if self._m is None:
            self._m = self._load()
        return self._m

    # ---- model switching: the repo's settings are read on every LLM call ----

    def set_model(self, model: str) -> None:
        s = self.m.settings
        if model.startswith("claude"):
            s.llm_provider, s.anthropic_model = "anthropic", model
        else:
            s.llm_provider, s.ollama_model = "ollama", model
            os.environ["OLLAMA_MODEL"] = model

    # ---- reachability ----

    def ollama_models(self) -> list[str]:
        """Names of the models the Ollama server has (GET /api/tags: metadata only, loads nothing)."""
        url = self.m.settings.ollama_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - local URL from settings
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]

    def check_ollama(self, model: str) -> list[str]:
        if model.startswith("claude"):
            return (
                []
                if getattr(self.m.settings, "anthropic_api_key", None)
                else ["ANTHROPIC_API_KEY is not set for a claude model"]
            )
        try:
            names = self.ollama_models()
        except (OSError, urllib.error.URLError, ValueError) as e:
            return [f"Ollama is not reachable at {self.m.settings.ollama_url} ({type(e).__name__})"]
        if model not in names and f"{model}:latest" not in names:
            return [
                f"model '{model}' is not in Ollama (ollama list shows: {', '.join(sorted(names)) or 'nothing'})"
            ]
        return []

    def warm_up_llm(self, model: str, embed: bool = False) -> None:
        """One untimed chat call (and one embedding call when the feature uses retrieval)."""
        self.set_model(model)
        llm = self.m.llm
        s = self.m.settings
        old = s.llm_timeout_s
        s.llm_timeout_s = max(old, WARMUP_TIMEOUT_S)
        try:
            resp = llm.chat([{"role": "user", "content": "Reply with one word: ready"}])
            if not getattr(resp, "content", "x"):
                raise InfraError("warm-up reply was empty")
            if embed:
                llm.embed(["warm-up"], kind="query")
        finally:
            s.llm_timeout_s = old

    def close(self) -> None:
        if self._m is not None:
            db = sys.modules.get("app.db")
            if db is not None and hasattr(db, "reset_pool"):
                db.reset_pool()
