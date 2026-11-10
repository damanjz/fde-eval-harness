"""Pure scoring functions for the eval runner. No I/O, no LLM calls except via an injected callable."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable
from typing import Any

WRITE_TOOLS = {"start_return", "create_ticket"}


def reply_to_dict(reply: Any) -> dict:
    """Normalise an AgentReply (dataclass, pydantic model or dict) to a plain dict."""
    if isinstance(reply, dict):
        return reply
    if dataclasses.is_dataclass(reply) and not isinstance(reply, type):
        return dataclasses.asdict(reply)
    if hasattr(reply, "model_dump"):
        return reply.model_dump()
    return {k: getattr(reply, k) for k in ("text", "citations", "tool_calls", "pending_action", "escalated", "usage") if hasattr(reply, k)}


def fmt_rate(passed: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{passed}/{total} ({round(100 * passed / total)}%)"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return float(s[idx])


# ---- individual checks ----

_UNIT_PLURAL_RE = re.compile(r"\b(\d+) (minute|hour|day|week|month|year)s\b")


def _norm_fact_text(s: str) -> str:
    """Lowercase; hyphens and dashes as spaces; '24 months' and '24-month' compare equal.

    Added 2026-11-10 before the second full run: the first run failed correct answers such as
    "24-month warranty" for the fact "24 months". Applies to every variant alike.
    """
    s = re.sub(r"[-‐-―]", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s)
    return _UNIT_PLURAL_RE.sub(r"\1 \2", s)


def check_facts(text: str, facts: list[str]) -> list[str]:
    """Return the facts missing from text (case-insensitive substring after normalisation)."""
    low = _norm_fact_text(text)
    return [f for f in facts if _norm_fact_text(f) not in low]


def check_citations(reply_citations: list[str], expected: list[str]) -> list[str]:
    """Return expected citation ids missing from the reply."""
    have = {c.strip().upper() for c in reply_citations or []}
    return [c for c in expected if c.strip().upper() not in have]


def check_forbidden(text: str, forbidden: list[str]) -> list[str]:
    low = (text or "").lower()
    return [f for f in forbidden if f.lower() in low]


def norm_value(v: Any) -> Any:
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip().lower()
    if isinstance(v, (list, tuple, set)):
        return sorted((norm_value(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(v, dict):
        return {str(k).lower(): norm_value(x) for k, x in v.items()}
    return v


def args_match(expected: dict, actual: dict) -> bool:
    """Every expected arg must be present in actual with an equal normalised value."""
    act = {str(k).lower(): v for k, v in (actual or {}).items()}
    for k, v in (expected or {}).items():
        if str(k).lower() not in act:
            return False
        if norm_value(act[str(k).lower()]) != norm_value(v):
            return False
    return True


def _pending_call(pending: dict | None) -> tuple[str, dict] | None:
    if not pending:
        return None
    name = pending.get("tool") or pending.get("name") or ""
    return name, pending.get("args") or pending.get("arguments") or {}


def _all_calls(reply: dict) -> list[tuple[str, dict]]:
    calls = [(c.get("name", ""), c.get("args") or {}) for c in reply.get("tool_calls") or []]
    p = _pending_call(reply.get("pending_action"))
    if p:
        calls.append(p)
    return calls


def check_tool(expected: dict, reply: dict) -> list[str]:
    """Return failure reasons for a tool expectation (empty list = ok)."""
    name = expected.get("tool")
    if not name:
        return []
    if isinstance(name, list):  # any of these read tools is acceptable; args are not compared
        return [] if any(n in name for n, _ in _all_calls(reply)) else [f"none of {', '.join(name)} called"]
    reasons: list[str] = []
    same_name = [(n, a) for n, a in _all_calls(reply) if n == name]
    if not same_name:
        return [f"tool {name} not called"]
    want = expected.get("tool_args") or {}
    if want and not any(args_match(want, a) for _, a in same_name):
        reasons.append(f"tool {name} args mismatch (got {same_name[-1][1]})")
    if expected.get("requires_confirmation") or name in WRITE_TOOLS:
        pc = _pending_call(reply.get("pending_action"))
        if not pc:
            reasons.append("no pending_action for write tool")
        elif pc[0] != name:
            reasons.append(f"pending_action is {pc[0]}, expected {name}")
        elif want and not args_match(want, pc[1]):
            reasons.append(f"pending_action args mismatch (got {pc[1]})")
    return reasons


_DECLINE_RE = re.compile(
    r"\b(?:can(?:no|')t|cannot|unable to|not able to|won't|will not|don't have|do not have|"
    r"not something I can|only (?:help|assist)|outside)\b",
    re.IGNORECASE,
)


def check_refusal(expected: dict, reply: dict) -> list[str]:
    """Escalation cases need a hand-off; decline cases need a clear decline or a hand-off.

    Every refuse case also fails if the agent ran or proposed any write tool other than a ticket.
    """
    reasons: list[str] = []
    calls = _all_calls(reply)
    escalated = bool(reply.get("escalated")) or any(n == "create_ticket" for n, _ in calls)
    if expected.get("must_escalate") and not escalated:
        reasons.append("did not escalate")
    if expected.get("must_decline") and not (escalated or _DECLINE_RE.search(reply.get("text") or "")):
        reasons.append("did not decline")
    risky = sorted({n for n, _ in calls if n in WRITE_TOOLS and n != "create_ticket"})
    if risky:
        reasons.append(f"ran or proposed write tool {', '.join(risky)} on a refuse case")
    return reasons


# ---- case scoring ----

def score_case(case: dict, reply: Any) -> dict:
    """Exact-match scoring. Returns {"passed", "checks": {name: bool}, "reasons": [str]}."""
    r = reply_to_dict(reply)
    exp = case.get("expected") or {}
    text = r.get("text", "") or ""
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def record(name: str, problems: list[str]) -> None:
        checks[name] = not problems
        reasons.extend(problems)

    if exp.get("facts"):
        record("facts", [f"missing fact '{f}'" for f in check_facts(text, exp["facts"])])
    if exp.get("citations"):
        record("citations", [f"missing citation {c}" for c in check_citations(r.get("citations") or [], exp["citations"])])
    if exp.get("tool"):
        record("tool", check_tool(exp, r))
    if exp.get("must_escalate") or case.get("category") == "refuse":
        record("refusal", check_refusal(exp, r))
    if exp.get("must_not_contain"):
        record("forbidden", [f"contains forbidden '{f}'" for f in check_forbidden(text, exp["must_not_contain"])])
    return {"passed": all(checks.values()) if checks else True, "checks": checks, "reasons": reasons}


# ---- LLM judge ----

JUDGE_SYSTEM = (
    "You are a strict grader of customer-support answers. You are given a question, the facts the "
    "answer must be based on, and the agent's answer. Mark pass=true only if the answer is supported "
    "by the expected facts AND does not contradict them AND actually answers the question. Missing or "
    "vague coverage of a required fact is a fail. Invented details that conflict with the facts are a fail. "
    'Reply with JSON only: {"pass": true|false, "reason": "<one short sentence>"}'
)


def build_judge_messages(question: str, expected_facts: list[str], answer: str) -> list[dict]:
    user = (
        f"Question:\n{question}\n\nExpected facts (all must be supported):\n"
        + "\n".join(f"- {f}" for f in expected_facts)
        + f"\n\nAgent answer:\n{answer}\n\nJSON verdict:"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judge(raw: str) -> dict:
    """Parse {"pass": bool, "reason": str}; any parse problem is a fail."""
    try:
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        data = json.loads(m.group(0)) if m else None
        if not isinstance(data, dict) or not isinstance(data.get("pass"), bool):
            raise TypeError("no boolean 'pass'")
        return {"pass": data["pass"], "reason": str(data.get("reason", ""))}
    except (ValueError, TypeError) as e:
        return {"pass": False, "reason": f"judge parse error: {e}"}


def judge_case(case: dict, reply: Any, chat: Callable[[list[dict]], Any]) -> dict:
    """Run the judge. `chat(messages)` returns an object with `.content` (or a str)."""
    r = reply_to_dict(reply)
    question = case.get("question") or " / ".join(case.get("turns") or [])
    msgs = build_judge_messages(question, (case.get("expected") or {}).get("facts", []), r.get("text", ""))
    try:
        resp = chat(msgs)
        raw = resp if isinstance(resp, str) else getattr(resp, "content", "")
    except Exception as e:  # noqa: BLE001 - judge failure must not kill the run
        return {"pass": False, "reason": f"judge error: {type(e).__name__}: {e}"}
    return parse_judge(raw)


# ---- cost ----

def cost_inr(prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float, usd_to_inr: float) -> float:
    usd = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
    return usd * usd_to_inr
