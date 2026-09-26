"""One place that talks to the Claude API, so one place can be turned off.

`complete_json` returns a parsed dict, or None on any failure at all -- no key,
no SDK, rate limit, timeout, malformed JSON. Every caller treats None as
"degrade to the deterministic path". That is what makes the demo survive a
dead conference network, and it is why nothing else in the codebase imports
`anthropic` directly.
"""

from __future__ import annotations

import json
import os
import re

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TIMEOUT = float(os.environ.get("AGENT_TIMEOUT", "30"))

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def available() -> bool:
    """True when an API key and the SDK are both present."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _client():
    import anthropic

    return anthropic.Anthropic(timeout=DEFAULT_TIMEOUT)


def complete_json(system: str, user: str, model: str | None = None,
                  max_tokens: int = DEFAULT_MAX_TOKENS) -> dict | None:
    """Ask for JSON and return it parsed, or None if anything goes wrong."""
    if os.environ.get("AGENT_LLM", "").strip().lower() == "off" or not available():
        return None
    try:
        response = _client().messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
    except Exception:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)  # tolerate prose or fences around the JSON
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
