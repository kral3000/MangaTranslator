"""Token usage accounting shared by the LLM endpoints.

Providers report usage under different keys, so each endpoint normalizes its
raw payload into the same four counters:

    prompt   input tokens (prompt + images)
    thoughts internal reasoning tokens (0 when the model does not reason)
    output   tokens in the visible answer
    total    provider-reported total, or the sum of the above as a fallback

Usage is carried to the UI through an optional mutable dict, mirroring the
``ocr_texts_out`` side-channel already used by the pipeline, so no existing
return signature has to change.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

USAGE_KEYS = ("prompt", "thoughts", "output", "total", "calls")


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def make_usage(
    prompt: Any = 0,
    thoughts: Any = 0,
    output: Any = 0,
    total: Any = 0,
) -> Dict[str, int]:
    """Build a normalized usage record for a single API call."""
    usage = {
        "prompt": _as_int(prompt),
        "thoughts": _as_int(thoughts),
        "output": _as_int(output),
        "calls": 1,
    }
    reported_total = _as_int(total)
    usage["total"] = reported_total or (
        usage["prompt"] + usage["thoughts"] + usage["output"]
    )
    return usage


def parse_google_usage(result: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """Read Gemini/Vertex ``usageMetadata`` from a response body."""
    meta = result.get("usageMetadata")
    if not isinstance(meta, Mapping):
        return None
    return make_usage(
        prompt=meta.get("promptTokenCount"),
        thoughts=meta.get("thoughtsTokenCount"),
        output=meta.get("candidatesTokenCount"),
        total=meta.get("totalTokenCount"),
    )


def parse_openai_style_usage(result: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """Read an OpenAI-compatible ``usage`` block (DeepSeek, OpenAI, xAI, ...).

    Reasoning tokens are reported inside ``completion_tokens_details`` and are
    already counted in ``completion_tokens``, so they are split back out to keep
    ``output`` meaning "visible answer only".
    """
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        return None

    completion = _as_int(usage.get("completion_tokens"))
    details = usage.get("completion_tokens_details")
    reasoning = 0
    if isinstance(details, Mapping):
        reasoning = _as_int(details.get("reasoning_tokens"))
    if not reasoning:
        reasoning = _as_int(usage.get("reasoning_tokens"))

    visible = max(0, completion - reasoning)
    return make_usage(
        prompt=usage.get("prompt_tokens"),
        thoughts=reasoning,
        output=visible,
        total=usage.get("total_tokens"),
    )


def accumulate_usage(target: Optional[Dict[str, int]], usage: Optional[Mapping[str, Any]]) -> None:
    """Add one call's usage into a running total. Safe with None on either side."""
    if target is None or not usage:
        return
    for key in USAGE_KEYS:
        target[key] = _as_int(target.get(key)) + _as_int(usage.get(key))


def format_usage(usage: Optional[Mapping[str, Any]]) -> str:
    """Render usage for the UI receipt / logs. Empty string when unavailable."""
    if not usage or not _as_int(usage.get("total")):
        return ""
    parts = [f"prompt {_as_int(usage.get('prompt')):,}"]
    thoughts = _as_int(usage.get("thoughts"))
    if thoughts:
        parts.append(f"thoughts {thoughts:,}")
    parts.append(f"output {_as_int(usage.get('output')):,}")
    parts.append(f"total {_as_int(usage.get('total')):,}")
    text = " / ".join(parts)

    calls = _as_int(usage.get("calls"))
    if calls > 1:
        text += f"  ({calls} calls)"
    return text
