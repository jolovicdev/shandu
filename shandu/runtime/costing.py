from __future__ import annotations

import math
from typing import Any, TypeGuard

_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _is_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def collect_llm_usage(report: Any) -> dict[str, Any] | None:
    metrics = getattr(report, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    collected: dict[str, Any] = {}
    usage = metrics.get("usage")
    if isinstance(usage, dict):
        for key in _TOKEN_KEYS:
            value = usage.get(key)
            if _is_number(value):
                collected[key] = value
        if (
            "total_tokens" not in collected
            and "prompt_tokens" in collected
            and "completion_tokens" in collected
        ):
            collected["total_tokens"] = (
                collected["prompt_tokens"] + collected["completion_tokens"]
            )
    cost = metrics.get("cost_usd")
    # A price table that lacks the model reports a cost of zero. Counting
    # that as a price would make a cost budget look enforced when it is not.
    if _is_number(cost) and cost > 0:
        collected["cost_usd"] = cost
    return collected or None


def add_usage(
    total: dict[str, Any] | None, usage: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not usage:
        return total
    merged = dict(total or {})
    for key, value in usage.items():
        if key == "llm_calls" or not _is_number(value):
            continue
        merged[key] = merged.get(key, 0) + value
    raw_calls = usage.get("llm_calls", 1)
    added_calls = int(raw_calls) if _is_number(raw_calls) and raw_calls > 0 else 1
    merged["llm_calls"] = (total or {}).get("llm_calls", 0) + added_calls
    return merged
