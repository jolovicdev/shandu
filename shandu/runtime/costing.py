from __future__ import annotations

from typing import Any

_TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    cost = metrics.get("cost_usd")
    if _is_number(cost):
        collected["cost_usd"] = cost
    return collected or None
