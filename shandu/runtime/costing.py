from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "llm_calls",
    "cost_events",
)


def collect_llm_usage(runtime: Any, report: Any) -> dict[str, Any] | None:
    run_id = getattr(report, "run_id", None)
    inspect_run = getattr(runtime, "inspect_run", None)
    if not run_id or inspect_run is None:
        return None
    try:
        inspection = inspect_run(run_id)
    except Exception:
        logger.warning("Failed to inspect blackgeorge run %s", run_id, exc_info=True)
        return None
    if not isinstance(inspection, dict):
        return None
    usage = inspection.get("usage")
    if not isinstance(usage, dict):
        return None
    collected = {key: usage[key] for key in USAGE_KEYS if key in usage}
    return collected or None
