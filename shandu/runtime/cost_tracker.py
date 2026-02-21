from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    llm_calls: int = 0
    cost_events: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CostTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = CostSnapshot()

    def handle_event(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "") or "").strip()
        if event_type and event_type != "llm.completed":
            return

        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            return

        cost = self._to_float(payload.get("cost"))
        prompt_tokens = self._to_int(payload.get("prompt_tokens"))
        completion_tokens = self._to_int(payload.get("completion_tokens"))
        total_tokens = self._to_int(payload.get("total_tokens"))
        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        with self._lock:
            current = self._snapshot
            self._snapshot = CostSnapshot(
                llm_calls=current.llm_calls + 1,
                cost_events=current.cost_events + (1 if cost is not None else 0),
                total_cost_usd=current.total_cost_usd + (cost or 0.0),
                total_tokens=current.total_tokens + (total_tokens or 0),
                prompt_tokens=current.prompt_tokens + (prompt_tokens or 0),
                completion_tokens=current.completion_tokens + (completion_tokens or 0),
            )

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return self._snapshot

    def delta_since(self, baseline: CostSnapshot) -> CostSnapshot:
        current = self.snapshot()
        return CostSnapshot(
            llm_calls=max(0, current.llm_calls - baseline.llm_calls),
            cost_events=max(0, current.cost_events - baseline.cost_events),
            total_cost_usd=max(0.0, current.total_cost_usd - baseline.total_cost_usd),
            total_tokens=max(0, current.total_tokens - baseline.total_tokens),
            prompt_tokens=max(0, current.prompt_tokens - baseline.prompt_tokens),
            completion_tokens=max(0, current.completion_tokens - baseline.completion_tokens),
        )

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed
