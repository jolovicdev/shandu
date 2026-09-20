from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blackgeorge import Desk
from blackgeorge.memory.sqlite import SQLiteMemoryStore
import litellm

from ..config import DEFAULT_MODEL, config
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeSettings:
    model: str
    temperature: float
    max_tokens: int
    storage_dir: str
    structured_output_retries: int
    max_iterations: int
    max_tool_calls: int
    num_retries: int
    max_context_messages: int


class RuntimeBootstrap:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.cost_tracker = CostTracker()
        config.apply_provider_api_key(settings.model)
        litellm.suppress_debug_info = True
        storage = Path(settings.storage_dir)
        storage.mkdir(parents=True, exist_ok=True)
        memory_path = storage / "memory.db"
        self.memory_store = SQLiteMemoryStore(str(memory_path))
        self.desk = Desk(
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            storage_dir=str(storage),
            structured_output_retries=settings.structured_output_retries,
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
            num_retries=settings.num_retries,
            max_context_messages=settings.max_context_messages,
            respect_context_window=True,
            memory_store=self.memory_store,
        )
        try:
            self.desk.event_bus.subscribe("llm.completed", self.cost_tracker.handle_event)
        except Exception:
            logger.warning("Failed to subscribe cost tracker to llm.completed", exc_info=True)

    def close(self) -> None:
        try:
            self.desk.close()
        finally:
            self.memory_store.close()

    @classmethod
    def from_config(cls) -> "RuntimeBootstrap":
        def lookup(section: str, key: str, default: Any) -> Any:
            return config.get(section, key, default)

        return cls(
            RuntimeSettings(
                model=str(lookup("api", "model", DEFAULT_MODEL)),
                temperature=float(lookup("api", "temperature", 0.2)),
                max_tokens=int(lookup("api", "max_tokens", 16384)),
                storage_dir=str(lookup("runtime", "storage_dir", ".blackgeorge")),
                structured_output_retries=int(
                    lookup("runtime", "structured_output_retries", 3)
                ),
                max_iterations=int(lookup("runtime", "max_iterations", 12)),
                max_tool_calls=int(lookup("runtime", "max_tool_calls", 24)),
                num_retries=int(lookup("runtime", "num_retries", 2)),
                max_context_messages=int(
                    lookup("runtime", "max_context_messages", 30)
                ),
            )
        )

    def inspect_run(self, run_id: str) -> dict[str, object]:
        record = self.desk.run_store.get_run(run_id)
        if record is not None:
            events = self.desk.run_store.get_events(run_id)
            usage_tracker = CostTracker()
            for event in events:
                usage_tracker.handle_event(event)
            usage_snapshot = usage_tracker.snapshot()
            return {
                "exists": True,
                "run_id": record.run_id,
                "status": record.status,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "input": record.input,
                "output": record.output,
                "output_json": record.output_json,
                "usage": {
                    "prompt_tokens": usage_snapshot.prompt_tokens,
                    "completion_tokens": usage_snapshot.completion_tokens,
                    "total_tokens": usage_snapshot.total_tokens,
                    "cost_usd": usage_snapshot.total_cost_usd,
                    "llm_calls": usage_snapshot.llm_calls,
                    "cost_events": usage_snapshot.cost_events,
                },
                "events": [
                    {
                        "type": event.type,
                        "timestamp": event.timestamp.isoformat(),
                        "source": event.source,
                        "payload": event.payload,
                    }
                    for event in events
                ],
            }

        scope = f"run:{run_id}"
        status = self.memory_store.read("status", scope)
        if status is None:
            return {"exists": False, "run_id": run_id}

        created_at = self.memory_store.read("created_at", scope)
        updated_at = self.memory_store.read("updated_at", scope)
        request_payload = self.memory_store.read("request", scope)
        result_payload = self.memory_store.read("result", scope)
        events_payload = self.memory_store.read("events", scope) or []
        return {
            "exists": True,
            "run_id": run_id,
            "status": status,
            "created_at": created_at or "",
            "updated_at": updated_at or created_at or "",
            "input": request_payload,
            "output": None,
            "output_json": result_payload,
            "events": events_payload if isinstance(events_payload, list) else [],
        }


_bootstrap: RuntimeBootstrap | None = None
_bootstrap_lock = threading.Lock()


def get_bootstrap() -> RuntimeBootstrap:
    global _bootstrap
    if _bootstrap is None:
        with _bootstrap_lock:
            if _bootstrap is None:
                _bootstrap = RuntimeBootstrap.from_config()
    return _bootstrap


def reset_bootstrap() -> None:
    global _bootstrap
    current = _bootstrap
    _bootstrap = None
    if current is not None:
        try:
            current.close()
        except Exception:
            logger.warning("Failed to close previous runtime bootstrap", exc_info=True)
