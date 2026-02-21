from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from blackgeorge import Desk
from blackgeorge.memory.sqlite import SQLiteMemoryStore
import litellm

from ..config import config
from .cost_tracker import CostTracker


@dataclass(slots=True)
class RuntimeSettings:
    model: str
    temperature: float
    max_tokens: int
    storage_dir: str
    structured_output_retries: int
    max_iterations: int
    max_tool_calls: int


class RuntimeBootstrap:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.cost_tracker = CostTracker()
        config.apply_provider_api_key()
        api_key_env = config.get_api_key_env_name(settings.model)
        api_key_value = str(config.get("api", "api_key", "")).strip()
        if api_key_env and api_key_value and not os.getenv(api_key_env):
            os.environ[api_key_env] = api_key_value
        litellm.set_verbose = False
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
            respect_context_window=True,
            memory_store=self.memory_store,
        )
        try:
            self.desk.event_bus.subscribe("llm.completed", self.cost_tracker.handle_event)
        except Exception:
            pass

    @classmethod
    def from_config(cls) -> "RuntimeBootstrap":
        return cls(
            RuntimeSettings(
                model=str(config.get("api", "model", "deepseek/deepseek-chat")),
                temperature=float(config.get("api", "temperature", 0.2)),
                max_tokens=int(config.get("api", "max_tokens", 8192)),
                storage_dir=str(config.get("runtime", "storage_dir", ".blackgeorge")),
                structured_output_retries=int(
                    config.get("runtime", "structured_output_retries", 3)
                ),
                max_iterations=int(config.get("runtime", "max_iterations", 12)),
                max_tool_calls=int(config.get("runtime", "max_tool_calls", 24)),
            )
        )

    def inspect_run(self, run_id: str) -> dict[str, object]:
        record = self.desk.run_store.get_run(run_id)
        if record is not None:
            events = self.desk.run_store.get_events(run_id)
            return {
                "exists": True,
                "run_id": record.run_id,
                "status": record.status,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "input": record.input,
                "output": record.output,
                "output_json": record.output_json,
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


def get_bootstrap() -> RuntimeBootstrap:
    global _bootstrap
    if _bootstrap is None:
        _bootstrap = RuntimeBootstrap.from_config()
    return _bootstrap


def reset_bootstrap() -> None:
    global _bootstrap
    _bootstrap = None
