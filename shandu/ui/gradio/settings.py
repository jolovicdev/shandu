from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ...config import config, infer_api_key_env_name
from ...interfaces import DepthPolicy, DetailLevel
from ...runtime import reset_bootstrap
from .constants import DEPTH_POLICIES, DETAIL_LEVELS


# Only these config values feed RuntimeBootstrap. Orchestration values are
# read per-request, so saving them must not tear down the shared runtime:
# reset_bootstrap() closes SQLite stores that any in-flight run still holds,
# and the GUI saves configuration at the start of every run.
_RUNTIME_KEYS: tuple[tuple[str, str], ...] = (
    ("api", "model"),
    ("api", "api_key_env"),
    ("api", "api_key"),
    ("api", "temperature"),
    ("api", "max_tokens"),
)


def _runtime_snapshot() -> tuple[str, ...]:
    return tuple(str(config.get(section, key, "")) for section, key in _RUNTIME_KEYS)


@dataclass(frozen=True, slots=True)
class GuiDefaults:
    model: str
    api_key_env: str
    temperature: float
    max_tokens: int
    max_iterations: int
    parallelism: int
    detail_level: str
    depth_policy: str
    max_results_per_query: int
    max_pages_per_task: int


def load_defaults() -> GuiDefaults:
    model = str(config.get("api", "model", "deepseek/deepseek-v4-flash"))
    return GuiDefaults(
        model=model,
        api_key_env=config.get_api_key_env_name(model),
        temperature=float(config.get("api", "temperature", 0.2)),
        max_tokens=int(config.get("api", "max_tokens", 16384)),
        max_iterations=int(config.get("orchestration", "max_iterations", 2)),
        parallelism=int(config.get("orchestration", "parallelism", 3)),
        detail_level=str(config.get("orchestration", "detail_level", "high")),
        depth_policy=str(config.get("orchestration", "depth_policy", "adaptive")),
        max_results_per_query=int(
            config.get("orchestration", "max_results_per_query", 5)
        ),
        max_pages_per_task=int(config.get("orchestration", "max_pages_per_task", 3)),
    )


def resolved_detail_level(value: object) -> DetailLevel:
    text = str(value or "high")
    if text in DETAIL_LEVELS:
        return cast(DetailLevel, text)
    return "high"


def resolved_depth_policy(value: object) -> DepthPolicy:
    text = str(value or "adaptive")
    if text in DEPTH_POLICIES:
        return cast(DepthPolicy, text)
    return "adaptive"


def save_configuration(
    model: object,
    api_key_env: object,
    api_key_value: object,
    temperature: object,
    max_tokens: object,
    max_iterations: object,
    parallelism: object,
    detail_level: object,
    depth_policy: object,
    max_results_per_query: object,
    max_pages_per_task: object,
) -> str:
    model_text = str(model or "").strip() or "deepseek/deepseek-v4-flash"
    env_text = str(api_key_env or "").strip()
    key_text = str(api_key_value or "").strip()

    resolved_env = env_text or infer_api_key_env_name(model_text)
    runtime_before = _runtime_snapshot()
    config.set("api", "model", model_text)
    config.set("api", "api_key_env", resolved_env)
    if key_text:
        config.set("api", "api_key", key_text)
    config.set(
        "api", "temperature", float(temperature) if temperature is not None else 0.2
    )
    config.set(
        "api", "max_tokens", int(max_tokens) if max_tokens is not None else 16384
    )
    config.set(
        "orchestration",
        "max_iterations",
        int(max_iterations) if max_iterations is not None else 2,
    )
    config.set(
        "orchestration",
        "parallelism",
        int(parallelism) if parallelism is not None else 3,
    )
    config.set("orchestration", "detail_level", resolved_detail_level(detail_level))
    config.set("orchestration", "depth_policy", resolved_depth_policy(depth_policy))
    config.set(
        "orchestration",
        "max_results_per_query",
        int(max_results_per_query) if max_results_per_query is not None else 5,
    )
    config.set(
        "orchestration",
        "max_pages_per_task",
        int(max_pages_per_task) if max_pages_per_task is not None else 3,
    )
    config.save()
    if _runtime_snapshot() != runtime_before:
        reset_bootstrap()
    config.apply_provider_api_key()
    return f"Saved configuration for `{model_text}` using env key `{resolved_env}`."


def persist_report_markdown(run_id: str, markdown: str) -> str | None:
    text = markdown.strip()
    if not text:
        return None
    try:
        storage = Path(str(config.get("runtime", "storage_dir", ".blackgeorge")))
        export_dir = storage / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        safe_run = (
            "".join(char if char.isalnum() else "_" for char in run_id).strip("_")
            or "report"
        )
        file_path = export_dir / f"{safe_run}.md"
        file_path.write_text(text, encoding="utf-8")
        return str(file_path)
    except Exception:
        return None


_resolved_detail_level = resolved_detail_level
_resolved_depth_policy = resolved_depth_policy
_save_configuration = save_configuration
_persist_report_markdown = persist_report_markdown
