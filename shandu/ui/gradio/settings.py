from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

from ...config import DEFAULT_MODEL, api_key_env_to_keep, config, infer_api_key_env_name
from ...contracts import ReportLength
from ...interfaces import DepthPolicy
from ...runtime import reset_bootstrap
from .constants import DEPTH_POLICIES, DETAIL_LEVELS


# Only these config values feed RuntimeBootstrap. Orchestration values are
# read per-request, so saving them must not tear down the shared runtime:
# reset_bootstrap() closes SQLite stores that any in-flight run still holds.
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
    model = str(config.get("api", "model", DEFAULT_MODEL))
    return GuiDefaults(
        model=model,
        api_key_env=str(config.get("api", "api_key_env", "")),
        temperature=float(config.get("api", "temperature", 0.2)),
        max_tokens=int(config.get("api", "max_tokens", 32768)),
        max_iterations=int(config.get("orchestration", "max_iterations", 2)),
        parallelism=int(config.get("orchestration", "parallelism", 3)),
        detail_level=str(config.get("orchestration", "detail_level", "auto")),
        depth_policy=str(config.get("orchestration", "depth_policy", "adaptive")),
        max_results_per_query=int(
            config.get("orchestration", "max_results_per_query", 8)
        ),
        max_pages_per_task=int(config.get("orchestration", "max_pages_per_task", 6)),
    )


def resolved_detail_level(value: object) -> ReportLength:
    text = str(value or "auto")
    if text in DETAIL_LEVELS:
        return cast(ReportLength, text)
    return "auto"


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
    model_text = str(model or "").strip() or DEFAULT_MODEL
    env_text = str(api_key_env or "").strip()
    key_text = str(api_key_value or "").strip()

    env_text = api_key_env_to_keep(env_text, str(config.get("api", "model", "")), model_text)
    resolved_env = env_text or infer_api_key_env_name(model_text)
    runtime_before = _runtime_snapshot()
    config.set("api", "model", model_text)
    config.set("api", "api_key_env", env_text)
    if key_text:
        config.set("api", "api_key", key_text)
    config.set(
        "api", "temperature", float(temperature) if temperature is not None else 0.2
    )
    config.set(
        "api", "max_tokens", int(max_tokens) if max_tokens is not None else 32768
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
        int(max_results_per_query) if max_results_per_query is not None else 8,
    )
    config.set(
        "orchestration",
        "max_pages_per_task",
        int(max_pages_per_task) if max_pages_per_task is not None else 6,
    )
    config.save()
    if _runtime_snapshot() != runtime_before:
        reset_bootstrap()
    config.apply_provider_api_key()
    if not resolved_env:
        return f"Saved. Model {model_text}; its provider manages the key."
    if os.getenv(resolved_env, "").strip():
        return f"Saved. Model {model_text}; the API key is in {resolved_env}."
    return (
        f"Saved. Model {model_text}; no API key yet. Paste one above or set {resolved_env}."
    )


_resolved_detail_level = resolved_detail_level
_resolved_depth_policy = resolved_depth_policy
_save_configuration = save_configuration
