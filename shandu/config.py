from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)


_PROVIDER_MANAGED_KEY_PROVIDERS = frozenset(
    {"bedrock", "vertex_ai", "sagemaker", "ollama"}
)
_PLACEHOLDER_API_KEYS = frozenset({"xxx", "changeme", "placeholder", "example"})
_PLACEHOLDER_API_KEY_MARKERS = ("your_", "your-", "changeme", "placeholder")


def _is_placeholder_api_key(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _PLACEHOLDER_API_KEYS:
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_API_KEY_MARKERS)


def _provider_for_bare_model(model: str) -> str:
    if not model:
        return ""
    try:
        from litellm import get_llm_provider

        return str(get_llm_provider(model)[1] or "")
    except Exception:
        return ""


def infer_api_key_env_name(model: str) -> str:
    text = (model or "").strip()
    if "/" in text:
        provider = text.split("/", 1)[0].strip()
    else:
        provider = _provider_for_bare_model(text)
    if provider.lower() in _PROVIDER_MANAGED_KEY_PROVIDERS:
        return ""
    if not provider:
        return "OPENAI_API_KEY"
    normalized = "".join(char if char.isalnum() else "_" for char in provider.upper())
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized:
        return "OPENAI_API_KEY"
    return f"{normalized}_API_KEY"


DEFAULT_MODEL = "deepseek/deepseek-flash"
_LEGACY_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "api": {
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "max_tokens": 16384,
        "api_key_env": "",
        "api_key": "",
    },
    "runtime": {
        "storage_dir": ".blackgeorge",
        "structured_output_retries": 3,
        "max_iterations": 12,
        "max_tool_calls": 24,
        "num_retries": 2,
        "max_context_messages": 30,
    },
    "orchestration": {
        "max_iterations": 2,
        "parallelism": 3,
        "max_results_per_query": 5,
        "max_pages_per_task": 3,
        "detail_level": "high",
        "depth_policy": "adaptive",
    },
    "search": {
        "region": "wt-wt",
        "safesearch": "moderate",
    },
    "scraper": {
        "timeout": 20,
        "max_concurrent": 5,
        "proxy": None,
    },
}


class Config:
    def __init__(self) -> None:
        self._config: dict[str, dict[str, Any]] = {
            section: values.copy() for section, values in DEFAULT_CONFIG.items()
        }
        self._overlay: dict[str, dict[str, Any]] = {}
        self._exported_api_keys: dict[str, str] = {}
        self._path = Path(os.path.expanduser("~/.shandu/config.json"))
        self._load_file()
        self._load_env()
        self._normalize_storage_dir()
        self.apply_provider_api_key()

    def _normalize_storage_dir(self) -> None:
        raw = str(self._config["runtime"].get("storage_dir", ".blackgeorge") or ".blackgeorge")
        self._config["runtime"]["storage_dir"] = str(
            Path(raw).expanduser().resolve()
        )

    def _load_file(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for section, values in payload.items():
                if not isinstance(values, dict):
                    continue
                defaults = DEFAULT_CONFIG.get(section, {})
                for key, value in values.items():
                    if key in defaults and defaults[key] == value:
                        continue
                    if (
                        section == "api"
                        and key == "model"
                        and value == _LEGACY_DEFAULT_MODEL
                    ):
                        continue
                    self._overlay.setdefault(section, {})[key] = value
                    self._config.setdefault(section, {})[key] = value
        except Exception:
            logger.warning(
                "Ignoring malformed config at %s; using defaults", self._path, exc_info=True
            )
            return

    def _load_env(self) -> None:
        model = os.getenv("SHANDU_MODEL") or os.getenv("OPENAI_MODEL_NAME")
        if model:
            self._config["api"]["model"] = model

        if os.getenv("SHANDU_TEMPERATURE"):
            try:
                self._config["api"]["temperature"] = float(
                    os.getenv("SHANDU_TEMPERATURE", "0.2")
                )
            except ValueError:
                pass

        if os.getenv("SHANDU_MAX_TOKENS"):
            try:
                self._config["api"]["max_tokens"] = int(
                    os.getenv("SHANDU_MAX_TOKENS", "16384")
                )
            except ValueError:
                pass

        if os.getenv("SHANDU_API_KEY_ENV"):
            self._config["api"]["api_key_env"] = os.getenv("SHANDU_API_KEY_ENV", "")

        if os.getenv("SHANDU_API_KEY"):
            self._config["api"]["api_key"] = os.getenv("SHANDU_API_KEY", "")

        if os.getenv("SHANDU_STORAGE_DIR"):
            self._config["runtime"]["storage_dir"] = os.getenv("SHANDU_STORAGE_DIR")

        if os.getenv("SHANDU_PROXY"):
            self._config["scraper"]["proxy"] = os.getenv("SHANDU_PROXY")

    def get_api_key_env_name(self, model: str | None = None) -> str:
        configured = str(self.get("api", "api_key_env", "")).strip()
        if configured:
            return configured
        selected_model = model if model is not None else str(self.get("api", "model", ""))
        return infer_api_key_env_name(selected_model)

    def apply_provider_api_key(self, model: str | None = None) -> None:
        env_name = self.get_api_key_env_name(model)
        if not env_name:
            return
        configured_key = str(self.get("api", "api_key", "")).strip()
        if not configured_key:
            return
        if _is_placeholder_api_key(configured_key):
            logger.warning(
                "Ignoring placeholder value for %s; set a real API key", env_name
            )
            return
        existing = os.getenv(env_name)
        if existing and self._exported_api_keys.get(env_name) != existing:
            return
        os.environ[env_name] = configured_key
        self._exported_api_keys[env_name] = configured_key

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        self._overlay.setdefault(section, {})[key] = value
        self._config.setdefault(section, {})[key] = value

    def get_section(self, section: str) -> dict[str, Any]:
        return self._config.get(section, {}).copy()

    def save(self) -> None:
        payload: dict[str, dict[str, Any]] = {}
        for section, values in self._overlay.items():
            defaults = DEFAULT_CONFIG.get(section, {})
            kept = {
                key: value
                for key, value in values.items()
                if key not in defaults or defaults[key] != value
            }
            if kept:
                payload[section] = kept
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.chmod(self._path, 0o600)


config = Config()
