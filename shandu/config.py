from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def infer_api_key_env_name(model: str) -> str:
    provider = (model or "").strip().split("/", 1)[0].strip()
    if not provider:
        return "OPENAI_API_KEY"
    normalized = "".join(char if char.isalnum() else "_" for char in provider.upper())
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized:
        return "OPENAI_API_KEY"
    return f"{normalized}_API_KEY"


DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "api": {
        "model": "deepseek/deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 8192,
        "api_key_env": "",
        "api_key": "",
    },
    "runtime": {
        "storage_dir": ".blackgeorge",
        "structured_output_retries": 3,
        "max_iterations": 12,
        "max_tool_calls": 24,
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
        self._path = Path(os.path.expanduser("~/.shandu/config.json"))
        self._load_file()
        self._load_env()
        self.apply_provider_api_key()

    def _load_file(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._merge(self._config, payload)
        except Exception:
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
                    os.getenv("SHANDU_MAX_TOKENS", "8192")
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

    def apply_provider_api_key(self) -> None:
        env_name = self.get_api_key_env_name()
        if not env_name:
            return
        existing = os.getenv(env_name)
        if existing:
            return
        configured_key = str(self.get("api", "api_key", "")).strip()
        if configured_key:
            os.environ[env_name] = configured_key

    def _merge(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge(target[key], value)
            else:
                target[key] = value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        self._config.setdefault(section, {})[key] = value

    def get_section(self, section: str) -> dict[str, Any]:
        return self._config.get(section, {}).copy()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(self._config, handle, indent=2)


config = Config()
