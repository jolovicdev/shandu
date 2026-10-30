from __future__ import annotations

import json
from pathlib import Path

from shandu.config import DEFAULT_CONFIG, DEFAULT_STORAGE_DIR, Config


def test_config_save_writes_only_user_set_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHANDU_API_KEY", "env-key")
    monkeypatch.setenv("SHANDU_MODEL", "env/model")
    cfg = Config()
    cfg.set("orchestration", "parallelism", 5)
    cfg.save()

    payload = json.loads((tmp_path / ".shandu" / "config.json").read_text())

    assert payload == {"orchestration": {"parallelism": 5}}
    assert cfg.get("api", "api_key") == "env-key"
    assert cfg.get("api", "model") == "env/model"


def test_config_load_drops_default_equal_entries(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SHANDU_API_KEY", raising=False)
    monkeypatch.delenv("SHANDU_MODEL", raising=False)
    conf_dir = tmp_path / ".shandu"
    conf_dir.mkdir()
    (conf_dir / "config.json").write_text(
        json.dumps(
            {
                "api": {
                    "model": DEFAULT_CONFIG["api"]["model"],
                    "temperature": 0.9,
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = Config()

    assert cfg.get("api", "temperature") == 0.9
    assert cfg.get("api", "model") == DEFAULT_CONFIG["api"]["model"]
    cfg.save()
    payload = json.loads((conf_dir / "config.json").read_text())
    assert payload == {"api": {"temperature": 0.9}}


def test_saved_legacy_default_model_is_treated_as_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SHANDU_MODEL", raising=False)
    conf_dir = tmp_path / ".shandu"
    conf_dir.mkdir()
    (conf_dir / "config.json").write_text(
        json.dumps({"api": {"model": "deepseek/deepseek-v4-flash"}}),
        encoding="utf-8",
    )

    cfg = Config()

    assert cfg.get("api", "model") == "deepseek/deepseek-flash"
    cfg.save()
    payload = json.loads((conf_dir / "config.json").read_text())
    assert payload == {}


def test_config_saved_by_every_setting_loads_current_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SHANDU_MODEL", raising=False)
    monkeypatch.delenv("SHANDU_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SHANDU_STORAGE_DIR", raising=False)
    conf_dir = tmp_path / ".shandu"
    conf_dir.mkdir()
    (conf_dir / "config.json").write_text(
        json.dumps(
            {
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
                    "max_results_per_query": 10,
                    "max_pages_per_task": 3,
                    "detail_level": "high",
                    "depth_policy": "adaptive",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = Config()

    assert cfg.get("api", "model") == DEFAULT_CONFIG["api"]["model"]
    assert cfg.get("api", "max_tokens") == DEFAULT_CONFIG["api"]["max_tokens"]
    assert cfg.get("runtime", "storage_dir") == str(
        Path(DEFAULT_STORAGE_DIR).expanduser().resolve()
    )
    assert cfg.get("orchestration", "max_pages_per_task") == (
        DEFAULT_CONFIG["orchestration"]["max_pages_per_task"]
    )
    assert cfg.get("orchestration", "detail_level") == (
        DEFAULT_CONFIG["orchestration"]["detail_level"]
    )
    assert cfg.get("orchestration", "max_results_per_query") == 10


def test_config_of_user_set_keys_keeps_a_value_equal_to_an_old_default(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    conf_dir = tmp_path / ".shandu"
    conf_dir.mkdir()
    (conf_dir / "config.json").write_text(
        json.dumps({"orchestration": {"max_pages_per_task": 3}}),
        encoding="utf-8",
    )

    cfg = Config()

    assert cfg.get("orchestration", "max_pages_per_task") == 3
    cfg.save()
    payload = json.loads((conf_dir / "config.json").read_text())
    assert payload == {"orchestration": {"max_pages_per_task": 3}}
