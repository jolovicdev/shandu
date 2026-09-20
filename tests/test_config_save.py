from __future__ import annotations

import json

from shandu.config import DEFAULT_CONFIG, Config


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
