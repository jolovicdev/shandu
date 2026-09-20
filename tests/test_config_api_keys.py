from __future__ import annotations

import os
import stat

from shandu.config import Config, infer_api_key_env_name


def test_infer_api_key_env_name_for_common_models() -> None:
    assert infer_api_key_env_name("deepseek/deepseek-v4-flash") == "DEEPSEEK_API_KEY"
    assert infer_api_key_env_name("openrouter/minimax/minimax-m2.5") == "OPENROUTER_API_KEY"
    assert infer_api_key_env_name("anthropic/claude-sonnet-4") == "ANTHROPIC_API_KEY"


def test_config_save_restricts_file_permissions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    saved = Config()
    saved.set("api", "api_key", "secret")
    saved.save()
    assert stat.S_IMODE(os.stat(saved._path).st_mode) == 0o600
    os.chmod(saved._path, 0o644)
    saved.save()
    assert stat.S_IMODE(os.stat(saved._path).st_mode) == 0o600