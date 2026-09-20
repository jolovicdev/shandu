from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from shandu.config import DEFAULT_CONFIG, Config, infer_api_key_env_name

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "model,expected",
    [
        ("deepseek/deepseek-v4-flash", "DEEPSEEK_API_KEY"),
        ("openrouter/minimax/minimax-m2.5", "OPENROUTER_API_KEY"),
        ("anthropic/claude-sonnet-4", "ANTHROPIC_API_KEY"),
        ("gpt-4o", "OPENAI_API_KEY"),
        ("not-a-real-model-xyz", "OPENAI_API_KEY"),
        ("bedrock/anthropic.claude-3-sonnet", ""),
        ("vertex_ai/gemini-1.5-pro", ""),
        ("ollama/llama3", ""),
    ],
)
def test_infer_api_key_env_name(model, expected) -> None:
    assert infer_api_key_env_name(model) == expected


def test_config_save_restricts_file_permissions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    saved = Config()
    saved.set("api", "api_key", "secret")
    saved.save()
    assert stat.S_IMODE(os.stat(saved._path).st_mode) == 0o600
    os.chmod(saved._path, 0o644)
    saved.save()
    assert stat.S_IMODE(os.stat(saved._path).st_mode) == 0o600


def test_apply_provider_api_key_rotates_own_export_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = Config()
    cfg.set("api", "model", "deepseek/deepseek-v4-flash")
    cfg.set("api", "api_key", "key-one")
    cfg.apply_provider_api_key()
    assert os.environ["DEEPSEEK_API_KEY"] == "key-one"
    cfg.set("api", "api_key", "key-two")
    cfg.apply_provider_api_key()
    assert os.environ["DEEPSEEK_API_KEY"] == "key-two"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "shell-key")
    fresh = Config()
    fresh.set("api", "api_key", "key-three")
    fresh.apply_provider_api_key()
    assert os.environ["DEEPSEEK_API_KEY"] == "shell-key"


def test_apply_provider_api_key_ignores_placeholder_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = Config()
    cfg.set("api", "api_key", "your_provider_api_key_here")
    cfg.apply_provider_api_key()
    assert os.getenv("DEEPSEEK_API_KEY") is None


def test_env_example_has_no_active_placeholders() -> None:
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    active = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    assert not any("your_provider_api_key_here" in line for line in active)
    tokens = [line for line in active if line.startswith("SHANDU_MAX_TOKENS=")]
    assert tokens == [f"SHANDU_MAX_TOKENS={DEFAULT_CONFIG['api']['max_tokens']}"]