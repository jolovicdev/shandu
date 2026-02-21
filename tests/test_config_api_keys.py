from __future__ import annotations

from shandu.config import infer_api_key_env_name


def test_infer_api_key_env_name_for_common_models() -> None:
    assert infer_api_key_env_name("deepseek/deepseek-chat") == "DEEPSEEK_API_KEY"
    assert infer_api_key_env_name("openrouter/minimax/minimax-m2.5") == "OPENROUTER_API_KEY"
    assert infer_api_key_env_name("anthropic/claude-sonnet-4") == "ANTHROPIC_API_KEY"