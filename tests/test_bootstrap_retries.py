from __future__ import annotations

from shandu.runtime.bootstrap import RuntimeBootstrap, RuntimeSettings


def test_runtime_settings_includes_retries_and_context():
    settings = RuntimeSettings(
        model="deepseek/deepseek-v4-flash",
        temperature=0.2,
        max_tokens=16384,
        storage_dir=".blackgeorge",
        structured_output_retries=3,
        max_iterations=12,
        max_tool_calls=24,
        num_retries=2,
        max_context_messages=30,
    )
    assert settings.num_retries == 2
    assert settings.max_context_messages == 30


def test_bootstrap_from_config_defaults():
    bootstrap = RuntimeBootstrap.from_config()
    assert bootstrap.settings.num_retries == 2
    assert bootstrap.settings.max_context_messages == 30
