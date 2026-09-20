from __future__ import annotations

import shandu.runtime.bootstrap as boot
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


class _FakeBootstrap:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_reset_defers_close_until_runs_release(monkeypatch) -> None:
    monkeypatch.setattr(
        RuntimeBootstrap,
        "from_config",
        classmethod(lambda cls: _FakeBootstrap()),
    )
    monkeypatch.setattr(boot, "_bootstrap", None)
    monkeypatch.setattr(boot, "_active_runs", 0)
    monkeypatch.setattr(boot, "_retired", [])

    first = boot.get_bootstrap()
    boot.acquire_bootstrap_run()
    boot.reset_bootstrap()

    assert first.closed is False
    second = boot.get_bootstrap()
    assert second is not first

    boot.release_bootstrap_run()

    assert first.closed is True
    assert second.closed is False
