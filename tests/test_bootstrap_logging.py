from __future__ import annotations

import logging

import litellm
from blackgeorge.event_bus import EventBus

from shandu.runtime.bootstrap import RuntimeBootstrap, RuntimeSettings


def _settings(storage_dir: str) -> RuntimeSettings:
    return RuntimeSettings(
        model="fake/model",
        temperature=0.2,
        max_tokens=100,
        storage_dir=storage_dir,
        structured_output_retries=1,
        max_iterations=2,
        max_tool_calls=4,
        num_retries=0,
        max_context_messages=10,
    )


def test_bootstrap_logs_metering_subscribe_failure(tmp_path, caplog, monkeypatch) -> None:
    def failing_subscribe(self, event_type, handler):
        del self, event_type, handler
        raise RuntimeError("bus down")

    monkeypatch.setattr(EventBus, "subscribe", failing_subscribe)
    with caplog.at_level(logging.WARNING, logger="shandu.runtime.bootstrap"):
        bootstrap = RuntimeBootstrap(_settings(str(tmp_path)))
    try:
        assert any(
            "llm.completed" in record.message for record in caplog.records
        )
    finally:
        bootstrap.close()


def test_bootstrap_leaves_verbose_flag_alone(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(litellm, "set_verbose", True)
    bootstrap = RuntimeBootstrap(_settings(str(tmp_path)))
    try:
        assert litellm.set_verbose is True
        assert litellm.suppress_debug_info is True
    finally:
        bootstrap.close()
