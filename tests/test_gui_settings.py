from __future__ import annotations

from shandu.ui.gradio import settings


def _isolate_config(monkeypatch) -> dict[tuple[str, str], object]:
    store: dict[tuple[str, str], object] = {}
    monkeypatch.setattr(
        settings.config, "get", lambda section, key, default=None: store.get((section, key), default)
    )
    monkeypatch.setattr(
        settings.config, "set", lambda section, key, value: store.__setitem__((section, key), value)
    )
    monkeypatch.setattr(settings.config, "save", lambda: None)
    monkeypatch.setattr(settings.config, "apply_provider_api_key", lambda: None)
    return store


_ARGS = {
    "model": "deepseek/deepseek-v4-flash",
    "api_key_env": "DEEPSEEK_API_KEY",
    "api_key_value": "",
    "temperature": 0.2,
    "max_tokens": 16384,
    "max_iterations": 2,
    "parallelism": 3,
    "detail_level": "high",
    "depth_policy": "adaptive",
    "max_results_per_query": 5,
    "max_pages_per_task": 3,
}


def test_save_configuration_resets_runtime_only_when_runtime_values_change(
    monkeypatch,
) -> None:
    _isolate_config(monkeypatch)
    resets: list[int] = []
    monkeypatch.setattr(settings, "reset_bootstrap", lambda: resets.append(1))

    settings.save_configuration(**_ARGS)
    assert len(resets) == 1

    settings.save_configuration(**_ARGS)
    assert len(resets) == 1

    settings.save_configuration(**{**_ARGS, "temperature": 0.7})
    assert len(resets) == 2


def test_save_configuration_ignores_orchestration_only_changes(monkeypatch) -> None:
    _isolate_config(monkeypatch)
    resets: list[int] = []
    monkeypatch.setattr(settings, "reset_bootstrap", lambda: resets.append(1))

    settings.save_configuration(**_ARGS)
    settings.save_configuration(**{**_ARGS, "parallelism": 6, "max_iterations": 4})

    assert len(resets) == 1


def test_save_configuration_resets_on_new_api_key(monkeypatch) -> None:
    _isolate_config(monkeypatch)
    resets: list[int] = []
    monkeypatch.setattr(settings, "reset_bootstrap", lambda: resets.append(1))

    settings.save_configuration(**_ARGS)
    settings.save_configuration(**{**_ARGS, "api_key_value": "sk-new"})

    assert len(resets) == 2
