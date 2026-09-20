from __future__ import annotations

import shandu.runtime.bootstrap as boot


class _FakeBootstrap:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_reset_defers_close_until_runs_release(monkeypatch) -> None:
    monkeypatch.setattr(
        boot.RuntimeBootstrap,
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
