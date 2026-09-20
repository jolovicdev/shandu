from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from shandu.runtime.async_runner import get_async_runner


def test_async_runner_reuses_single_event_loop() -> None:
    runner = get_async_runner()

    async def current_loop_id() -> int:
        return id(asyncio.get_running_loop())

    first = runner.run(current_loop_id())
    second = runner.run(current_loop_id())

    assert first == second


def test_async_runner_submit_returns_cancellable_future() -> None:
    import concurrent.futures

    runner = get_async_runner()
    started = threading.Event()

    async def work() -> str:
        started.set()
        await asyncio.sleep(3600)
        return "done"

    future = runner.submit(work())
    assert started.wait(timeout=5)
    assert future.cancel()
    with pytest.raises(concurrent.futures.CancelledError):
        future.result(timeout=5)

    async def probe() -> str:
        return "alive"

    assert runner.run(probe()) == "alive"


def test_package_has_no_asyncio_run_calls() -> None:
    package_root = Path("shandu")
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "asyncio.run(" in text:
            offenders.append(str(path))

    assert offenders == []
