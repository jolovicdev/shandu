from __future__ import annotations

import asyncio
from pathlib import Path

from shandu.runtime.async_runner import get_async_runner


def test_async_runner_reuses_single_event_loop() -> None:
    runner = get_async_runner()

    async def current_loop_id() -> int:
        return id(asyncio.get_running_loop())

    first = runner.run(current_loop_id())
    second = runner.run(current_loop_id())

    assert first == second


def test_package_has_no_asyncio_run_calls() -> None:
    package_root = Path("shandu")
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "asyncio.run(" in text:
            offenders.append(str(path))

    assert offenders == []
