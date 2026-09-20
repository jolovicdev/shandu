from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any


class AsyncRunner:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return

            def runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._ready.set()
                loop.run_forever()

            self._thread = threading.Thread(target=runner, daemon=True)
            self._thread.start()

    def submit(self, awaitable: Any) -> Future[Any]:
        self._start()
        self._ready.wait()
        if self._loop is None:
            raise RuntimeError("Async runner loop is not initialized")
        return asyncio.run_coroutine_threadsafe(awaitable, self._loop)

    def run(self, awaitable: Any) -> Any:
        return self.submit(awaitable).result()


_runner: AsyncRunner | None = None
_runner_lock = threading.Lock()


def get_async_runner() -> AsyncRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = AsyncRunner()
    return _runner
