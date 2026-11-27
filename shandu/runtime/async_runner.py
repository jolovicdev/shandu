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
                loop: asyncio.AbstractEventLoop | None = None
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self._loop = loop
                finally:
                    self._ready.set()
                if loop is not None:
                    loop.run_forever()

            self._thread = threading.Thread(target=runner, daemon=True)
            self._thread.start()

    def submit(self, awaitable: Any) -> Future[Any]:
        try:
            self._start()
            self._ready.wait()
            if self._loop is None:
                raise RuntimeError("Async runner loop is not initialized")
            done = threading.Event()

            async def _tracked() -> Any:
                try:
                    return await awaitable
                finally:
                    done.set()

            future = asyncio.run_coroutine_threadsafe(_tracked(), self._loop)
            setattr(future, "_shandu_done", done)
            return future
        except BaseException:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise

    @staticmethod
    def cancel_and_wait(future: Future[Any], timeout: float = 5.0) -> None:
        """Cancel a submitted future and wait for its coroutine cleanup to run."""
        future.cancel()
        done = getattr(future, "_shandu_done", None)
        if isinstance(done, threading.Event):
            done.wait(timeout=timeout)
        else:
            try:
                future.result(timeout=timeout)
            except BaseException:
                pass

    def run(self, awaitable: Any) -> Any:
        if threading.current_thread() is self._thread:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise RuntimeError("AsyncRunner.run() cannot be called from its own event loop thread")
        future = self.submit(awaitable)
        try:
            return future.result()
        except BaseException:
            self.cancel_and_wait(future)
            raise


_runner: AsyncRunner | None = None
_runner_lock = threading.Lock()


def get_async_runner() -> AsyncRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = AsyncRunner()
    return _runner
