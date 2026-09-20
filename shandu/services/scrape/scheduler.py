from __future__ import annotations

import asyncio
import time


class _DomainScheduler:
    def __init__(self, max_concurrent_per_domain: int = 2, base_delay: float = 0.5) -> None:
        self._max_concurrent = max(1, max_concurrent_per_domain)
        self._base_delay = base_delay
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._last_fetch: dict[str, float] = {}
        self._backoff: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, domain: str) -> None:
        async with self._lock:
            sem = self._semaphores.setdefault(domain, asyncio.Semaphore(self._max_concurrent))
        await sem.acquire()
        try:
            async with self._lock:
                last = self._last_fetch.get(domain, 0)
                back = self._backoff.get(domain, 0)
                now = time.monotonic()
                # Reserve the fire time before sleeping so concurrent slots
                # for one domain stagger instead of waking together.
                fire_at = max(now, last + back + self._base_delay)
                self._last_fetch[domain] = fire_at
                wait = fire_at - now
            if wait > 0:
                await asyncio.sleep(wait)
        except asyncio.CancelledError:
            sem.release()
            raise

    async def release(self, domain: str) -> None:
        async with self._lock:
            # A slow fetch pushes spacing out; a fast one keeps the
            # reservation so the next slot still waits its turn.
            self._last_fetch[domain] = max(
                self._last_fetch.get(domain, 0), time.monotonic()
            )
        sem = self._semaphores.get(domain)
        if sem:
            sem.release()

    def bump_backoff(self, domain: str) -> None:
        current = self._backoff.get(domain, 0)
        self._backoff[domain] = min(current * 2 + 1, 60) if current else 2

    def reset_backoff(self, domain: str) -> None:
        self._backoff.pop(domain, None)
