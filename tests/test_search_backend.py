from __future__ import annotations

import asyncio
import threading
import time
from unittest import mock

from shandu.services.search import SearchService


def test_search_service_constructs() -> None:
    service = SearchService()
    assert service.last_error is None
    assert service._cache == {}
    assert service._inflight == {}
    assert service._region
    assert service._safesearch
    assert service._get_cached(service._cache_key("q", 3)) is None


def test_search_backends_pair_engines_with_auto_fallback() -> None:
    from shandu.services.search import _TEXT_BACKENDS

    assert "lite" not in _TEXT_BACKENDS
    assert "html" not in _TEXT_BACKENDS
    primary = _TEXT_BACKENDS[0].split(",")
    assert primary == ["bing", "yahoo"]
    assert _TEXT_BACKENDS[-1] == "auto"


def test_search_cache_evicts_oldest() -> None:
    from shandu.services.search import _SEARCH_CACHE_MAX

    service = SearchService()
    for i in range(_SEARCH_CACHE_MAX + 5):
        service._set_cached(f"k{i}", [])
    assert len(service._cache) == _SEARCH_CACHE_MAX
    assert "k0" not in service._cache
    assert f"k{_SEARCH_CACHE_MAX + 4}" in service._cache


class _BlockingClient:
    def __init__(self, tracker: dict[str, int], lock: threading.Lock) -> None:
        self._tracker = tracker
        self._lock = lock

    def text(self, *, query, region, safesearch, max_results, backend):
        del region, safesearch, max_results, backend
        with self._lock:
            self._tracker["active"] += 1
            self._tracker["peak"] = max(
                self._tracker["peak"], self._tracker["active"]
            )
        try:
            time.sleep(0.05)
        finally:
            with self._lock:
                self._tracker["active"] -= 1
        return [{"href": f"https://x.example/{query}", "title": "t", "body": "b"}]


def test_search_service_bounds_concurrent_backend_calls() -> None:
    tracker = {"active": 0, "peak": 0}
    lock = threading.Lock()
    service = SearchService()
    service._ddgs = lambda *, timeout: _BlockingClient(tracker, lock)

    async def run_all() -> None:
        await asyncio.gather(
            *(service.search(f"query-{index}", 3) for index in range(6))
        )

    asyncio.run(run_all())

    assert 1 < tracker["peak"] <= 4


class _SlowClient:
    def __init__(self) -> None:
        self.calls = 0

    def text(self, *, query, region, safesearch, max_results, backend):
        del query, region, safesearch, max_results, backend
        self.calls += 1
        time.sleep(0.05)
        return [{"href": "https://x.example/a", "title": "t", "body": "b"}]


def test_cancelling_one_searcher_leaves_the_shared_search_to_the_others() -> None:
    service = SearchService()
    client = _SlowClient()
    service._ddgs = lambda *, timeout: client

    async def run_all() -> tuple[asyncio.Task, list]:
        first = asyncio.create_task(service.search("q", 3))
        second = asyncio.create_task(service.search("q", 3))
        await asyncio.sleep(0.01)
        first.cancel()
        third = asyncio.create_task(service.search("q", 3))
        results = await asyncio.gather(second, third)
        await asyncio.gather(first, return_exceptions=True)
        await asyncio.sleep(0)
        return first, results

    first, results = asyncio.run(run_all())

    assert first.cancelled()
    assert [[hit.url for hit in hits] for hits in results] == [["https://x.example/a"]] * 2
    assert client.calls == 1
    assert service._inflight == {}


class _FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def text(self, *, query, region, safesearch, max_results, backend):
        del query, region, safesearch, max_results, backend
        self.calls += 1
        if self.calls <= 2:
            raise RuntimeError("429 rate limited")
        return [{"href": "https://x.example/a", "title": "t", "body": "b"}]


def test_search_service_retries_once_after_backend_errors() -> None:
    service = SearchService()
    client = _FlakyClient()
    service._ddgs = lambda *, timeout: client

    with mock.patch(
        "shandu.services.search._SEARCH_RETRY_DELAY_SECONDS", 0
    ):
        hits = asyncio.run(service.search("q", 3))

    assert [hit.url for hit in hits] == ["https://x.example/a"]
    assert client.calls == 3
    assert service.last_error is None


class _CountingEmptyClient:
    def __init__(self) -> None:
        self.calls = 0

    def text(self, *, query, region, safesearch, max_results, backend):
        del query, region, safesearch, max_results, backend
        self.calls += 1
        return []


def test_search_service_does_not_retry_empty_results() -> None:
    from shandu.services.search import _TEXT_BACKENDS

    service = SearchService()
    client = _CountingEmptyClient()
    service._ddgs = lambda *, timeout: client

    hits = asyncio.run(service.search("q", 3))

    assert hits == []
    assert client.calls == len(_TEXT_BACKENDS)
    assert service.last_error is None
