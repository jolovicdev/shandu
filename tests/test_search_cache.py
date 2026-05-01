from __future__ import annotations

import asyncio
import time

from shandu.services.search import SearchService


def test_search_cache_deduplicates_queries(monkeypatch):
    call_count = 0

    class FakeDDGS:
        def __init__(self, *, timeout: int):
            pass

        def text(self, **kwargs):
            nonlocal call_count
            call_count += 1
            return [
                {"href": "https://example.com/1", "title": "T1", "body": "B1"},
            ]

    monkeypatch.setattr("shandu.services.search._resolve_ddgs", lambda: FakeDDGS)
    service = SearchService()

    async def run_searches():
        hits1 = await service.search("quantum computing", 5)
        hits2 = await service.search("quantum computing", 5)
        return hits1, hits2

    hits1, hits2 = asyncio.run(run_searches())

    assert len(hits1) == 1
    assert hits1[0].url == "https://example.com/1"
    assert len(hits2) == 1
    # Only one actual DDGS call due to cache
    assert call_count == 1


def test_search_cache_respects_different_params(monkeypatch):
    calls: list[str] = []

    class CountingDDGS:
        def __init__(self, *, timeout: int):
            pass

        def text(self, *, query, region, safesearch, max_results, backend):
            calls.append(f"{query}:{max_results}")
            return [{"href": f"https://example.com/{max_results}", "title": "T", "body": "B"}]

    monkeypatch.setattr("shandu.services.search._resolve_ddgs", lambda: CountingDDGS)
    service = SearchService()

    asyncio.run(service.search("ai", 5))
    asyncio.run(service.search("ai", 10))

    assert len(calls) == 2
    assert calls[0] == "ai:5"
    assert calls[1] == "ai:10"


def test_search_cache_expires_after_ttl(monkeypatch):
    class MinimalDDGS:
        def __init__(self, *, timeout: int):
            pass

        def text(self, **kwargs):
            return [{"href": "https://x.com", "title": "X", "body": "Y"}]

    monkeypatch.setattr("shandu.services.search._resolve_ddgs", lambda: MinimalDDGS)
    service = SearchService()
    service._cache_ttl = 0.01

    asyncio.run(service.search("test", 3))
    time.sleep(0.02)
    cached = service._get_cached(service._cache_key("test", 3))
    assert cached is None
