from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections import OrderedDict
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Protocol, cast

from pydantic import BaseModel

from ..config import config

logger = logging.getLogger(__name__)

# ddgs 9.x text backends. A comma-delimited entry queries those engines in
# parallel threads and merges URL-deduped results, so the primary entry gets
# per-query source diversity and survives either engine having a bad day;
# "auto" stays as the everything-else fallback. "lite"/"html" were
# duckduckgo_search names rejected by ddgs; "bing" is disabled and "google"
# is blocked (HTTP 403) as of ddgs 9.10.
_TEXT_BACKENDS: tuple[str, ...] = ("brave,duckduckgo", "auto")
_SEARCH_CACHE_MAX = 256


class SearchHit(BaseModel):
    query: str
    url: str
    title: str
    snippet: str


class _DDGSClient(Protocol):
    def text(
        self,
        *,
        query: str,
        region: str,
        safesearch: str,
        max_results: int,
        backend: str,
    ) -> list[Mapping[str, Any]]: ...


class _DDGSFactory(Protocol):
    def __call__(self, *, timeout: int) -> _DDGSClient: ...


def _resolve_ddgs() -> _DDGSFactory | None:
    try:
        module: ModuleType = importlib.import_module("ddgs")
    except Exception:
        return None
    cls = getattr(module, "DDGS", None)
    if callable(cls):
        return cast(_DDGSFactory, cls)
    return None


class SearchService:
    def __init__(self) -> None:
        self._ddgs = _resolve_ddgs()
        self._region = str(config.get("search", "region", "wt-wt"))
        self._safesearch = str(config.get("search", "safesearch", "moderate"))
        self._cache: OrderedDict[str, tuple[float, list[SearchHit]]] = OrderedDict()
        self._cache_ttl = 300.0
        self._inflight: dict[str, asyncio.Task[list[SearchHit]]] = {}
        self.last_error: str | None = None
        self._errors: dict[str, str] = {}

    def last_error_for(self, query: str, max_results: int) -> str | None:
        # last_error holds the latest outcome only; concurrent callers need
        # the error of their own query.
        return self._errors.get(self._cache_key(query, max_results))

    def _cache_key(self, query: str, max_results: int) -> str:
        return f"{query}:{max_results}:{self._region}:{self._safesearch}"

    def _get_cached(self, key: str) -> list[SearchHit] | None:
        if key not in self._cache:
            return None
        timestamp, hits = self._cache[key]
        if time.monotonic() - timestamp > self._cache_ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return hits

    def _set_cached(self, key: str, hits: list[SearchHit]) -> None:
        self._cache[key] = (time.monotonic(), hits)
        self._cache.move_to_end(key)
        while len(self._cache) > _SEARCH_CACHE_MAX:
            self._cache.popitem(last=False)

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        if self._ddgs is None:
            return []

        key = self._cache_key(query, max_results)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        in_flight = self._inflight.get(key)
        if in_flight is not None:
            return await in_flight

        task = asyncio.create_task(self._do_search(key, query, max_results))
        self._inflight[key] = task
        try:
            return await task
        finally:
            self._inflight.pop(key, None)

    async def _do_search(self, key: str, query: str, max_results: int) -> list[SearchHit]:
        self._errors.pop(key, None)
        raw: list[Mapping[str, Any]] | None = None
        backend_errors: list[str] = []
        for backend in _TEXT_BACKENDS:
            try:
                raw = await asyncio.to_thread(self._fetch_backend, query, max_results, backend)
            except Exception as exc:
                raw = None
                backend_errors.append(f"{backend}: {exc}")
                logger.warning("Search backend %s failed", backend, exc_info=True)
            if raw:
                break
        if not raw:
            if len(backend_errors) == len(_TEXT_BACKENDS):
                self.last_error = self._errors[key] = "; ".join(backend_errors)[:300]
                logger.warning("All search backends failed for query %r", query)
            else:
                self.last_error = None
            return []
        self.last_error = None

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for entry in raw or []:
            url = str(entry.get("href", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append(
                SearchHit(
                    query=query,
                    url=url,
                    title=str(entry.get("title", url)).strip(),
                    snippet=str(entry.get("body", "")).strip(),
                )
            )
            if len(hits) >= max_results:
                break

        self._set_cached(key, hits)
        return hits

    def _fetch_backend(
        self,
        query: str,
        max_results: int,
        backend: str,
    ) -> list[Mapping[str, Any]]:
        if self._ddgs is None:
            return []
        client = self._ddgs(timeout=12)
        return list(
            client.text(
                query=query,
                region=self._region,
                safesearch=self._safesearch,
                max_results=max_results,
                backend=backend,
            )
        )
