from __future__ import annotations

import asyncio
import importlib
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Collection, Mapping
from types import ModuleType
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from pydantic import BaseModel

from ..config import config
from .search_apis import API_ENV, ApiSearchClient

logger = logging.getLogger(__name__)
# ddgs picks a browser profile at random, and primp logs a warning for each
# one it does not know before it falls back to another, which lands in the
# middle of the CLI's progress stream. primp's Rust logger reads a Python
# logger's level on its first message only, so this must run before any search.
logging.getLogger("primp.impersonate").setLevel(logging.ERROR)

# ddgs 9.x text backends. A comma-delimited entry queries those engines in
# parallel threads and merges URL-deduped results, so the primary entry gets
# per-query source diversity and survives either engine having a bad day;
# "auto" stays as the everything-else fallback. "lite"/"html" were
# duckduckgo_search names rejected by ddgs. As of ddgs 9.10 in September
# 2026, "brave" answers HTTP 429 and "duckduckgo" returns no results on
# every query, while "bing" and "yahoo" both return full result pages.
_TEXT_BACKENDS: tuple[str, ...] = ("bing,yahoo", "auto")
_SEARCH_CACHE_MAX = 256
_SEARCH_CONCURRENCY = 4
_SEARCH_RETRY_DELAY_SECONDS = 1.0


class SearchHit(BaseModel):
    query: str
    url: str
    title: str
    snippet: str
    published_at: str | None = None


def _host_matches(url: str, domains: Collection[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


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
        self._api: ApiSearchClient | None = None
        backend = str(config.get("search", "backend", "ddgs"))
        if backend in API_ENV:
            credential = os.getenv(API_ENV[backend], "").strip()
            if credential:
                self._api = ApiSearchClient(backend, credential)
            else:
                logger.warning(
                    "Search backend %s needs %s; searching with ddgs", backend, API_ENV[backend]
                )
        self._region = str(config.get("search", "region", "wt-wt"))
        self._safesearch = str(config.get("search", "safesearch", "moderate"))
        self._cache: OrderedDict[str, tuple[float, list[SearchHit]]] = OrderedDict()
        self._cache_ttl = 300.0
        self._inflight: dict[str, asyncio.Task[list[SearchHit]]] = {}
        self._search_semaphore = asyncio.Semaphore(_SEARCH_CONCURRENCY)
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
        if self._ddgs is None and self._api is None:
            return []

        key = self._cache_key(query, max_results)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._do_search(key, query, max_results))
            self._inflight[key] = task
            task.add_done_callback(lambda _: self._inflight.pop(key, None))
        # Shielded so that one caller's cancellation leaves the search the
        # other callers wait on running.
        return await asyncio.shield(task)

    async def _do_search(self, key: str, query: str, max_results: int) -> list[SearchHit]:
        self._errors.pop(key, None)
        async with self._search_semaphore:
            raw, backend_errors = await self._try_backends(query, max_results)
            if not raw and backend_errors:
                await asyncio.sleep(_SEARCH_RETRY_DELAY_SECONDS)
                raw, backend_errors = await self._try_backends(query, max_results)
        if not raw:
            attempted = 1 if self._api is not None else len(_TEXT_BACKENDS)
            if len(backend_errors) == attempted:
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
            published_at = entry.get("published_at")
            hits.append(
                SearchHit(
                    query=query,
                    url=url,
                    title=str(entry.get("title") or url).strip(),
                    snippet=str(entry.get("body") or "").strip(),
                    published_at=str(published_at) if published_at else None,
                )
            )
            if len(hits) >= max_results:
                break

        self._set_cached(key, hits)
        return hits

    async def _try_backends(
        self, query: str, max_results: int
    ) -> tuple[list[Mapping[str, Any]] | None, list[str]]:
        if self._api is not None:
            try:
                return list(await self._api.search(query, max_results)), []
            except Exception as exc:
                logger.warning(
                    "Search backend %s failed: %s", self._api.provider, str(exc)[:200]
                )
                return None, [f"{self._api.provider}: {exc}"]
        raw: list[Mapping[str, Any]] | None = None
        backend_errors: list[str] = []
        for backend in _TEXT_BACKENDS:
            try:
                raw = await asyncio.to_thread(
                    self._fetch_backend, query, max_results, backend
                )
            except Exception as exc:
                raw = None
                backend_errors.append(f"{backend}: {exc}")
                # One line: the error also reaches the run as a search_failed
                # event, and a rate-limit traceback per query floods the CLI.
                logger.warning("Search backend %s failed: %s", backend, str(exc)[:200])
            if raw:
                break
        return raw, backend_errors

    async def aclose(self) -> None:
        if self._api is not None:
            await self._api.aclose()

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
