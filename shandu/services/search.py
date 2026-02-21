from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Protocol, cast

from pydantic import BaseModel

from ..config import config


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

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        if self._ddgs is None:
            return []

        raw: list[Mapping[str, Any]] | None = None
        for backend in ("duckduckgo", "lite", "html", "auto"):
            try:
                raw = await asyncio.to_thread(self._fetch_backend, query, max_results, backend)
            except Exception:
                raw = None
            if raw:
                break
        if not raw:
            return []

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
