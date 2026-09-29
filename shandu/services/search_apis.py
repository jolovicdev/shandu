from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import aiohttp

# Each API's environment variable: a key, or for SearXNG the instance URL.
API_ENV = {
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_API_KEY",
    "exa": "EXA_API_KEY",
    "searxng": "SEARXNG_URL",
}
_TIMEOUT = aiohttp.ClientTimeout(total=20)
# Tavily and Brave cap one request at 20 results.
_MAX_RESULTS_PER_CALL = 20


class ApiSearchClient:
    """Web search through a hosted search API instead of scraped engines.

    Results use the ddgs field names (href, title, body) so the search
    service handles both sources the same way.
    """

    def __init__(self, provider: str, credential: str) -> None:
        self.provider = provider
        self._credential = credential
        self._session: aiohttp.ClientSession | None = None

    @classmethod
    def from_env(cls, provider: str) -> ApiSearchClient | None:
        credential = os.getenv(API_ENV[provider], "").strip()
        return cls(provider, credential) if credential else None

    async def search(self, query: str, max_results: int) -> list[Mapping[str, Any]]:
        count = min(max_results, _MAX_RESULTS_PER_CALL)
        if self.provider == "tavily":
            payload = await self._request(
                "POST",
                "https://api.tavily.com/search",
                json={"query": query, "max_results": count, "search_depth": "basic"},
                headers={"Authorization": f"Bearer {self._credential}"},
            )
            items, body, date = payload.get("results"), "content", "published_date"
        elif self.provider == "brave":
            payload = await self._request(
                "GET",
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"X-Subscription-Token": self._credential, "Accept": "application/json"},
            )
            items, body, date = (payload.get("web") or {}).get("results"), "description", "page_age"
        elif self.provider == "exa":
            payload = await self._request(
                "POST",
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "numResults": count,
                    "type": "auto",
                    "contents": {"highlights": True},
                },
                headers={"x-api-key": self._credential},
            )
            items, body, date = payload.get("results"), "highlights", "publishedDate"
        else:
            payload = await self._request(
                "GET",
                f"{self._credential.rstrip('/')}/search",
                params={"q": query, "format": "json"},
            )
            items, body, date = payload.get("results"), "content", "publishedDate"
        entries: list[Mapping[str, Any]] = []
        for item in (items or [])[:count]:
            text = item.get(body)
            entries.append(
                {
                    "href": item.get("url"),
                    "title": item.get("title"),
                    # Exa returns highlight passages instead of one snippet.
                    "body": " ".join(text) if isinstance(text, list) else text,
                    "published_at": item.get(date),
                }
            )
        return entries

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_TIMEOUT)
        async with self._session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            payload = await response.json(content_type=None)
        return payload if isinstance(payload, dict) else {}
