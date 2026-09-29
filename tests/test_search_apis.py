from __future__ import annotations

import asyncio

import pytest

from shandu.config import config
from shandu.services.search import SearchService
from shandu.services.search_apis import ApiSearchClient

_CASES = {
    "tavily": (
        {"results": [{"url": "https://a.org", "title": "A", "content": "body", "published_date": "2025-01-02"}]},
        ("POST", "https://api.tavily.com/search"),
        ("headers", "Authorization", "Bearer secret"),
        ("json", "max_results", 20),
    ),
    "brave": (
        {"web": {"results": [{"url": "https://a.org", "title": "A", "description": "body", "page_age": "2025-01-02"}]}},
        ("GET", "https://api.search.brave.com/res/v1/web/search"),
        ("headers", "X-Subscription-Token", "secret"),
        ("params", "count", 20),
    ),
    "exa": (
        {"results": [{"url": "https://a.org", "title": "A", "highlights": ["bo", "dy"], "publishedDate": "2025-01-02"}]},
        ("POST", "https://api.exa.ai/search"),
        ("headers", "x-api-key", "secret"),
        ("json", "numResults", 20),
    ),
    "searxng": (
        {"results": [{"url": "https://a.org", "title": "A", "content": "body", "publishedDate": "2025-01-02"}]},
        ("GET", "https://searx.example.org/search"),
        ("params", "format", "json"),
        ("params", "q", "kv cache"),
    ),
}


@pytest.mark.parametrize("provider", sorted(_CASES))
def test_api_client_request_and_mapping(provider) -> None:
    payload, endpoint, *expected = _CASES[provider]
    credential = "https://searx.example.org/" if provider == "searxng" else "secret"
    client = ApiSearchClient(provider, credential)
    seen: dict = {}

    async def fake_request(method, url, **kwargs):
        seen.update(method=method, url=url, **kwargs)
        return payload

    client._request = fake_request
    entries = asyncio.run(client.search("kv cache", 50))

    assert (seen["method"], seen["url"]) == endpoint
    for section, key, value in expected:
        assert seen[section][key] == value
    assert [dict(entry) for entry in entries] == [
        {
            "href": "https://a.org",
            "title": "A",
            "body": "bo dy" if provider == "exa" else "body",
            "published_at": "2025-01-02",
        }
    ]


def test_search_service_uses_configured_api_and_reports_its_failures(monkeypatch) -> None:
    monkeypatch.setitem(config._config["search"], "backend", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    service = SearchService()
    service._ddgs = None

    async def good(query, max_results):
        return [{"href": "https://a.org", "title": "A", "body": "b", "published_at": "2025-01-02"}]

    service._api.search = good
    hits = asyncio.run(service.search("kv cache", 5))
    assert [(hit.url, hit.published_at) for hit in hits] == [("https://a.org", "2025-01-02")]

    async def down(query, max_results):
        raise ConnectionError("402 payment required")

    service._api.search = down
    assert asyncio.run(service.search("other query", 5)) == []
    assert "tavily" in service.last_error_for("other query", 5)


def test_search_service_without_api_key_falls_back_to_ddgs(monkeypatch, caplog) -> None:
    monkeypatch.setitem(config._config["search"], "backend", "exa")
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    service = SearchService()

    assert service._api is None
    assert "EXA_API_KEY" in caplog.text
