from __future__ import annotations

import asyncio
import json
import logging

from shandu.services import scholarly
from shandu.services.scholarly import ScholarlySearchService

_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <published>2025-01-02T00:00:00Z</published>
    <title>Paper
      Alpha</title>
    <summary>Alpha abstract.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.00002v1</id>
    <published>2025-01-03T00:00:00Z</published>
    <title>Paper Beta: Results</title>
    <summary>Beta abstract.</summary>
  </entry>
</feed>"""

_OPENALEX = {
    "results": [
        {
            "display_name": "Paper beta - results",
            "doi": "https://doi.org/10.1/b",
            "publication_date": "2024-11-30",
            "abstract_inverted_index": {"measures": [1], "Beta": [0], "things": [2]},
            "best_oa_location": {"pdf_url": "https://oa.example.org/beta.pdf"},
        },
        {
            "display_name": "Paper Gamma",
            "doi": "https://doi.org/10.1/g",
            "publication_date": "2023-05-01",
            "primary_location": {"landing_page_url": "http://x.example.org/a.html>,"},
        },
    ]
}

_PUBMED_SEARCH = {"esearchresult": {"idlist": ["111", "222"]}}
_PUBMED_SUMMARY = {
    "result": {
        "uids": ["111", "222"],
        "111": {
            "title": "Paper Delta",
            "sortpubdate": "2024/07/24 00:00",
            "fulljournalname": "Journal of Deltas",
            "pubdate": "2024 Jul 24",
            "articleids": [
                {"idtype": "pubmed", "value": "111"},
                {"idtype": "pmc", "value": "PMC999"},
            ],
        },
        "222": {"title": "Paper Epsilon", "sortpubdate": "2022/01/05 00:00"},
    }
}


def _service(failing: set[str] = frozenset()) -> ScholarlySearchService:
    service = ScholarlySearchService()
    responses = {
        scholarly._ARXIV_URL: _ARXIV_FEED,
        scholarly._OPENALEX_URL: json.dumps(_OPENALEX),
        scholarly._PUBMED_SEARCH_URL: json.dumps(_PUBMED_SEARCH),
        scholarly._PUBMED_SUMMARY_URL: json.dumps(_PUBMED_SUMMARY),
    }

    async def fake_get(url, params):
        service.seen_params.append(params)
        if any(name in url for name in failing):
            raise ConnectionError(f"{url} down")
        return responses[url]

    service.seen_params = []
    service._get = fake_get
    return service


def test_merges_providers_round_robin_with_dedup_and_parsed_fields() -> None:
    service = _service()
    hits = asyncio.run(service.search("q site:example.org", 8))

    assert {"search_query": "all:q", "max_results": 8, "sortBy": "relevance"} in (
        service.seen_params
    )

    assert [hit.title for hit in hits] == [
        "Paper Alpha",
        "Paper beta - results",
        "Paper Delta",
        "Paper Gamma",
        "Paper Epsilon",
    ]
    alpha, beta, delta, gamma, epsilon = hits
    assert alpha.url == "https://arxiv.org/abs/2501.00001v1"
    assert alpha.published_at == "2025-01-02"
    assert beta.url == "https://oa.example.org/beta.pdf"
    assert beta.snippet == "Beta measures things"
    assert delta.url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC999/"
    assert epsilon.url == "https://pubmed.ncbi.nlm.nih.gov/222/"
    assert delta.published_at == "2024-07-24"
    assert delta.snippet == "Journal of Deltas, 2024 Jul 24"
    assert gamma.url == "https://doi.org/10.1/g"


def test_query_fails_only_when_every_provider_fails() -> None:
    partial = _service(failing={"arxiv"})
    hits = asyncio.run(partial.search("q", 8))
    assert [hit.title for hit in hits][:2] == ["Paper beta - results", "Paper Delta"]
    assert partial.last_error_for("q", 8) is None

    broken = _service(failing={"arxiv", "openalex", "eutils"})
    assert asyncio.run(broken.search("q", 8)) == []
    error = broken.last_error_for("q", 8)
    assert error is not None
    assert "arxiv" in error and "openalex" in error and "pubmed" in error


def test_titles_in_other_scripts_are_not_merged_as_duplicates() -> None:
    from shandu.services.search import SearchHit

    service = ScholarlySearchService()
    titles = ["量子计算综述", "Обзор квантовых вычислений", "量子计算综述。", "!!!", "???"]

    async def openalex(query, terms, max_results):
        return [
            SearchHit(query=query, url=f"https://doi.org/{index}", title=title, snippet="")
            for index, title in enumerate(titles)
        ]

    async def nothing(query, terms, max_results):
        return []

    service._search_openalex = openalex
    service._search_arxiv = nothing
    service._search_pubmed = nothing

    hits = asyncio.run(service.search("q", 10))

    assert [hit.title for hit in hits] == [
        "量子计算综述",
        "Обзор квантовых вычислений",
        "!!!",
        "???",
    ]


def test_a_spent_rate_limit_pauses_the_provider_with_one_warning(caplog) -> None:
    from shandu.services.search import SearchHit

    service = ScholarlySearchService()
    calls: list[str] = []

    async def spent(query, terms, max_results):
        calls.append(query)
        raise scholarly._RateLimitSpent(3600)

    async def one_hit(query, terms, max_results):
        return [SearchHit(query=query, url="https://arxiv.org/abs/1", title="A", snippet="")]

    service._search_openalex = spent
    service._search_arxiv = one_hit
    service._search_pubmed = one_hit

    async def run():
        await asyncio.gather(service.search("a", 8), service.search("b", 8))
        return await service.search("c", 8)

    with caplog.at_level(logging.WARNING, logger="shandu.services.scholarly"):
        hits = asyncio.run(run())

    assert sorted(calls) == ["a", "b"]
    assert [hit.title for hit in hits] == ["A"]
    assert service.last_error_for("c", 8) is None
    warnings = [record.message for record in caplog.records]
    assert len(warnings) == 1
    assert "openalex is rate limited for 60 minutes" in warnings[0]


def test_only_a_long_retry_after_skips_the_retry(monkeypatch) -> None:
    from aiohttp import web

    monkeypatch.setattr(scholarly, "_RATE_LIMITED_RETRY_SECONDS", 0)

    async def run(retry_after: str) -> tuple[BaseException | None, int]:
        requests: list[int] = []

        async def handler(request: web.Request) -> web.Response:
            requests.append(1)
            return web.Response(status=429, headers={"Retry-After": retry_after})

        app = web.Application()
        app.router.add_get("/", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        service = ScholarlySearchService()
        try:
            await service._get(f"http://127.0.0.1:{port}/", {})
            error = None
        except Exception as caught:
            error = caught
        finally:
            await service.aclose()
            await runner.cleanup()
        return error, len(requests)

    spent, spent_requests = asyncio.run(run("22539"))
    assert isinstance(spent, scholarly._RateLimitSpent)
    assert spent.seconds == 22539
    assert spent_requests == 1

    burst, burst_requests = asyncio.run(run("1"))
    assert not isinstance(burst, scholarly._RateLimitSpent)
    assert burst_requests == 2
