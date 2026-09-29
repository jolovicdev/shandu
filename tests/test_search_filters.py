from __future__ import annotations

import asyncio
from types import SimpleNamespace

from shandu.agents.search_subagent import SearchSubagent, _ExtractionPayload
from shandu.contracts import ResearchRequest, SubagentTask
from shandu.services.scrape import ScrapedPage
from shandu.services.search import SearchHit


class _RecordingSearch:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del max_results
        self.queries.append(query)
        return [SearchHit(query=query, url=url, title=url, snippet="s") for url in self.urls]


class _EchoScrape:
    def __init__(self, dates: dict[str, str | None] | None = None) -> None:
        self.dates = dates or {}
        self.requested: list[str] = []

    async def scrape_many(self, urls):
        self.requested.extend(urls)
        return [
            ScrapedPage(
                requested_url=url,
                url=url,
                title=url,
                text="page text",
                domain="example.org",
                published_at=self.dates.get(url),
            )
            for url in urls
        ], 0


class _ExtractDesk:
    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(
            status="completed",
            data=_ExtractionPayload(snippet="s", extracted_text="t"),
            metrics={},
        )


def _subagent(search, scrape) -> SearchSubagent:
    runtime = SimpleNamespace(settings=SimpleNamespace(model="m"), desk=_ExtractDesk())
    return SearchSubagent(runtime, search, scrape)


def _task() -> SubagentTask:
    return SubagentTask(task_id="t1", focus="focus", search_queries=["kv cache"])


def test_domain_filters_scope_queries_and_hits() -> None:
    search = _RecordingSearch(
        [
            "https://a.example.org/x",
            "https://other.com/y",
            "https://blocked.example.org/z",
        ]
    )
    scrape = _EchoScrape()
    request = ResearchRequest(
        query="q",
        include_domains=["example.org"],
        exclude_domains=["blocked.example.org"],
    )

    asyncio.run(_subagent(search, scrape).execute_task("run:r", _task(), request))

    assert search.queries == ["kv cache site:example.org"]
    assert scrape.requested == ["https://a.example.org/x"]


def test_since_skips_pages_dated_before_and_keeps_undated_and_coarse() -> None:
    old, coarse, undated = (
        "https://example.org/old",
        "https://example.org/coarse",
        "https://example.org/undated",
    )
    search = _RecordingSearch([old, coarse, undated])
    scrape = _EchoScrape({old: "2023-05-01T10:00:00Z", coarse: "2024", undated: None})
    traces: list[tuple[str, dict]] = []

    async def on_trace(trace_type, payload) -> None:
        traces.append((trace_type, payload))

    evidence = asyncio.run(
        _subagent(search, scrape).execute_task(
            "run:r",
            _task(),
            ResearchRequest(query="q", since="2024-06-01"),
            progress_callback=on_trace,
        )
    )

    assert sorted(item.requested_url for item in evidence) == [coarse, undated]
    completed = next(payload for kind, payload in traces if kind == "scrape_completed")
    assert completed["dated_before_since"] == 1


def test_scholarly_task_also_searches_papers_and_drops_old_ones_before_scraping() -> None:
    web = _RecordingSearch(["https://example.org/web"])
    papers = _RecordingSearch(["https://arxiv.org/abs/new", "https://arxiv.org/abs/old"])

    async def dated_search(query, max_results):
        hits = await _RecordingSearch.search(papers, query, max_results)
        dates = {"https://arxiv.org/abs/new": "2025-02-01", "https://arxiv.org/abs/old": "2019-01-01"}
        return [hit.model_copy(update={"published_at": dates[hit.url]}) for hit in hits]

    papers.search = dated_search
    scrape = _EchoScrape()
    runtime = SimpleNamespace(settings=SimpleNamespace(model="m"), desk=_ExtractDesk())
    subagent = SearchSubagent(runtime, web, scrape, papers)
    request = ResearchRequest(query="q", since="2024-01-01")

    asyncio.run(subagent.execute_task("run:r", _task(), request))
    assert papers.queries == []

    scholarly_task = _task().model_copy(update={"scholarly": True})
    asyncio.run(subagent.execute_task("run:r", scholarly_task, request))
    assert papers.queries == ["kv cache"]
    assert "https://arxiv.org/abs/new" in scrape.requested
    assert "https://arxiv.org/abs/old" not in scrape.requested
