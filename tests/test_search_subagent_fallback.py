from __future__ import annotations

import asyncio
from types import SimpleNamespace

from shandu.agents.search_subagent import SearchSubagent
from shandu.contracts import ResearchRequest, SubagentTask
from shandu.services.search import SearchHit
from shandu.services.scrape import ScrapedPage


class FakeRuntime:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(model="deepseek/deepseek-v4-flash")
        self.desk = SimpleNamespace(arun=None)


class FakeSearchService:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del query, max_results
        return [
            SearchHit(
                query="q",
                url="https://example.com/a",
                title="Alpha",
                snippet="Alpha snippet",
            ),
            SearchHit(
                query="q",
                url="https://example.com/b",
                title="Beta",
                snippet="Beta snippet",
            ),
        ]


class RootSearchService:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del query, max_results
        return [
            SearchHit(
                query="q",
                url="https://example.com",
                title="Root",
                snippet="Root snippet",
            )
        ]


class DocumentSearchService:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del query, max_results
        return [
            SearchHit(
                query="q",
                url="https://example.com/report.pdf",
                title="Report",
                snippet="Report snippet",
            )
        ]


class EmptyScrapeService:
    async def scrape_many(self, urls: list[str]):
        del urls
        return [], 0


class CanonicalScrapeService:
    async def scrape_many(self, urls: list[str]):
        del urls
        return [
            ScrapedPage(
                requested_url="https://example.com/",
                url="https://example.com/",
                title="Canonical",
                text="Canonical page text",
                domain="example.com",
            )
        ], 0


class DocumentScrapeService:
    async def scrape_many(self, urls: list[str]):
        del urls
        return [
            ScrapedPage(
                requested_url="https://example.com/report.pdf",
                url="https://example.com/report.pdf",
                title="Report",
                text="PDF text",
                domain="example.com",
                content_type="application/pdf",
            )
        ], 0


def test_search_subagent_uses_search_hit_fallback_when_scrape_fails() -> None:
    subagent = SearchSubagent(
        runtime=FakeRuntime(),
        search_service=FakeSearchService(),
        scrape_service=EmptyScrapeService(),
    )
    task = SubagentTask(
        task_id="task-1",
        focus="focus",
        search_queries=["query"],
        expected_output="out",
    )
    request = ResearchRequest(query="q", max_pages_per_task=2, max_results_per_query=2)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert len(evidence) == 2
    assert {item.requested_url for item in evidence} == {"https://example.com/a", "https://example.com/b"}
    assert all(item.confidence == 0.33 for item in evidence)


def test_search_subagent_does_not_duplicate_canonicalized_scrapes() -> None:
    subagent = SearchSubagent(
        runtime=FakeRuntime(),
        search_service=RootSearchService(),
        scrape_service=CanonicalScrapeService(),
    )
    task = SubagentTask(
        task_id="task-1",
        focus="focus",
        search_queries=["query"],
        expected_output="out",
    )
    request = ResearchRequest(query="q", max_pages_per_task=1, max_results_per_query=1)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert len(evidence) == 1
    assert evidence[0].requested_url == "https://example.com/"


def test_search_subagent_labels_document_evidence() -> None:
    subagent = SearchSubagent(
        runtime=FakeRuntime(),
        search_service=DocumentSearchService(),
        scrape_service=DocumentScrapeService(),
    )
    task = SubagentTask(
        task_id="task-1",
        focus="focus",
        search_queries=["query"],
        expected_output="out",
    )
    request = ResearchRequest(query="q", max_pages_per_task=1, max_results_per_query=1)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert evidence[0].source_type == "document"
    assert evidence[0].extraction_method == "pdf"


def test_search_subagent_emits_search_and_scrape_traces() -> None:
    subagent = SearchSubagent(
        runtime=FakeRuntime(),
        search_service=FakeSearchService(),
        scrape_service=EmptyScrapeService(),
    )
    task = SubagentTask(
        task_id="task-1",
        focus="focus",
        search_queries=["query"],
        expected_output="out",
    )
    request = ResearchRequest(query="q", max_pages_per_task=2, max_results_per_query=2)
    traces: list[str] = []

    async def on_trace(trace_type: str, payload: dict[str, object]) -> None:
        traces.append(trace_type)
        assert payload.get("task_id") == "task-1"

    asyncio.run(subagent.execute_task("run:1", task, request, progress_callback=on_trace))

    assert "query_started" in traces
    assert "query_completed" in traces
    assert "scrape_started" in traces
    assert "scrape_completed" in traces
    assert "fallback_evidence" in traces
