from __future__ import annotations

import asyncio
from types import SimpleNamespace

from shandu.agents.search_subagent import SearchSubagent
from shandu.contracts import ResearchRequest, SubagentTask
from shandu.services.search import SearchHit


class FakeRuntime:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(model="deepseek/deepseek-chat")
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


class EmptyScrapeService:
    async def scrape_many(self, urls: list[str]):
        del urls
        return []


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
    assert {item.url for item in evidence} == {"https://example.com/a", "https://example.com/b"}
    assert all(item.confidence == 0.33 for item in evidence)


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
