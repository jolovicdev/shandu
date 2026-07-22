from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import get_args

from shandu.agents.search_subagent import (
    SearchSubagent,
    _ExtractionPayload,
    assess_source_quality,
)
from shandu.contracts import ResearchRequest, SourceClass, SubagentTask
from shandu.services.search import SearchHit
from shandu.services.scrape import ScrapedPage


class AlphaSearch:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del query, max_results
        return [
            SearchHit(
                query="q",
                url="https://example.com/a",
                title="Alpha",
                snippet="Alpha snippet",
            )
        ]


class _ModelRuntime:
    def __init__(self, desk: object) -> None:
        self.settings = SimpleNamespace(model="m")
        self.desk = desk


class _StaticDesk:
    def __init__(self, data: _ExtractionPayload) -> None:
        self._data = data

    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(status="completed", data=self._data)


class _RaisingDesk:
    async def arun(self, worker, job):
        del worker, job
        raise RuntimeError("boom")


class _SinglePageScrape:
    def __init__(self, page: ScrapedPage) -> None:
        self._page = page

    async def scrape_many(self, urls):
        del urls
        return [self._page], 0


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


def test_extraction_preserves_page_order_under_concurrency() -> None:
    from shandu.agents.search_subagent import _ExtractionPayload

    class ThreeHitSearch:
        async def search(self, query: str, max_results: int):
            del query, max_results
            return [
                SearchHit(query="q", url=f"https://example.com/{i}", title=f"H{i}", snippet=f"s{i}")
                for i in range(3)
            ]

    class ThreePageScrape:
        async def scrape_many(self, urls: list[str]):
            del urls
            return [
                ScrapedPage(
                    requested_url=f"https://example.com/{i}",
                    url=f"https://example.com/{i}",
                    title=f"T{i}",
                    text=f"text {i}",
                    domain="example.com",
                )
                for i in range(3)
            ], 0

    class OutOfOrderDesk:
        def __init__(self) -> None:
            self._n = 0

        async def arun(self, worker, job):
            del worker, job
            idx = self._n
            self._n += 1
            await asyncio.sleep(0.03 * (3 - idx))
            return SimpleNamespace(
                status="completed",
                data=_ExtractionPayload(snippet="s", extracted_text="body", confidence=0.7),
            )

    class Runtime:
        def __init__(self, desk: OutOfOrderDesk) -> None:
            self.settings = SimpleNamespace(model="deepseek/deepseek-v4-flash")
            self.desk = desk

    subagent = SearchSubagent(
        runtime=Runtime(OutOfOrderDesk()),
        search_service=ThreeHitSearch(),
        scrape_service=ThreePageScrape(),
    )
    task = SubagentTask(task_id="task-1", focus="focus", search_queries=["q"], expected_output="out")
    request = ResearchRequest(query="q", max_pages_per_task=3, max_results_per_query=3)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert [item.requested_url for item in evidence] == [
        "https://example.com/0",
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert all(item.extracted_text == "body" for item in evidence)


def test_query_merge_preserves_query_order_under_concurrency() -> None:
    class PerQuerySearch:
        async def search(self, query: str, max_results: int):
            del max_results
            if query == "q1":
                await asyncio.sleep(0.04)
            mapping = {
                "q1": [("https://example.com/a", "A"), ("https://example.com/b", "B")],
                "q2": [("https://example.com/b", "B2"), ("https://example.com/c", "C")],
            }
            return [
                SearchHit(query=query, url=url, title=title, snippet="s")
                for url, title in mapping.get(query, [])
            ]

    subagent = SearchSubagent(
        runtime=FakeRuntime(),
        search_service=PerQuerySearch(),
        scrape_service=EmptyScrapeService(),
    )
    task = SubagentTask(
        task_id="task-1", focus="focus", search_queries=["q1", "q2"], expected_output="out"
    )
    request = ResearchRequest(query="q", max_pages_per_task=5, max_results_per_query=5)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert [item.requested_url for item in evidence] == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


def test_assess_source_quality_class_tiers_order_sensibly() -> None:
    def score(source_class: SourceClass) -> float:
        value, _ = assess_source_quality(source_class, "named", False, False, True)
        return value

    assert (
        score("peer_reviewed")
        > score("journalism")
        > score("corporate")
        > score("personal_blog")
        > score("social_profile")
    )
    assert score("primary") > score("community")


def test_every_source_class_is_priced() -> None:
    for source_class in get_args(SourceClass):
        score, _ = assess_source_quality(source_class, "named", False, False, True)
        assert 0.0 <= score <= 1.0


def test_assess_source_quality_penalties_lower_score() -> None:
    base, base_flags = assess_source_quality("journalism", "named", False, False, True)
    undated, undated_flags = assess_source_quality(
        "journalism", "named", False, False, False
    )
    anon, anon_flags = assess_source_quality(
        "journalism", "anonymous", False, False, True
    )
    promo, promo_flags = assess_source_quality("journalism", "named", True, False, True)

    assert base_flags == []
    assert undated < base and "undated" in undated_flags
    assert anon < base and "no_author" in anon_flags
    assert promo < base and "promotional" in promo_flags


def test_assess_source_quality_clamps() -> None:
    worst, _ = assess_source_quality("social_profile", "anonymous", True, True, False)
    assert worst == 0.0
    best, _ = assess_source_quality("peer_reviewed", "named", False, False, True)
    assert 0.0 <= best <= 1.0


def test_execute_task_populates_source_quality_on_success() -> None:
    desk = _StaticDesk(
        _ExtractionPayload(
            snippet="s",
            extracted_text="body",
            confidence=0.82,
            source_class="journalism",
            authorship="named",
        )
    )
    scrape = _SinglePageScrape(
        ScrapedPage(
            requested_url="https://example.com/a",
            url="https://example.com/a",
            title="Alpha",
            text="page text",
            domain="example.com",
            published_at="2026-01-01",
        )
    )
    subagent = SearchSubagent(
        runtime=_ModelRuntime(desk),
        search_service=AlphaSearch(),
        scrape_service=scrape,
    )
    task = SubagentTask(task_id="t", focus="focus", search_queries=["q"], expected_output="out")
    request = ResearchRequest(query="q", max_pages_per_task=1, max_results_per_query=1)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.source_class == "journalism"
    assert ev.credibility_score is not None and ev.credibility_score > 0.6
    assert ev.relevance_score == 0.82
    assert "snippet_only" not in ev.quality_flags
    assert "unassessed" not in ev.quality_flags


def test_execute_task_fetch_error_is_snippet_only() -> None:
    scrape = _SinglePageScrape(
        ScrapedPage(
            requested_url="https://example.com/a",
            url="https://example.com/a",
            title="Alpha",
            text="",
            domain="example.com",
            fetch_error="timeout",
        )
    )
    subagent = SearchSubagent(
        runtime=FakeRuntime(),
        search_service=AlphaSearch(),
        scrape_service=scrape,
    )
    task = SubagentTask(task_id="t", focus="focus", search_queries=["q"], expected_output="out")
    request = ResearchRequest(query="q", max_pages_per_task=1, max_results_per_query=1)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.source_class is None
    assert ev.quality_flags == ["snippet_only"]
    assert ev.credibility_score == 0.20


def test_execute_task_never_returned_url_is_snippet_only() -> None:
    subagent = SearchSubagent(
        runtime=FakeRuntime(),
        search_service=FakeSearchService(),
        scrape_service=EmptyScrapeService(),
    )
    task = SubagentTask(task_id="t", focus="focus", search_queries=["q"], expected_output="out")
    request = ResearchRequest(query="q", max_pages_per_task=2, max_results_per_query=2)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert evidence
    for ev in evidence:
        assert ev.source_class is None
        assert ev.quality_flags == ["snippet_only"]
        assert ev.credibility_score == 0.20


def test_execute_task_extraction_fallback_is_unassessed() -> None:
    scrape = _SinglePageScrape(
        ScrapedPage(
            requested_url="https://example.com/a",
            url="https://example.com/a",
            title="Alpha",
            text="page text body",
            domain="example.com",
        )
    )
    subagent = SearchSubagent(
        runtime=_ModelRuntime(_RaisingDesk()),
        search_service=AlphaSearch(),
        scrape_service=scrape,
    )
    task = SubagentTask(task_id="t", focus="focus", search_queries=["q"], expected_output="out")
    request = ResearchRequest(query="q", max_pages_per_task=1, max_results_per_query=1)

    evidence = asyncio.run(subagent.execute_task("run:1", task, request))

    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.credibility_score is None
    assert ev.quality_flags == ["unassessed"]
    assert ev.source_class is None
    assert ev.relevance_score == 0.45


def test_extract_completed_trace_carries_credibility() -> None:
    desk = _StaticDesk(
        _ExtractionPayload(
            snippet="s",
            extracted_text="body",
            confidence=0.8,
            source_class="official",
            authorship="organizational",
        )
    )
    scrape = _SinglePageScrape(
        ScrapedPage(
            requested_url="https://example.com/a",
            url="https://example.com/a",
            title="Alpha",
            text="text",
            domain="example.com",
            published_at="2026-01-01",
        )
    )
    subagent = SearchSubagent(
        runtime=_ModelRuntime(desk),
        search_service=AlphaSearch(),
        scrape_service=scrape,
    )
    task = SubagentTask(task_id="t", focus="focus", search_queries=["q"], expected_output="out")
    request = ResearchRequest(query="q", max_pages_per_task=1, max_results_per_query=1)
    captured: dict[str, object] = {}

    async def on_trace(trace_type: str, payload: dict[str, object]) -> None:
        if trace_type == "extract_completed":
            captured.update(payload)

    asyncio.run(subagent.execute_task("run:1", task, request, progress_callback=on_trace))

    assert captured.get("credibility") is not None
