from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

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
            data=_ExtractionPayload(
                snippet="s",
                extracted_text="t",
                confidence=0.5,
                source_class="unknown",
                authorship="anonymous",
                is_promotional=False,
                summarizes_inaccessible_source=False,
                claims=[],
            ),
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


def test_pipeline_claims_its_pages_and_ranks_claimed_pages_last() -> None:
    search = _RecordingSearch(["https://example.org/a", "https://example.org/b"])
    scrape = _EchoScrape()
    shared = {"https://example.org/a"}

    asyncio.run(
        _subagent(search, scrape).execute_task(
            "run:r", _task(), ResearchRequest(query="q", max_pages_per_task=1), extracted_urls=shared
        )
    )

    assert scrape.requested == ["https://example.org/b"]
    assert shared == {"https://example.org/a", "https://example.org/b"}


def test_pipeline_skips_a_page_a_sibling_claims_while_it_ranks() -> None:
    from shandu.agents.search_subagent import _SelectionPayload

    shared: set[str] = set()

    class _SiblingClaimsDuringRanking:
        async def arun(self, worker, job):
            if worker.name == "SubagentSelector":
                shared.add("https://example.org/a")
                return SimpleNamespace(
                    status="completed", data=_SelectionPayload(ranked_ids=[0, 1, 2]), metrics={}
                )
            return await _ExtractDesk().arun(worker, job)

    search = _RecordingSearch(
        ["https://example.org/a", "https://example.org/b", "https://example.org/c"]
    )
    scrape = _EchoScrape()
    runtime = SimpleNamespace(settings=SimpleNamespace(model="m"), desk=_SiblingClaimsDuringRanking())

    asyncio.run(
        SearchSubagent(runtime, search, scrape).execute_task(
            "run:r", _task(), ResearchRequest(query="q", max_pages_per_task=1), extracted_urls=shared
        )
    )

    assert scrape.requested == ["https://example.org/b"]


def test_domain_scoping_keeps_every_planned_query_under_the_search_cap() -> None:
    web = _RecordingSearch([])
    papers = _RecordingSearch([])
    runtime = SimpleNamespace(settings=SimpleNamespace(model="m"), desk=_ExtractDesk())
    task = SubagentTask(
        task_id="t1", focus="f", search_queries=["q1", "q2", "q3", "q4"], scholarly=True
    )
    request = ResearchRequest(
        query="q", include_domains=["a.org", "b.org", "c.org"], max_searches_per_task=6
    )

    asyncio.run(SearchSubagent(runtime, web, _EchoScrape(), papers).execute_task("run:r", task, request))

    assert web.queries == [
        "q1 site:a.org",
        "q2 site:a.org",
        "q3 site:a.org",
        "q4 site:a.org",
        "q1 site:b.org",
        "q2 site:b.org",
    ]
    assert papers.queries == ["q1", "q2", "q3", "q4"]


def test_pipeline_never_reads_a_page_already_read_or_claimed() -> None:
    search = _RecordingSearch(
        ["https://example.org/a", "https://example.org/b", "https://example.org/c"]
    )
    scrape = _EchoScrape()
    shared = {"https://example.org/a", "https://example.org/b"}

    asyncio.run(
        _subagent(search, scrape).execute_task(
            "run:r", _task(), ResearchRequest(query="q", max_pages_per_task=3), extracted_urls=shared
        )
    )

    assert scrape.requested == ["https://example.org/c"]


def test_domain_filters_ignore_ports_and_schemes() -> None:
    request = ResearchRequest(
        query="q",
        include_domains=["https://Docs.Example.com:443/guide", "b.org:8080"],
        exclude_domains=["www.Bad.org:80"],
    )

    assert request.include_domains == ["docs.example.com", "b.org"]
    assert request.exclude_domains == ["bad.org"]


def test_domain_filters_split_typed_lists_and_drop_wildcards() -> None:
    request = ResearchRequest(
        query="q", include_domains=["*.nature.com", "who.int, cdc.gov; nih.gov  who.int"]
    )

    assert request.include_domains == ["nature.com", "who.int", "cdc.gov", "nih.gov"]


@pytest.mark.parametrize("junk", ["nature", "nature..com", "-bad.org", "who.int/news site"])
def test_domain_filters_refuse_what_is_not_a_site_name(junk: str) -> None:
    with pytest.raises(ValidationError, match="is not a site name"):
        ResearchRequest(query="q", exclude_domains=[junk])


def test_a_named_effort_fills_the_budgets_a_request_leaves_unset() -> None:
    request = ResearchRequest(query="q", effort="high", max_iterations=1)

    assert (request.max_iterations, request.parallelism, request.max_pages_per_task) == (1, 4, 6)
    assert request.max_searches_per_task == 8


@pytest.mark.parametrize(
    "fields",
    [
        {"query": "  \n"},
        {"query": "q", "max_cost_usd": float("inf")},
        {"query": "q", "effort": ["low"]},
    ],
    ids=["blank question", "endless budget", "listed effort"],
)
def test_a_request_refuses_what_no_run_can_use(fields: dict) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(**fields)


def test_a_page_title_in_capitals_takes_the_search_hits_casing() -> None:
    hit_titles = {
        "https://arxiv.org/abs/2309.17453": "Efficient Streaming Language Models with Attention Sinks",
        "https://example.org/b": "Another page",
    }
    page_titles = {
        "https://arxiv.org/abs/2309.17453": "EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS",
        "https://example.org/b": "A DIFFERENT TITLE",
    }

    class Search:
        async def search(self, query: str, max_results: int) -> list[SearchHit]:
            return [
                SearchHit(query=query, url=url, title=title, snippet="s")
                for url, title in hit_titles.items()
            ]

    class Scrape:
        async def scrape_many(self, urls):
            return [
                ScrapedPage(requested_url=url, url=url, title=page_titles[url], text="page text", domain="x.org")
                for url in urls
            ], 0

    request = ResearchRequest(query="q", max_pages_per_task=2, max_results_per_query=2)
    evidence = asyncio.run(_subagent(Search(), Scrape()).execute_task("run:1", _task(), request))

    assert {record.requested_url: record.title for record in evidence} == {
        "https://arxiv.org/abs/2309.17453": "Efficient Streaming Language Models with Attention Sinks",
        "https://example.org/b": "A DIFFERENT TITLE",
    }
