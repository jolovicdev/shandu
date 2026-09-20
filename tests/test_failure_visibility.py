from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from blackgeorge.memory.in_memory import InMemoryMemoryStore

from shandu.agents.lead import LeadAgent
from shandu.agents.search_subagent import SearchSubagent
from shandu.contracts import (
    IterationPlan,
    IterationSynthesis,
    ResearchRequest,
    SubagentTask,
)
from shandu.orchestration.lead_orchestrator import LeadOrchestrator
from shandu.services.memory import MemoryService
from shandu.services.report import ReportService
from shandu.services.scrape import ScrapedPage
from shandu.services.search import SearchHit, SearchService


class _FailedDesk:
    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(status="failed", errors=["401 Unauthorized"], data=None)


class _ModelRuntime:
    def __init__(self, desk: object) -> None:
        self.settings = SimpleNamespace(model="m")
        self.desk = desk


def _request() -> ResearchRequest:
    return ResearchRequest(query="q", max_iterations=1, parallelism=1)


@pytest.mark.parametrize(
    "method",
    ["create_iteration_plan", "synthesize_iteration", "build_final_report"],
)
def test_lead_failure_logs_and_exposes_reason(caplog, method) -> None:
    agent = LeadAgent(runtime=_ModelRuntime(_FailedDesk()))
    with caplog.at_level(logging.WARNING, logger="shandu.agents.lead"):
        if method == "create_iteration_plan":
            plan = asyncio.run(agent.create_iteration_plan(_request(), 0, [], []))
            assert isinstance(plan, IterationPlan)
        elif method == "synthesize_iteration":
            synthesis = asyncio.run(
                agent.synthesize_iteration(_request(), 0, [], [])
            )
            assert isinstance(synthesis, IterationSynthesis)
        else:
            draft = asyncio.run(agent.build_final_report(_request(), [], [], []))
            assert draft.title

    assert any("401" in record.message for record in caplog.records)
    assert agent.last_fallback_reason is not None
    assert "401" in agent.last_fallback_reason


class _AlphaSearch:
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


class _SinglePageScrape:
    def __init__(self, page: ScrapedPage) -> None:
        self._page = page

    async def scrape_many(self, urls):
        del urls
        return [self._page], 0


def test_extractor_failure_logs_and_traces_reason(caplog) -> None:
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
        runtime=_ModelRuntime(_FailedDesk()),
        search_service=_AlphaSearch(),
        scrape_service=scrape,
    )
    task = SubagentTask(
        task_id="t", focus="focus", search_queries=["q"], expected_output="out"
    )
    request = ResearchRequest(
        query="q", max_pages_per_task=1, max_results_per_query=1
    )
    traces: list[tuple[str, dict]] = []

    async def run():
        return await subagent.execute_task(
            "run:1", task, request, progress_callback=lambda *a: traces.append(a)
        )

    with caplog.at_level(logging.WARNING, logger="shandu.agents.search_subagent"):
        evidence = asyncio.run(run())

    assert len(evidence) == 1
    assert any("401" in record.message for record in caplog.records)
    fallbacks = [p for t, p in traces if t == "extraction_fallback"]
    assert len(fallbacks) == 1
    assert "401" in str(fallbacks[0].get("reason", ""))


class _FailingLead:
    fallback_count = 0
    last_fallback_reason: str | None = None

    async def create_iteration_plan(
        self, request, iteration, prior_summaries, memory_context
    ):
        del request, prior_summaries, memory_context
        type(self).fallback_count += 1
        type(self).last_fallback_reason = "planner 401 Unauthorized"
        return IterationPlan(
            iteration_index=iteration,
            goals=["q"],
            subagent_tasks=[],
            continue_loop=False,
        )

    async def synthesize_iteration(
        self, request, iteration, iteration_evidence, prior_summaries
    ):
        raise AssertionError("unreachable")

    async def build_final_report(
        self, request, iteration_summaries, evidence_payload, citations_payload
    ):
        from shandu.contracts import FinalReportDraft

        del request, iteration_summaries, evidence_payload, citations_payload
        return FinalReportDraft(title="t", executive_summary="s")


class _EmptySearchSubagent:
    async def execute_task(
        self, run_scope, task, request, progress_callback=None, extracted_urls=None
    ):
        del run_scope, task, request, progress_callback, extracted_urls
        return []


class _EmptyCitationAgent:
    async def build_citations(self, query, evidence):
        del query, evidence
        return []


def test_orchestrator_carries_fallback_reason_in_event_and_stats() -> None:
    _FailingLead.fallback_count = 0
    _FailingLead.last_fallback_reason = None
    orchestrator = LeadOrchestrator(
        lead_agent=_FailingLead(),
        search_subagent=_EmptySearchSubagent(),
        citation_agent=_EmptyCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=ReportService(),
    )
    events: list = []

    async def run():
        return await orchestrator.run(_request(), progress_callback=events.append)

    result = asyncio.run(run())

    error_events = [e for e in events if e.stage == "error"]
    assert error_events
    assert "401" in str(error_events[0].payload.get("reason", ""))
    assert "fallback_reasons" in result.run_stats
    assert any("401" in str(r) for r in result.run_stats["fallback_reasons"])


class _RaisingClient:
    def text(self, *, query, region, safesearch, max_results, backend):
        del query, region, safesearch, max_results, backend
        raise ConnectionError("network down")


class _EmptyClient:
    def text(self, *, query, region, safesearch, max_results, backend):
        del query, region, safesearch, max_results, backend
        return []


def test_search_all_backends_failed_reports_failure(caplog) -> None:
    service = SearchService()
    service._ddgs = lambda *, timeout: _RaisingClient()

    with caplog.at_level(logging.WARNING, logger="shandu.services.search"):
        hits = asyncio.run(service.search("q", 5))

    assert hits == []
    assert any("backend" in record.message.lower() for record in caplog.records)
    assert service.last_error is not None
    assert "network down" in service.last_error


def test_search_zero_hits_is_not_a_backend_failure() -> None:
    service = SearchService()
    service._ddgs = lambda *, timeout: _EmptyClient()

    hits = asyncio.run(service.search("q", 5))

    assert hits == []
    assert service.last_error is None


class _PlanDesk:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(status="completed", data=self._payload)


def test_planner_success_keeps_short_task_list_unpadded() -> None:
    from shandu.agents.lead import _PlanPayload

    desk = _PlanDesk(
        _PlanPayload(
            goals=["g"],
            subagent_tasks=[
                SubagentTask(
                    task_id="t1",
                    focus="physics question",
                    search_queries=["physics q"],
                    expected_output="out",
                )
            ],
            continue_loop=True,
        )
    )
    agent = LeadAgent(runtime=_ModelRuntime(desk))
    request = ResearchRequest(query="physics q", max_iterations=2, parallelism=3)

    plan = asyncio.run(agent.create_iteration_plan(request, 0, [], []))

    assert [task.task_id for task in plan.subagent_tasks] == ["t1"]


def test_planner_success_caps_overlong_task_list() -> None:
    from shandu.agents.lead import _PlanPayload

    desk = _PlanDesk(
        _PlanPayload(
            goals=["g"],
            subagent_tasks=[
                SubagentTask(
                    task_id=f"t{index}",
                    focus=f"focus {index}",
                    search_queries=[f"q{index}"],
                    expected_output="out",
                )
                for index in range(5)
            ],
            continue_loop=True,
        )
    )
    agent = LeadAgent(runtime=_ModelRuntime(desk))
    request = ResearchRequest(query="q", max_iterations=2, parallelism=2)

    plan = asyncio.run(agent.create_iteration_plan(request, 0, [], []))

    assert [task.task_id for task in plan.subagent_tasks] == ["t0", "t1"]


def test_planner_failure_pads_to_parallelism() -> None:
    agent = LeadAgent(runtime=_ModelRuntime(_FailedDesk()))
    request = ResearchRequest(query="q", max_iterations=2, parallelism=3)

    plan = asyncio.run(agent.create_iteration_plan(request, 0, [], []))

    assert len(plan.subagent_tasks) == 3


def test_normalize_tasks_drops_empty_focus_and_repairs_ids() -> None:
    request = ResearchRequest(query="q", max_iterations=2, parallelism=3)

    tasks = LeadAgent._normalize_tasks(
        [
            SubagentTask(task_id="t1", focus="  ", search_queries=["q"]),
            SubagentTask(task_id="dup", focus="first lane", search_queries=[]),
            SubagentTask(task_id="dup", focus="second lane", search_queries=["q2"]),
        ],
        request,
        0,
    )

    assert [task.focus for task in tasks] == ["first lane", "second lane"]
    assert tasks[0].task_id == "dup"
    assert tasks[0].search_queries == ["first lane"]
    assert tasks[1].task_id != "dup"
    assert tasks[1].task_id.startswith("iter_1_task_")


def test_fallback_tasks_cover_parallelism_with_unique_ids() -> None:
    request = ResearchRequest(query="physics question", max_iterations=2, parallelism=3)

    tasks = LeadAgent._fallback_tasks(request, 1)

    assert len(tasks) == 3
    assert len({task.task_id for task in tasks}) == 3
    assert tasks[0].focus == "physics question"
    assert all(task.search_queries for task in tasks)


def test_extract_title_prefers_h1_then_query() -> None:
    assert (
        LeadAgent._extract_title("# Real Title\n\nBody", "fallback query")
        == "Real Title"
    )
    assert LeadAgent._extract_title("no heading here", "fallback query") == (
        "fallback query"
    )
    assert LeadAgent._extract_title("", "") == "Research Report"


def test_extract_summary_reads_executive_section_then_prose() -> None:
    markdown = (
        "# Title\n\n## Executive Summary\n\nFirst line.\nSecond line.\n\n"
        "## Details\n\nLater."
    )
    assert LeadAgent._extract_summary(markdown) == "First line. Second line."
    assert LeadAgent._extract_summary("# Title\n\nBody prose here.") == (
        "Body prose here."
    )
    assert LeadAgent._extract_summary("# Only Headings\n\n## Sub") == (
        "Summary unavailable."
    )
