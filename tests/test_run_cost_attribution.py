from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from blackgeorge.core.event import Event
from blackgeorge.memory.in_memory import InMemoryMemoryStore

from shandu.agents.lead import LeadAgent, _PlanPayload, _SynthesisPayload
from shandu.agents.search_subagent import SearchSubagent, _ExtractionPayload
from shandu.contracts import ResearchRequest, SubagentTask
from shandu.orchestration.lead_orchestrator import LeadOrchestrator
from shandu.runtime.bootstrap import RuntimeBootstrap, RuntimeSettings
from shandu.services.ai_search import AISearchService
from shandu.services.memory import MemoryService
from shandu.services.report import ReportService
from shandu.services.scrape import ScrapedPage
from shandu.services.search import SearchHit

_USAGE = {
    "prompt_tokens": 120,
    "completion_tokens": 40,
    "total_tokens": 160,
    "cost_usd": 0.0125,
}


_METRICS = {
    "cost_usd": _USAGE["cost_usd"],
    "usage": {
        "prompt_tokens": _USAGE["prompt_tokens"],
        "completion_tokens": _USAGE["completion_tokens"],
        "total_tokens": _USAGE["total_tokens"],
    },
}


class _MeteredRuntime:
    def __init__(self, desk: object) -> None:
        self.settings = SimpleNamespace(model="m")
        self.desk = desk


class _LeadDesk:
    async def arun(self, worker, job):
        del worker
        schema = getattr(job, "response_schema", None)
        if schema is _PlanPayload:
            data = _PlanPayload(
                goals=["g"], subagent_tasks=[], continue_loop=False, stop_reason=None
            )
            content = None
        elif schema is _SynthesisPayload:
            data = _SynthesisPayload(
                summary="s",
                key_findings=[],
                open_questions=[],
                continue_loop=False,
                stop_reason=None,
                coverage_score=1.0,
                open_question_severity=0.0,
                contradiction_count=0,
                recency_score=1.0,
                coverage_should_continue=False,
            )
            content = None
        else:
            data, content = None, "# Title\n\nBody text here."
        return SimpleNamespace(
            status="completed",
            data=data,
            content=content,
            run_id="bg-1",
            errors=[],
            metrics=dict(_METRICS),
        )


class _EmptySearchSubagent:
    async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None):
        del run_scope, task, request, progress_callback, extracted_urls
        return []


class _EmptyCitationAgent:
    async def build_citations(self, query, evidence):
        del query, evidence
        return []


def test_lead_usage_reaches_completion_events_and_run_stats() -> None:
    orchestrator = LeadOrchestrator(
        lead_agent=LeadAgent(runtime=_MeteredRuntime(_LeadDesk())),
        search_subagent=_EmptySearchSubagent(),
        citation_agent=_EmptyCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=ReportService(),
    )
    events: list = []

    async def run():
        return await orchestrator.run(
            ResearchRequest(query="q", max_iterations=1, parallelism=1),
            progress_callback=events.append,
        )

    result = asyncio.run(run())

    plan_events = [e for e in events if e.message == "Iteration 1 plan ready"]
    assert len(plan_events) == 1
    assert plan_events[0].metrics.get("llm_usage") == _USAGE
    usage = result.run_stats.get("llm_usage")
    assert usage is not None
    assert usage["prompt_tokens"] == 240
    assert usage["completion_tokens"] == 80
    assert usage["total_tokens"] == 320
    assert usage["cost_usd"] == pytest.approx(0.025)


class _ExtractionDesk:
    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(
            status="completed",
            data=_ExtractionPayload(snippet="s", extracted_text="t" * 60),
            run_id="bg-1",
            errors=[],
            metrics=dict(_METRICS),
        )


class _SingleHitSearch:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del query, max_results
        return [
            SearchHit(
                query="q", url="https://example.com/a", title="A", snippet="s"
            )
        ]


class _SinglePageScrape:
    async def scrape_many(self, urls):
        pages = [
            ScrapedPage(
                requested_url="https://example.com/a",
                url="https://example.com/a",
                title="A",
                text="page text body " * 20,
                domain="example.com",
            )
        ]
        del urls
        return pages, 0


def test_extraction_usage_reaches_trace_payload() -> None:
    subagent = SearchSubagent(
        runtime=_MeteredRuntime(_ExtractionDesk()),
        search_service=_SingleHitSearch(),
        scrape_service=_SinglePageScrape(),
    )
    task = SubagentTask(
        task_id="t", focus="f", search_queries=["q"], expected_output="o"
    )
    request = ResearchRequest(query="q", max_pages_per_task=1, max_results_per_query=1)
    traces: list[tuple[str, dict]] = []

    async def run():
        return await subagent.execute_task(
            "run:1", task, request, progress_callback=lambda *a: traces.append(a)
        )

    asyncio.run(run())

    completed = [p for t, p in traces if t == "extract_completed"]
    assert len(completed) == 1
    assert completed[0].get("llm_usage") == _USAGE


class _AnswerDesk:
    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(
            status="completed",
            content="# Answer\n\nBody [1]",
            run_id="bg-1",
            errors=[],
            metrics=dict(_METRICS),
        )


def test_ai_search_usage_reaches_run_stats() -> None:
    service = AISearchService(
        runtime=_MeteredRuntime(_AnswerDesk()),
        search_service=_SingleHitSearch(),
        scrape_service=_SinglePageScrape(),
    )

    result = asyncio.run(service.search("q"))

    assert result.run_stats.get("llm_usage") == _USAGE


def test_collect_reads_usage_from_report_metrics() -> None:
    from shandu.runtime.costing import collect_llm_usage

    assert collect_llm_usage(SimpleNamespace(status="completed", data=None)) is None
    report = SimpleNamespace(status="completed", run_id="bg-1", metrics=dict(_METRICS))
    assert collect_llm_usage(report) == _USAGE


class _UsageLead:
    fallback_count = 0
    last_fallback_reason = None

    def __init__(self) -> None:
        self.usage: dict | None = None

    @property
    def last_llm_usage(self) -> dict | None:
        return self.usage

    async def create_iteration_plan(
        self, request, iteration, prior_summaries, memory_context
    ):
        from shandu.contracts import IterationPlan, SubagentTask

        del request, prior_summaries, memory_context
        return IterationPlan(
            iteration_index=iteration,
            goals=["q"],
            subagent_tasks=[
                SubagentTask(
                    task_id="t",
                    focus="f",
                    search_queries=["q"],
                    expected_output="o",
                )
            ],
            continue_loop=False,
        )

    async def synthesize_iteration(
        self, request, iteration, iteration_evidence, prior_summaries
    ):
        from shandu.contracts import IterationSynthesis

        del request, iteration, iteration_evidence, prior_summaries
        return IterationSynthesis(summary="s")

    async def build_final_report(
        self, request, iteration_summaries, evidence_payload, citations_payload
    ):
        from shandu.contracts import FinalReportDraft

        del request, iteration_summaries, evidence_payload, citations_payload
        return FinalReportDraft(title="t", executive_summary="s")


def test_sequential_runs_report_independent_costs() -> None:
    lead = _UsageLead()
    orchestrator = LeadOrchestrator(
        lead_agent=lead,
        search_subagent=_EmptySearchSubagent(),
        citation_agent=_EmptyCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=ReportService(),
    )

    async def run_once():
        return await orchestrator.run(
            ResearchRequest(query="q", max_iterations=1, parallelism=1)
        )

    lead.usage = {
        "prompt_tokens": 60,
        "completion_tokens": 40,
        "total_tokens": 100,
        "cost_usd": 0.01,
    }
    first = asyncio.run(run_once())
    lead.usage = {
        "prompt_tokens": 120,
        "completion_tokens": 80,
        "total_tokens": 200,
        "cost_usd": 0.02,
    }
    second = asyncio.run(run_once())

    assert first.run_stats["metered_calls"] == 3
    assert first.run_stats["llm_tokens"] == 300
    assert first.run_stats["usd_spent"] == pytest.approx(0.03)
    assert second.run_stats["metered_calls"] == 3
    assert second.run_stats["llm_tokens"] == 600
    assert second.run_stats["usd_spent"] == pytest.approx(0.06)


def test_overlapping_runs_report_independent_costs() -> None:
    def make_orchestrator(total_tokens: int, cost_usd: float) -> LeadOrchestrator:
        lead = _UsageLead()
        lead.usage = {
            "prompt_tokens": total_tokens * 6 // 10,
            "completion_tokens": total_tokens * 4 // 10,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }
        return LeadOrchestrator(
            lead_agent=lead,
            search_subagent=_EmptySearchSubagent(),
            citation_agent=_EmptyCitationAgent(),
            memory_service=MemoryService(InMemoryMemoryStore()),
            report_service=ReportService(),
        )

    first_orchestrator = make_orchestrator(100, 0.01)
    second_orchestrator = make_orchestrator(200, 0.02)

    async def main():
        return await asyncio.gather(
            first_orchestrator.run(
                ResearchRequest(query="a", max_iterations=1, parallelism=1)
            ),
            second_orchestrator.run(
                ResearchRequest(query="b", max_iterations=1, parallelism=1)
            ),
        )

    first, second = asyncio.run(main())

    assert first.run_stats["metered_calls"] == 3
    assert first.run_stats["llm_tokens"] == 300
    assert first.run_stats["usd_spent"] == pytest.approx(0.03)
    assert second.run_stats["metered_calls"] == 3
    assert second.run_stats["llm_tokens"] == 600
    assert second.run_stats["usd_spent"] == pytest.approx(0.06)


def test_inspect_run_aggregates_usage(tmp_path) -> None:
    settings = RuntimeSettings(
        model="m",
        temperature=0.2,
        max_tokens=100,
        storage_dir=str(tmp_path),
        structured_output_retries=1,
        max_iterations=2,
        max_tool_calls=4,
        num_retries=0,
        max_context_messages=10,
    )
    bootstrap = RuntimeBootstrap(settings)
    try:
        bootstrap.desk.run_store.create_run("run-1", {"job": "x"})
        bootstrap.desk.run_store.add_event(
            Event(
                event_id="e1",
                type="llm.completed",
                timestamp=datetime.now(timezone.utc),
                run_id="run-1",
                source="test",
                payload={
                    "cost": 0.01,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            )
        )
        out = bootstrap.inspect_run("run-1")
    finally:
        bootstrap.close()

    assert out["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost_usd": 0.01,
        "llm_calls": 1,
        "cost_events": 1,
    }
