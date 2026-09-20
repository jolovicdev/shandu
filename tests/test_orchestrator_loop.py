from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from blackgeorge.memory.in_memory import InMemoryMemoryStore

from shandu.contracts import (
    CitationEntry,
    CoverageAssessment,
    EvidenceRecord,
    IterationPlan,
    IterationSynthesis,
    ResearchRequest,
    SubagentTask,
)
from shandu.orchestration.lead_orchestrator import LeadOrchestrator
from shandu.services.memory import MemoryService
from shandu.services.report import ReportService


class FakeLeadAgent:
    fallback_count = 0

    async def create_iteration_plan(
        self, request, iteration, prior_summaries, memory_context
    ):
        del request, prior_summaries, memory_context
        return IterationPlan(
            iteration_index=iteration,
            goals=[f"goal-{iteration}"],
            subagent_tasks=[
                SubagentTask(
                    task_id=f"task-{iteration}",
                    focus="focus",
                    search_queries=["q"],
                    expected_output="out",
                )
            ],
            continue_loop=True,
        )

    async def synthesize_iteration(
        self, request, iteration, iteration_evidence, prior_summaries
    ):
        del request, iteration_evidence, prior_summaries
        return IterationSynthesis(
            summary=f"summary-{iteration}",
            key_findings=[f"finding-{iteration}"],
            open_questions=[],
            continue_loop=iteration == 0,
            stop_reason="enough evidence" if iteration > 0 else None,
        )

    async def build_final_report(
        self, request, iteration_summaries, evidence_payload, citations_payload
    ):
        del request, evidence_payload, citations_payload
        from shandu.contracts import FinalReportDraft, ReportSection

        return FinalReportDraft(
            title="Synthetic Final",
            executive_summary="done",
            sections=[
                ReportSection(
                    heading="Body",
                    content="\n".join(item.summary for item in iteration_summaries),
                )
            ],
        )


class FakeSearchSubagent:
    async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None):
        del run_scope, request, progress_callback, extracted_urls
        return [
            EvidenceRecord(
                evidence_id=f"e-{task.task_id}",
                task_id=task.task_id,
                query=task.focus,
                requested_url=f"https://example.com/{task.task_id}",
                title=f"Title {task.task_id}",
                snippet="snippet",
                extracted_text="text",
                confidence=0.8,
            )
        ]


class FakeCitationAgent:
    async def build_citations(self, query, evidence):
        del query
        return [
            CitationEntry(
                citation_id=1,
                evidence_ids=[entry.evidence_id for entry in evidence],
                url="https://example.com/ref",
                title="Ref",
                publisher="example.com",
                accessed_at="2026-02-21",
            )
        ]


class ExtraCitationAgent:
    async def build_citations(self, query, evidence):
        del query, evidence
        return [
            CitationEntry(
                citation_id=1,
                evidence_ids=["unused"],
                url="https://example.com/unused",
                title="Unused",
                publisher="example.com",
                accessed_at="2026-02-21",
            ),
            CitationEntry(
                citation_id=2,
                evidence_ids=["used"],
                url="https://example.com/used",
                title="Used",
                publisher="example.com",
                accessed_at="2026-02-21",
            ),
        ]


class FakeReportService(ReportService):
    def render_result(self, request, draft, citations):
        del request
        from shandu.services.report import RenderedReport

        return RenderedReport(
            markdown=f"# {draft.title}\n\n{draft.executive_summary}",
            citations=citations,
        )


class FilteringReportService(ReportService):
    def render_result(self, request, draft, citations):
        del request, draft
        from shandu.services.report import RenderedReport

        return RenderedReport(
            markdown="# Synthetic Final\n\nOnly one citation [1].",
            citations=[citations[1].model_copy(update={"citation_id": 1})],
        )


def test_orchestrator_overrides_model_stop_on_weak_corpus() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    orchestrator = LeadOrchestrator(
        lead_agent=FakeLeadAgent(),
        search_subagent=FakeSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=memory_service,
        report_service=FakeReportService(),
    )

    request = ResearchRequest(query="test", max_iterations=5, parallelism=2)
    result = asyncio.run(orchestrator.run(request))

    assert result.run_stats["iterations"] == 5
    assert result.run_stats["evidence_count"] == 5
    assert result.run_stats["citation_count"] == 1
    assert "Synthetic Final" in result.report_markdown


def test_orchestrator_returns_report_normalized_citation_ledger() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    orchestrator = LeadOrchestrator(
        lead_agent=FakeLeadAgent(),
        search_subagent=FakeSearchSubagent(),
        citation_agent=ExtraCitationAgent(),
        memory_service=memory_service,
        report_service=FilteringReportService(),
    )

    request = ResearchRequest(query="test", max_iterations=1, parallelism=1)
    result = asyncio.run(orchestrator.run(request))

    assert result.run_stats["candidate_citation_count"] == 2
    assert result.run_stats["citation_count"] == 1
    assert [item.citation_id for item in result.citations] == [1]
    assert result.citations[0].title == "Used"


class ParallelLeadAgent(FakeLeadAgent):
    async def create_iteration_plan(
        self, request, iteration, prior_summaries, memory_context
    ):
        del request, prior_summaries, memory_context
        if iteration > 0:
            return IterationPlan(
                iteration_index=iteration,
                goals=[],
                subagent_tasks=[],
                continue_loop=False,
                stop_reason="done",
            )
        return IterationPlan(
            iteration_index=iteration,
            goals=["parallel"],
            subagent_tasks=[
                SubagentTask(
                    task_id="task-1",
                    focus="q1",
                    search_queries=["q1"],
                    expected_output="out",
                ),
                SubagentTask(
                    task_id="task-2",
                    focus="q2",
                    search_queries=["q2"],
                    expected_output="out",
                ),
                SubagentTask(
                    task_id="task-3",
                    focus="q3",
                    search_queries=["q3"],
                    expected_output="out",
                ),
                SubagentTask(
                    task_id="task-4",
                    focus="q4",
                    search_queries=["q4"],
                    expected_output="out",
                ),
            ],
            continue_loop=False,
        )

    async def synthesize_iteration(
        self, request, iteration, iteration_evidence, prior_summaries
    ):
        del request, iteration_evidence, prior_summaries
        return IterationSynthesis(
            summary=f"summary-{iteration}",
            key_findings=[],
            open_questions=[],
            continue_loop=False,
            stop_reason="done",
        )


class SlowSearchSubagent(FakeSearchSubagent):
    async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None):
        del run_scope, request, progress_callback, extracted_urls
        await asyncio.sleep(0.2)
        return [
            EvidenceRecord(
                evidence_id=f"e-{task.task_id}",
                task_id=task.task_id,
                query=task.focus,
                requested_url=f"https://example.com/{task.task_id}",
                title=f"Title {task.task_id}",
                snippet="snippet",
                extracted_text="text",
                confidence=0.8,
            )
        ]


def test_orchestrator_parallelism_controls_task_concurrency() -> None:
    request_serial = ResearchRequest(
        query="parallel-test", max_iterations=1, parallelism=1
    )
    request_parallel = ResearchRequest(
        query="parallel-test", max_iterations=1, parallelism=2
    )

    orchestrator_serial = LeadOrchestrator(
        lead_agent=ParallelLeadAgent(),
        search_subagent=SlowSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    orchestrator_parallel = LeadOrchestrator(
        lead_agent=ParallelLeadAgent(),
        search_subagent=SlowSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )

    started = time.perf_counter()
    asyncio.run(orchestrator_serial.run(request_serial))
    serial_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    asyncio.run(orchestrator_parallel.run(request_parallel))
    parallel_elapsed = time.perf_counter() - started

    assert parallel_elapsed < serial_elapsed * 0.75


def test_orchestrator_emits_task_level_search_progress_events() -> None:
    orchestrator = LeadOrchestrator(
        lead_agent=ParallelLeadAgent(),
        search_subagent=SlowSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    request = ResearchRequest(query="parallel-test", max_iterations=1, parallelism=2)
    events = []

    async def on_event(event):
        events.append(event)

    asyncio.run(orchestrator.run(request, progress_callback=on_event))

    search_messages = [event.message for event in events if event.stage == "search"]
    assert any(message == "Task task-1 started" for message in search_messages)
    assert any(message == "Task task-4 completed" for message in search_messages)


def test_orchestrator_emits_live_model_call_count() -> None:
    orchestrator = LeadOrchestrator(
        lead_agent=FakeLeadAgent(),
        search_subagent=FakeSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    request = ResearchRequest(query="model-call-test", max_iterations=1, parallelism=1)
    events = []

    async def on_event(event):
        events.append(event)

    result = asyncio.run(orchestrator.run(request, progress_callback=on_event))

    live_counts = [
        event.metrics["agent_model_calls"]
        for event in events
        if "agent_model_calls" in event.metrics
    ]
    assert live_counts
    assert live_counts[0] == 1
    assert live_counts[-1] == result.run_stats["agent_model_calls"]


class TraceSearchSubagent(FakeSearchSubagent):
    async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None):
        del run_scope, extracted_urls
        if progress_callback is not None:
            await progress_callback(
                "query_started",
                {"task_id": task.task_id, "query": "q", "max_results": 5},
            )
            await progress_callback(
                "query_completed",
                {"task_id": task.task_id, "query": "q", "hits": 2},
            )
            await progress_callback(
                "scrape_completed",
                {"task_id": task.task_id, "scraped": 1, "missed": 1},
            )
        return await super().execute_task("x", task, request, progress_callback=None)


def test_orchestrator_forwards_subagent_trace_events() -> None:
    orchestrator = LeadOrchestrator(
        lead_agent=ParallelLeadAgent(),
        search_subagent=TraceSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    request = ResearchRequest(query="parallel-test", max_iterations=1, parallelism=2)
    events = []

    async def on_event(event):
        events.append(event)

    asyncio.run(orchestrator.run(request, progress_callback=on_event))

    trace_events = [
        event
        for event in events
        if event.stage == "search" and event.metrics.get("trace_type")
    ]
    assert any(
        event.metrics.get("trace_type") == "query_started" for event in trace_events
    )
    assert any(
        event.metrics.get("trace_type") == "query_completed" for event in trace_events
    )
    assert any(
        event.metrics.get("trace_type") == "scrape_completed" for event in trace_events
    )


def test_run_persists_effective_settings_to_scope() -> None:
    memory_service = MemoryService(InMemoryMemoryStore())
    orchestrator = LeadOrchestrator(
        lead_agent=FakeLeadAgent(),
        search_subagent=FakeSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=memory_service,
        report_service=FakeReportService(),
        runtime_settings={"model": "m", "max_tokens": 4096},
    )
    result = asyncio.run(
        orchestrator.run(
            ResearchRequest(query="settings-test", max_iterations=1, parallelism=1)
        )
    )

    stored = memory_service.read(f"run:{result.run_id}", "settings")
    assert stored == {"model": "m", "max_tokens": 4096}


def test_orchestrator_adds_cost_stats_when_available() -> None:
    lead = FakeLeadAgent()
    lead.last_llm_usage = {
        "prompt_tokens": 2000,
        "completion_tokens": 1000,
        "total_tokens": 3000,
        "cost_usd": 0.02,
        "llm_calls": 2,
        "cost_events": 1,
    }
    orchestrator = LeadOrchestrator(
        lead_agent=lead,
        search_subagent=FakeSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    request = ResearchRequest(query="cost-test", max_iterations=1, parallelism=1)
    result = asyncio.run(orchestrator.run(request))

    assert result.run_stats["agent_model_calls"] == 3
    assert result.run_stats["metered_calls"] == 6
    assert result.run_stats["cost_coverage"] == "full"
    assert result.run_stats["llm_tokens"] == 9000
    assert result.run_stats["usd_spent"] == 0.06


def test_run_stats_include_source_class_and_dated_summary() -> None:
    orchestrator = LeadOrchestrator(
        lead_agent=FakeLeadAgent(),
        search_subagent=FakeSearchSubagent(),
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    request = ResearchRequest(query="quality-test", max_iterations=1, parallelism=1)
    result = asyncio.run(orchestrator.run(request))

    assert "source_class_counts" in result.run_stats
    assert "dated_evidence_fraction" in result.run_stats
    assert result.run_stats["dated_evidence_fraction"] == 0.0


def test_compact_evidence_carries_source_quality_fields() -> None:
    from shandu.agents.lead import LeadAgent

    compact = LeadAgent._compact_evidence(
        [
            {
                "task_id": "t1",
                "query": "q",
                "requested_url": "https://example.com/a",
                "domain": "example.com",
                "title": "Alpha",
                "snippet": "s",
                "extracted_text": "body",
                "confidence": 0.8,
                "published_at": "2026-01-01",
                "source_class": "journalism",
                "credibility_score": 0.72,
                "quality_flags": ["undated"],
            }
        ],
        [],
    )

    entry = compact[0]
    assert entry["domain"] == "example.com"
    assert entry["published_at"] == "2026-01-01"
    assert entry["source_class"] == "journalism"
    assert entry["credibility_score"] == 0.72
    assert entry["quality_flags"] == ["undated"]
    assert entry["citation_id"] is None


def test_compact_evidence_attaches_citation_id_per_record() -> None:
    from shandu.agents.lead import LeadAgent

    compact = LeadAgent._compact_evidence(
        [
            {"evidence_id": "e1", "task_id": "t1"},
            {"evidence_id": "e2", "task_id": "t1"},
            {"evidence_id": "e3", "task_id": "t2"},
        ],
        [
            {"citation_id": 1, "evidence_ids": ["e1", "e2"]},
            {"citation_id": 2, "evidence_ids": ["e9"]},
        ],
    )

    assert [entry["citation_id"] for entry in compact] == [1, 1, None]


def test_compact_evidence_applies_budget_lowest_score_first() -> None:
    import json

    from shandu.agents.lead import LeadAgent, _REPORTER_EVIDENCE_BUDGET

    payload = [
        {
            "evidence_id": f"e{index}",
            "task_id": "t",
            "query": "q",
            "requested_url": f"https://x.example/{index}",
            "title": f"T{index}",
            "snippet": "s",
            "extracted_text": "x" * 2200,
            "confidence": 1.0,
            "credibility_score": (index + 1) / 150,
        }
        for index in range(150)
    ]
    citations = [{"citation_id": 1, "evidence_ids": ["e0"]}]

    compact = LeadAgent._compact_evidence(payload, citations)

    total = sum(
        len(json.dumps(record, ensure_ascii=False, default=str)) for record in compact
    )
    assert total <= _REPORTER_EVIDENCE_BUDGET
    assert len(compact) < 150
    urls = [record["url"] for record in compact]
    assert "https://x.example/0" in urls
    kept_scores = [
        record["credibility_score"]
        for record in compact
        if record["citation_id"] is None
    ]
    dropped_scores = [
        (index + 1) / 150
        for index in range(1, 150)
        if f"https://x.example/{index}" not in urls
    ]
    assert dropped_scores
    assert min(kept_scores) >= max(dropped_scores)
    assert urls == sorted(urls, key=lambda url: int(url.rsplit("/", 1)[1]))


def test_adaptive_loop_weighs_credibility() -> None:
    coverage = SimpleNamespace(
        coverage_score=0.7,
        open_question_severity=0.3,
        contradiction_count=0,
        should_continue=False,
    )
    weak = [
        EvidenceRecord(
            evidence_id=f"e{i}",
            task_id="t",
            query="q",
            requested_url=f"https://d{i}.example/a",
            domain=f"d{i}.example",
            title="T",
            snippet="s",
            extracted_text="x",
            confidence=0.8,
            credibility_score=0.3,
        )
        for i in range(3)
    ]

    assert LeadOrchestrator._adaptive_should_continue(coverage, weak, 3, 0) is True

    unscored = [ev.model_copy(update={"credibility_score": None}) for ev in weak]
    assert LeadOrchestrator._adaptive_should_continue(coverage, unscored, 3, 0) is False


class CapturingSearchSubagent(FakeSearchSubagent):
    def __init__(self) -> None:
        self.seen_calls: list[set[str]] = []

    async def execute_task(
        self, run_scope, task, request, progress_callback=None, extracted_urls=None
    ):
        self.seen_calls.append(set(extracted_urls or ()))
        return await super().execute_task(
            run_scope,
            task,
            request,
            progress_callback=progress_callback,
            extracted_urls=extracted_urls,
        )


def test_orchestrator_passes_already_extracted_urls_per_iteration() -> None:
    subagent = CapturingSearchSubagent()
    orchestrator = LeadOrchestrator(
        lead_agent=FakeLeadAgent(),
        search_subagent=subagent,
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    request = ResearchRequest(query="q", max_iterations=2, parallelism=1)

    asyncio.run(orchestrator.run(request))

    assert subagent.seen_calls[0] == set()
    assert subagent.seen_calls[1] == {"https://example.com/task-0"}


class _StoppingLead(FakeLeadAgent):
    def __init__(self, coverage: CoverageAssessment | None) -> None:
        self._coverage = coverage

    async def synthesize_iteration(
        self, request, iteration, iteration_evidence, prior_summaries
    ):
        del request, iteration, iteration_evidence, prior_summaries
        return IterationSynthesis(
            summary="s",
            key_findings=[],
            open_questions=[],
            continue_loop=False,
            stop_reason="model stop",
            coverage=self._coverage,
        )


class _RichSearchSubagent(FakeSearchSubagent):
    async def execute_task(
        self, run_scope, task, request, progress_callback=None, extracted_urls=None
    ):
        del run_scope, request, progress_callback, extracted_urls
        return [
            EvidenceRecord(
                evidence_id=f"e-{task.task_id}-{index}",
                task_id=task.task_id,
                query=task.focus,
                requested_url=f"https://d{index}.example/a",
                domain=f"d{index}.example",
                title="T",
                snippet="s",
                extracted_text="x",
                confidence=0.8,
            )
            for index in range(3)
        ]


def _run_with(lead, subagent, max_iterations: int):
    orchestrator = LeadOrchestrator(
        lead_agent=lead,
        search_subagent=subagent,
        citation_agent=FakeCitationAgent(),
        memory_service=MemoryService(InMemoryMemoryStore()),
        report_service=FakeReportService(),
    )
    return asyncio.run(
        orchestrator.run(
            ResearchRequest(query="q", max_iterations=max_iterations, parallelism=1)
        )
    )


def test_adaptive_ignores_model_stop_on_weak_coverage() -> None:
    lead = _StoppingLead(
        CoverageAssessment(
            coverage_score=0.1,
            open_question_severity=0.9,
            contradiction_count=0,
            recency_score=0.5,
            should_continue=False,
        )
    )

    result = _run_with(lead, FakeSearchSubagent(), 3)

    assert result.run_stats["iterations"] == 3


def test_adaptive_stops_on_strong_corpus() -> None:
    lead = _StoppingLead(
        CoverageAssessment(
            coverage_score=0.9,
            open_question_severity=0.1,
            contradiction_count=0,
            recency_score=0.9,
            should_continue=False,
        )
    )

    result = _run_with(lead, _RichSearchSubagent(), 5)

    assert result.run_stats["iterations"] == 1


def test_adaptive_without_coverage_uses_evidence_signals() -> None:
    result = _run_with(_StoppingLead(None), FakeSearchSubagent(), 3)

    assert result.run_stats["iterations"] == 3

    strong = [
        EvidenceRecord(
            evidence_id=f"e{i}",
            task_id="t",
            query="q",
            requested_url=f"https://d{i}.example/a",
            domain=f"d{i}.example",
            title="T",
            snippet="s",
            extracted_text="x",
            confidence=0.8,
        )
        for i in range(3)
    ]
    assert LeadOrchestrator._adaptive_should_continue(None, strong, 3, 0) is False


class _PlanStoppingLead(FakeLeadAgent):
    async def create_iteration_plan(
        self, request, iteration, prior_summaries, memory_context
    ):
        del request, prior_summaries, memory_context
        return IterationPlan(
            iteration_index=iteration,
            goals=[f"goal-{iteration}"],
            subagent_tasks=[
                SubagentTask(
                    task_id=f"task-{iteration}",
                    focus="focus",
                    search_queries=["q"],
                    expected_output="out",
                )
            ],
            continue_loop=iteration == 0,
        )


class _CountingSearchSubagent(FakeSearchSubagent):
    def __init__(self) -> None:
        self.calls = 0

    async def execute_task(
        self, run_scope, task, request, progress_callback=None, extracted_urls=None
    ):
        self.calls += 1
        return await super().execute_task(
            run_scope,
            task,
            request,
            progress_callback=progress_callback,
            extracted_urls=extracted_urls,
        )


def test_orchestrator_honors_plan_stop_before_fanout_after_iteration_one() -> None:
    subagent = _CountingSearchSubagent()
    result = _run_with(_PlanStoppingLead(), subagent, 3)

    assert result.run_stats["iterations"] == 1
    assert subagent.calls == 1


class _ContextCapturingLead(FakeLeadAgent):
    def __init__(self) -> None:
        self.contexts: list[list[tuple[str, object]]] = []

    async def create_iteration_plan(
        self, request, iteration, prior_summaries, memory_context
    ):
        self.contexts.append(list(memory_context))
        return await super().create_iteration_plan(
            request, iteration, prior_summaries, memory_context
        )


def test_planner_context_lists_prior_focuses_and_queries_only() -> None:
    import json

    lead = _ContextCapturingLead()
    _run_with(lead, FakeSearchSubagent(), 2)

    assert lead.contexts[0] == [("prior_task_focuses", []), ("queries_run", [])]
    assert lead.contexts[1] == [
        ("prior_task_focuses", ["focus"]),
        ("queries_run", ["q"]),
    ]
    assert "max_iterations" not in json.dumps(lead.contexts[1])
