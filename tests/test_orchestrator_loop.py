from __future__ import annotations

import asyncio
import time

from blackgeorge.memory.in_memory import InMemoryMemoryStore

from shandu.contracts import (
    CitationEntry,
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
    async def create_iteration_plan(self, request, iteration, prior_summaries, memory_context):
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

    async def synthesize_iteration(self, request, iteration, iteration_evidence, prior_summaries):
        del request, iteration_evidence, prior_summaries
        return IterationSynthesis(
            summary=f"summary-{iteration}",
            key_findings=[f"finding-{iteration}"],
            open_questions=[],
            continue_loop=iteration == 0,
            stop_reason="enough evidence" if iteration > 0 else None,
        )

    async def build_final_report(self, request, iteration_summaries, evidence_payload, citations_payload):
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
    async def execute_task(self, run_scope, task, request):
        del run_scope, request
        return [
            EvidenceRecord(
                evidence_id=f"e-{task.task_id}",
                task_id=task.task_id,
                query=task.focus,
                url=f"https://example.com/{task.task_id}",
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


class FakeReportService(ReportService):
    def render(self, request, draft, citations):
        del request, citations
        return f"# {draft.title}\n\n{draft.executive_summary}"


def test_orchestrator_stops_on_synthesis_decision() -> None:
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

    assert result.run_stats["iterations"] == 2
    assert result.run_stats["evidence_count"] == 2
    assert result.run_stats["citation_count"] == 1
    assert "Synthetic Final" in result.report_markdown


class ParallelLeadAgent(FakeLeadAgent):
    async def create_iteration_plan(self, request, iteration, prior_summaries, memory_context):
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
                SubagentTask(task_id="task-1", focus="q1", search_queries=["q1"], expected_output="out"),
                SubagentTask(task_id="task-2", focus="q2", search_queries=["q2"], expected_output="out"),
                SubagentTask(task_id="task-3", focus="q3", search_queries=["q3"], expected_output="out"),
                SubagentTask(task_id="task-4", focus="q4", search_queries=["q4"], expected_output="out"),
            ],
            continue_loop=False,
        )

    async def synthesize_iteration(self, request, iteration, iteration_evidence, prior_summaries):
        del request, iteration_evidence, prior_summaries
        return IterationSynthesis(
            summary=f"summary-{iteration}",
            key_findings=[],
            open_questions=[],
            continue_loop=False,
            stop_reason="done",
        )


class SlowSearchSubagent(FakeSearchSubagent):
    async def execute_task(self, run_scope, task, request):
        del run_scope, request
        await asyncio.sleep(0.2)
        return [
            EvidenceRecord(
                evidence_id=f"e-{task.task_id}",
                task_id=task.task_id,
                query=task.focus,
                url=f"https://example.com/{task.task_id}",
                title=f"Title {task.task_id}",
                snippet="snippet",
                extracted_text="text",
                confidence=0.8,
            )
        ]


def test_orchestrator_parallelism_controls_task_concurrency() -> None:
    request_serial = ResearchRequest(query="parallel-test", max_iterations=1, parallelism=1)
    request_parallel = ResearchRequest(query="parallel-test", max_iterations=1, parallelism=2)

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
