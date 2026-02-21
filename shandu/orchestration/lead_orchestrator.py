from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Any

from blackgeorge.collaboration import Blackboard, Channel
from blackgeorge.utils import new_id

from ..contracts import (
    EvidenceRecord,
    IterationSynthesis,
    ResearchRequest,
    ResearchRunResult,
    RunEvent,
    SubagentTask,
)
from ..interfaces import (
    CitationAgentLike,
    LeadAgentLike,
    ProgressCallback,
    ReportServiceLike,
    SearchSubagentLike,
)
from ..services.memory import MemoryService


class LeadOrchestrator:
    def __init__(
        self,
        lead_agent: LeadAgentLike,
        search_subagent: SearchSubagentLike,
        citation_agent: CitationAgentLike,
        memory_service: MemoryService,
        report_service: ReportServiceLike,
    ) -> None:
        self._lead = lead_agent
        self._search_subagent = search_subagent
        self._citation = citation_agent
        self._memory = memory_service
        self._report = report_service
        self._channel = Channel()
        self._blackboard = Blackboard()

    async def run(
        self,
        request: ResearchRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        run_id = new_id()
        scope = f"run:{run_id}"
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        event_log: list[dict[str, Any]] = []

        async def emit(event: RunEvent) -> None:
            event_log.append(event.model_dump(mode="json"))
            await self._emit(progress_callback, event)

        self._memory.write(scope, "created_at", started_at, author="orchestrator")
        self._memory.write(scope, "status", "running", author="orchestrator")
        await emit(
            RunEvent(stage="bootstrap", message="Initializing run", metrics={"run_id": run_id}),
        )
        self._memory.write(scope, "request", request.model_dump(mode="json"), author="lead")

        all_evidence: list[EvidenceRecord] = []
        iteration_summaries: list[IterationSynthesis] = []

        for iteration in range(request.max_iterations):
            memory_context = self._memory.search(scope, "iteration")
            plan = await self._lead.create_iteration_plan(
                request=request,
                iteration=iteration,
                prior_summaries=iteration_summaries,
                memory_context=memory_context,
            )
            self._memory.write(
                scope,
                f"iteration:{iteration}:plan",
                plan.model_dump(mode="json"),
                author="lead",
            )
            await emit(
                RunEvent(
                    stage="plan",
                    message=f"Iteration {iteration + 1} plan ready",
                    iteration=iteration,
                    metrics={"tasks": len(plan.subagent_tasks)},
                ),
            )

            if not plan.subagent_tasks:
                break

            semaphore = asyncio.Semaphore(request.parallelism)
            task_total = len(plan.subagent_tasks)
            completed_tasks = 0
            completed_lock = asyncio.Lock()

            async def run_task(task_index: int, task: SubagentTask) -> list[EvidenceRecord]:
                nonlocal completed_tasks
                await emit(
                    RunEvent(
                        stage="search",
                        message=f"Task {task.task_id} started",
                        iteration=iteration,
                        metrics={
                            "task_index": task_index,
                            "task_total": task_total,
                        },
                        payload={
                            "task_id": task.task_id,
                            "focus": task.focus,
                        },
                    ),
                )
                try:
                    async with semaphore:
                        self._channel.send(
                            sender="lead",
                            recipient=task.task_id,
                            content={"focus": task.focus, "queries": task.search_queries},
                        )
                        evidence = await self._search_subagent.execute_task(scope, task, request)
                    self._blackboard.write(
                        key=f"iteration:{iteration}:task:{task.task_id}",
                        value=[item.model_dump(mode="json") for item in evidence],
                        author=task.task_id,
                    )
                    self._memory.write(
                        scope,
                        f"iteration:{iteration}:task:{task.task_id}:evidence_count",
                        len(evidence),
                        author=task.task_id,
                    )
                    async with completed_lock:
                        completed_tasks += 1
                        finished = completed_tasks
                    await emit(
                        RunEvent(
                            stage="search",
                            message=f"Task {task.task_id} completed",
                            iteration=iteration,
                            metrics={
                                "task_index": task_index,
                                "task_total": task_total,
                                "tasks_completed": finished,
                                "evidence": len(evidence),
                            },
                            payload={"task_id": task.task_id},
                        ),
                    )
                    return evidence
                except Exception as exc:
                    await emit(
                        RunEvent(
                            stage="error",
                            message=f"Task {task.task_id} failed",
                            iteration=iteration,
                            payload={"task_id": task.task_id, "error": str(exc)},
                        ),
                    )
                    raise

            results: list[list[EvidenceRecord] | BaseException] = await asyncio.gather(
                *(
                    run_task(index, task)
                    for index, task in enumerate(plan.subagent_tasks, start=1)
                ),
                return_exceptions=True,
            )

            iteration_evidence: list[EvidenceRecord] = []
            task_errors = 0
            for task_result in results:
                if isinstance(task_result, list):
                    iteration_evidence.extend(task_result)
                else:
                    task_errors += 1

            all_evidence.extend(iteration_evidence)
            await emit(
                RunEvent(
                    stage="search",
                    message=f"Iteration {iteration + 1} subagents completed",
                    iteration=iteration,
                    metrics={
                        "tasks": len(plan.subagent_tasks),
                        "parallelism": request.parallelism,
                        "evidence": len(iteration_evidence),
                        "task_errors": task_errors,
                    },
                ),
            )

            synthesis = await self._lead.synthesize_iteration(
                request=request,
                iteration=iteration,
                iteration_evidence=[item.model_dump(mode="json") for item in iteration_evidence],
                prior_summaries=iteration_summaries,
            )
            iteration_summaries.append(synthesis)
            self._memory.write(
                scope,
                f"iteration:{iteration}:synthesis",
                synthesis.model_dump(mode="json"),
                author="lead",
            )
            await emit(
                RunEvent(
                    stage="synthesize",
                    message=f"Iteration {iteration + 1} synthesized",
                    iteration=iteration,
                    metrics={"continue_loop": synthesis.continue_loop},
                    payload={"stop_reason": synthesis.stop_reason or ""},
                ),
            )

            if not plan.continue_loop:
                break
            if not synthesis.continue_loop:
                break
            if not iteration_evidence:
                break

        citations = await self._citation.build_citations(request.query, all_evidence)
        await emit(
            RunEvent(
                stage="cite",
                message="Citation subagent completed",
                metrics={"citations": len(citations)},
            ),
        )

        draft = await self._lead.build_final_report(
            request=request,
            iteration_summaries=iteration_summaries,
            evidence_payload=[item.model_dump(mode="json") for item in all_evidence],
            citations_payload=[entry.model_dump(mode="json") for entry in citations],
        )
        report_markdown = self._report.render(request, draft, citations)
        await emit(
            RunEvent(
                stage="report",
                message="Lead researcher completed final report draft",
                metrics={"report_words": len(report_markdown.split())},
            ),
        )

        elapsed = time.perf_counter() - started
        result = ResearchRunResult(
            run_id=run_id,
            request=request,
            report_markdown=report_markdown,
            citations=citations,
            evidence=all_evidence,
            iteration_summaries=iteration_summaries,
            run_stats={
                "elapsed_seconds": round(elapsed, 2),
                "iterations": len(iteration_summaries),
                "evidence_count": len(all_evidence),
                "citation_count": len(citations),
            },
        )

        await emit(
            RunEvent(
                stage="complete",
                message="Run completed",
                metrics=result.run_stats,
                payload={"run_id": run_id},
            ),
        )
        self._memory.write(scope, "status", "completed", author="orchestrator")
        self._memory.write(
            scope,
            "updated_at",
            datetime.now(timezone.utc).isoformat(),
            author="orchestrator",
        )
        self._memory.write(scope, "events", event_log, author="orchestrator")
        self._memory.write(
            scope,
            "result",
            {
                "run_id": result.run_id,
                "run_stats": result.run_stats,
                "report_preview": result.report_markdown[:1800],
                "citation_count": len(result.citations),
                "evidence_count": len(result.evidence),
            },
            author="orchestrator",
        )

        return result

    async def _emit(
        self,
        callback: ProgressCallback | None,
        event: RunEvent,
    ) -> None:
        if callback is None:
            return
        result = callback(event)
        if isinstance(result, Awaitable):
            await result
