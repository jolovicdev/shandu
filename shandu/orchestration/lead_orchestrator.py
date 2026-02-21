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
from ..runtime.cost_tracker import CostTracker, CostSnapshot


class LeadOrchestrator:
    def __init__(
        self,
        lead_agent: LeadAgentLike,
        search_subagent: SearchSubagentLike,
        citation_agent: CitationAgentLike,
        memory_service: MemoryService,
        report_service: ReportServiceLike,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._lead = lead_agent
        self._search_subagent = search_subagent
        self._citation = citation_agent
        self._memory = memory_service
        self._report = report_service
        self._cost_tracker = cost_tracker
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
        cost_start = self._cost_tracker.snapshot() if self._cost_tracker is not None else None

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
        agent_model_calls = 0

        for iteration in range(request.max_iterations):
            memory_context = self._memory.search(scope, "iteration")
            agent_model_calls += 1
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

                async def on_search_trace(
                    trace_type: str,
                    payload: dict[str, Any],
                ) -> None:
                    nonlocal agent_model_calls
                    if trace_type == "extract_started":
                        agent_model_calls += 1
                    await emit(
                        self._build_search_trace_event(
                            iteration=iteration,
                            trace_type=trace_type,
                            payload=payload,
                        )
                    )

                try:
                    async with semaphore:
                        self._channel.send(
                            sender="lead",
                            recipient=task.task_id,
                            content={"focus": task.focus, "queries": task.search_queries},
                        )
                        evidence = await self._search_subagent.execute_task(
                            scope,
                            task,
                            request,
                            progress_callback=on_search_trace,
                        )
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

            agent_model_calls += 1
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

        agent_model_calls += 1
        citations = await self._citation.build_citations(request.query, all_evidence)
        await emit(
            RunEvent(
                stage="cite",
                message="Citation subagent completed",
                metrics={"citations": len(citations)},
            ),
        )

        agent_model_calls += 1
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
        run_stats: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 2),
            "iterations": len(iteration_summaries),
            "evidence_count": len(all_evidence),
            "citation_count": len(citations),
            "agent_model_calls": agent_model_calls,
        }
        self._append_cost_stats(run_stats, cost_start)

        result = ResearchRunResult(
            run_id=run_id,
            request=request,
            report_markdown=report_markdown,
            citations=citations,
            evidence=all_evidence,
            iteration_summaries=iteration_summaries,
            run_stats=run_stats,
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

    def _append_cost_stats(
        self,
        run_stats: dict[str, Any],
        baseline: CostSnapshot | None,
    ) -> None:
        if self._cost_tracker is None or baseline is None:
            return
        delta = self._cost_tracker.delta_since(baseline)
        model_calls = run_stats.get("agent_model_calls")
        if delta.llm_calls > 0:
            run_stats["metered_calls"] = delta.llm_calls
        if delta.total_tokens > 0:
            run_stats["llm_tokens"] = delta.total_tokens
        if delta.cost_events > 0:
            run_stats["usd_spent"] = round(delta.total_cost_usd, 6)
        if isinstance(model_calls, int) and model_calls > 0 and delta.llm_calls > 0:
            if delta.llm_calls < model_calls:
                run_stats["cost_coverage"] = "partial"
            else:
                run_stats["cost_coverage"] = "full"

    def _build_search_trace_event(
        self,
        *,
        iteration: int,
        trace_type: str,
        payload: dict[str, Any],
    ) -> RunEvent:
        task_id = str(payload.get("task_id", ""))
        metrics: dict[str, Any] = {"trace_type": trace_type}
        message = f"Task {task_id} update" if task_id else "Subagent update"

        if trace_type == "query_started":
            query = str(payload.get("query", "")).strip()
            message = f"Task {task_id} searching query" if task_id else "Searching query"
            if query:
                metrics["query"] = query
            if "max_results" in payload:
                metrics["max_results"] = payload["max_results"]
        elif trace_type == "query_completed":
            query = str(payload.get("query", "")).strip()
            message = f"Task {task_id} query completed" if task_id else "Query completed"
            if query:
                metrics["query"] = query
            if "hits" in payload:
                metrics["hits"] = payload["hits"]
        elif trace_type == "scrape_started":
            message = f"Task {task_id} scraping pages" if task_id else "Scraping pages"
            if "url_count" in payload:
                metrics["url_count"] = payload["url_count"]
        elif trace_type == "scrape_completed":
            message = f"Task {task_id} scrape completed" if task_id else "Scrape completed"
            if "scraped" in payload:
                metrics["scraped"] = payload["scraped"]
            if "missed" in payload:
                metrics["missed"] = payload["missed"]
        elif trace_type == "extract_completed":
            message = f"Task {task_id} extracted page" if task_id else "Extracted page"
            if "confidence" in payload:
                metrics["confidence"] = payload["confidence"]
        elif trace_type == "extract_started":
            message = f"Task {task_id} extracting page" if task_id else "Extracting page"
        elif trace_type == "fallback_evidence":
            message = f"Task {task_id} fallback evidence added" if task_id else "Fallback evidence added"
            if "confidence" in payload:
                metrics["confidence"] = payload["confidence"]

        return RunEvent(
            stage="search",
            message=message,
            iteration=iteration,
            metrics=metrics,
            payload=payload,
        )
