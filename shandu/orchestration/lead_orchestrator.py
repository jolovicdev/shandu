from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Any, Literal

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
            RunEvent(
                stage="bootstrap",
                message="Initializing run",
                metrics={"run_id": run_id},
            ),
        )
        self._memory.write(
            scope, "request", request.model_dump(mode="json"), author="lead"
        )

        agent_model_calls = 0
        all_evidence: list[EvidenceRecord] = []
        iteration_summaries: list[IterationSynthesis] = []
        lead_fallbacks = self._lead.fallback_count
        extraction_fallbacks = 0
        fallback_reasons: list[str] = []
        llm_totals: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "llm_calls": 0,
            "cost_events": 0,
        }
        saw_llm_usage = False

        def absorb_llm_usage(usage: Any) -> dict[str, Any] | None:
            nonlocal saw_llm_usage
            if not isinstance(usage, dict) or not usage:
                return None
            saw_llm_usage = True
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key, 0)
                if isinstance(value, (int, float)):
                    llm_totals[key] += value
            cost = usage.get("cost_usd", 0)
            if isinstance(cost, (int, float)):
                llm_totals["cost_usd"] += cost
            calls = usage.get("llm_calls", 1)
            if isinstance(calls, int):
                llm_totals["llm_calls"] += calls
            cost_hits = usage.get("cost_events", 1 if "cost_usd" in usage else 0)
            if isinstance(cost_hits, int):
                llm_totals["cost_events"] += cost_hits
            return usage

        def with_model_call_count(
            metrics: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            updated = dict(metrics or {})
            if agent_model_calls > 0:
                updated["agent_model_calls"] = agent_model_calls
            return updated

        for iteration in range(request.max_iterations):
            memory_context = self._memory.search(scope, "iteration")
            agent_model_calls += 1
            plan = await self._lead.create_iteration_plan(
                request=request,
                iteration=iteration,
                prior_summaries=iteration_summaries,
                memory_context=memory_context,
            )
            plan_usage = absorb_llm_usage(
                getattr(self._lead, "last_llm_usage", None)
            )
            if self._lead.fallback_count > lead_fallbacks:
                lead_fallbacks = self._lead.fallback_count
                await emit(
                    RunEvent(
                        stage="error",
                        message="Lead planner fell back to deterministic plan",
                        iteration=iteration,
                        payload=self._fallback_payload("create_iteration_plan"),
                    ),
                )
                self._record_fallback_reason(
                    fallback_reasons, "create_iteration_plan"
                )
            self._memory.write(
                scope,
                f"iteration:{iteration}:plan",
                plan.model_dump(mode="json"),
                author="lead",
            )
            plan_metrics = with_model_call_count(
                {"tasks": len(plan.subagent_tasks)}
            )
            if plan_usage:
                plan_metrics["llm_usage"] = plan_usage
            await emit(
                RunEvent(
                    stage="plan",
                    message=f"Iteration {iteration + 1} plan ready",
                    iteration=iteration,
                    metrics=plan_metrics,
                ),
            )

            if not plan.subagent_tasks:
                break

            semaphore = asyncio.Semaphore(request.parallelism)
            task_total = len(plan.subagent_tasks)
            completed_tasks = 0
            completed_lock = asyncio.Lock()

            async def run_task(
                task_index: int, task: SubagentTask
            ) -> list[EvidenceRecord]:
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
                    nonlocal agent_model_calls, extraction_fallbacks
                    if trace_type == "extract_started":
                        agent_model_calls += 1
                    elif trace_type == "extraction_fallback":
                        extraction_fallbacks += 1
                        trace_reason = payload.get("reason")
                        if trace_reason:
                            fallback_reasons.append(str(trace_reason)[:300])
                    trace_usage = absorb_llm_usage(payload.get("llm_usage"))
                    trace_event = self._build_search_trace_event(
                        iteration=iteration,
                        trace_type=trace_type,
                        payload=payload,
                    )
                    trace_event.metrics = with_model_call_count(trace_event.metrics)
                    if trace_usage:
                        trace_event.metrics["llm_usage"] = trace_usage
                    await emit(trace_event)

                try:
                    async with semaphore:
                        self._channel.send(
                            sender="lead",
                            recipient=task.task_id,
                            content={
                                "focus": task.focus,
                                "queries": task.search_queries,
                            },
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
                iteration_evidence=[
                    item.model_dump(mode="json") for item in iteration_evidence
                ],
                prior_summaries=iteration_summaries,
            )
            synthesis_usage = absorb_llm_usage(
                getattr(self._lead, "last_llm_usage", None)
            )
            if self._lead.fallback_count > lead_fallbacks:
                lead_fallbacks = self._lead.fallback_count
                await emit(
                    RunEvent(
                        stage="error",
                        message="Lead synthesizer fell back to deterministic synthesis",
                        iteration=iteration,
                        payload=self._fallback_payload("synthesize_iteration"),
                    ),
                )
                self._record_fallback_reason(
                    fallback_reasons, "synthesize_iteration"
                )
            iteration_summaries.append(synthesis)
            self._memory.write(
                scope,
                f"iteration:{iteration}:synthesis",
                synthesis.model_dump(mode="json"),
                author="lead",
            )
            synthesis_metrics = with_model_call_count(
                {
                    "continue_loop": synthesis.continue_loop,
                    "coverage_score": synthesis.coverage.coverage_score
                    if synthesis.coverage
                    else None,
                    "depth_policy": request.depth_policy,
                }
            )
            if synthesis_usage:
                synthesis_metrics["llm_usage"] = synthesis_usage
            await emit(
                RunEvent(
                    stage="synthesize",
                    message=f"Iteration {iteration + 1} synthesized",
                    iteration=iteration,
                    metrics=synthesis_metrics,
                    payload={"stop_reason": synthesis.stop_reason or ""},
                ),
            )

            if not plan.continue_loop:
                break
            if not iteration_evidence:
                break

            if request.depth_policy == "adaptive" and synthesis.coverage is not None:
                if not synthesis.continue_loop:
                    break
                if not self._adaptive_should_continue(
                    synthesis.coverage, all_evidence, request.max_iterations, iteration
                ):
                    break
            elif not synthesis.continue_loop:
                break

        agent_model_calls += 1
        citations = await self._citation.build_citations(request.query, all_evidence)
        citation_usage = absorb_llm_usage(
            getattr(self._citation, "last_llm_usage", None)
        )
        citation_metrics = with_model_call_count({"citations": len(citations)})
        if citation_usage:
            citation_metrics["llm_usage"] = citation_usage
        await emit(
            RunEvent(
                stage="cite",
                message="Citation subagent completed",
                metrics=citation_metrics,
            ),
        )

        agent_model_calls += 1
        draft = await self._lead.build_final_report(
            request=request,
            iteration_summaries=iteration_summaries,
            evidence_payload=[item.model_dump(mode="json") for item in all_evidence],
            citations_payload=[entry.model_dump(mode="json") for entry in citations],
        )
        if self._lead.fallback_count > lead_fallbacks:
            lead_fallbacks = self._lead.fallback_count
            await emit(
                RunEvent(
                    stage="error",
                    message="Lead reporter fell back to deterministic report",
                    payload=self._fallback_payload("build_final_report"),
                ),
            )
            self._record_fallback_reason(fallback_reasons, "build_final_report")
        rendered_report = self._report.render_result(request, draft, citations)
        report_markdown = rendered_report.markdown
        report_citations = rendered_report.citations
        report_usage = absorb_llm_usage(
            getattr(self._lead, "last_llm_usage", None)
        )
        report_metrics = with_model_call_count(
            {"report_words": len(report_markdown.split())}
        )
        if report_usage:
            report_metrics["llm_usage"] = report_usage
        await emit(
            RunEvent(
                stage="report",
                message="Lead researcher completed final report draft",
                metrics=report_metrics,
            ),
        )

        elapsed = time.perf_counter() - started
        run_stats: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 2),
            "iterations": len(iteration_summaries),
            "evidence_count": len(all_evidence),
            "candidate_citation_count": len(citations),
            "citation_count": len(report_citations),
            "agent_model_calls": agent_model_calls,
            "agent_fallbacks": lead_fallbacks + extraction_fallbacks,
        }
        if fallback_reasons:
            run_stats["fallback_reasons"] = list(fallback_reasons)
        if saw_llm_usage:
            run_stats["llm_usage"] = dict(llm_totals)
        if llm_totals["llm_calls"] > 0:
            run_stats["metered_calls"] = llm_totals["llm_calls"]
        if llm_totals["total_tokens"] > 0:
            run_stats["llm_tokens"] = llm_totals["total_tokens"]
        if llm_totals["cost_events"] > 0:
            run_stats["usd_spent"] = round(llm_totals["cost_usd"], 6)
        model_calls = run_stats.get("agent_model_calls")
        if (
            isinstance(model_calls, int)
            and model_calls > 0
            and llm_totals["llm_calls"] > 0
        ):
            if llm_totals["llm_calls"] < model_calls:
                run_stats["cost_coverage"] = "partial"
            else:
                run_stats["cost_coverage"] = "full"
        run_stats.update(self._quality_summary(all_evidence))

        result = ResearchRunResult(
            run_id=run_id,
            request=request,
            report_markdown=report_markdown,
            citations=report_citations,
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

    def _fallback_payload(self, method: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"method": method}
        reason = getattr(self._lead, "last_fallback_reason", None)
        if reason:
            payload["reason"] = str(reason)[:300]
        return payload

    def _record_fallback_reason(
        self, fallback_reasons: list[str], method: str
    ) -> None:
        reason = getattr(self._lead, "last_fallback_reason", None)
        if reason:
            fallback_reasons.append(f"{method}: {reason}"[:300])

    @staticmethod
    def _adaptive_should_continue(
        coverage: object,
        cumulative_evidence: list[Any],
        max_iterations: int,
        iteration: int,
    ) -> bool:
        if iteration + 1 >= max_iterations:
            return False

        cov = getattr(coverage, "coverage_score", 0.5)
        severity = getattr(coverage, "open_question_severity", 0.5)
        contradictions = getattr(coverage, "contradiction_count", 0)
        should = getattr(coverage, "should_continue", True)

        if should:
            return True

        domains: set[str] = set()
        high_conf = 0
        for ev in cumulative_evidence:
            d = getattr(ev, "domain", None)
            if d and isinstance(d, str):
                domains.add(d)
            conf = getattr(ev, "confidence", 0.0) or 0.0
            cred = getattr(ev, "credibility_score", None)
            if conf >= 0.7 and (cred is None or cred >= 0.6):
                high_conf += 1

        if float(cov) < 0.6:
            return True
        if float(severity) > 0.5:
            return True
        if int(contradictions) > 0:
            return True
        if len(domains) < 3:
            return True
        if high_conf < 2:
            return True

        return False

    @staticmethod
    def _quality_summary(evidence: list[EvidenceRecord]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        dated = 0
        for ev in evidence:
            label = ev.source_class or "unclassified"
            counts[label] = counts.get(label, 0) + 1
            if ev.published_at:
                dated += 1
        fraction = round(dated / len(evidence), 3) if evidence else 0.0
        return {"source_class_counts": counts, "dated_evidence_fraction": fraction}

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
        stage: Literal["search", "error"] = "search"

        if trace_type == "query_started":
            query = str(payload.get("query", "")).strip()
            message = (
                f"Task {task_id} searching query" if task_id else "Searching query"
            )
            if query:
                metrics["query"] = query
            if "max_results" in payload:
                metrics["max_results"] = payload["max_results"]
        elif trace_type == "query_completed":
            query = str(payload.get("query", "")).strip()
            message = (
                f"Task {task_id} query completed" if task_id else "Query completed"
            )
            if query:
                metrics["query"] = query
            if "hits" in payload:
                metrics["hits"] = payload["hits"]
        elif trace_type == "scrape_started":
            message = f"Task {task_id} scraping pages" if task_id else "Scraping pages"
            if "url_count" in payload:
                metrics["url_count"] = payload["url_count"]
        elif trace_type == "scrape_completed":
            message = (
                f"Task {task_id} scrape completed" if task_id else "Scrape completed"
            )
            if "scraped" in payload:
                metrics["scraped"] = payload["scraped"]
            if "missed" in payload:
                metrics["missed"] = payload["missed"]
        elif trace_type == "extract_completed":
            message = f"Task {task_id} extracted page" if task_id else "Extracted page"
            if "confidence" in payload:
                metrics["confidence"] = payload["confidence"]
            if payload.get("credibility") is not None:
                metrics["credibility"] = payload["credibility"]
        elif trace_type == "extract_started":
            message = (
                f"Task {task_id} extracting page" if task_id else "Extracting page"
            )
        elif trace_type == "fallback_evidence":
            message = (
                f"Task {task_id} fallback evidence added"
                if task_id
                else "Fallback evidence added"
            )
            if "confidence" in payload:
                metrics["confidence"] = payload["confidence"]
        elif trace_type == "extraction_fallback":
            message = (
                f"Task {task_id} extraction fell back to deterministic"
                if task_id
                else "Extraction fell back"
            )
            if "url" in payload:
                metrics["url"] = payload["url"]
        elif trace_type == "search_failed":
            stage = "error"
            message = (
                f"Task {task_id} search failed" if task_id else "Search failed"
            )
            if "query" in payload:
                metrics["query"] = payload["query"]

        return RunEvent(
            stage=stage,
            message=message,
            iteration=iteration,
            metrics=metrics,
            payload=payload,
        )
