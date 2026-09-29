from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Literal

from blackgeorge.utils import new_id

from ..contracts import (
    CitationVerdict,
    EvidenceRecord,
    IterationSynthesis,
    ResearchBrief,
    ResearchRequest,
    ResearchRunResult,
    RunEvent,
    SubagentTask,
)
from ..interfaces import (
    CitationAgentLike,
    CitationAuditorLike,
    LeadAgentLike,
    ProgressCallback,
    ReportServiceLike,
    SearchSubagentLike,
)
from ..services.findings import build_findings, coverage_matrix
from ..services.run_store import CompletedIteration, RunStore, StoredRun


logger = logging.getLogger(__name__)


def _empty_llm_totals() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "llm_calls": 0,
        "cost_events": 0,
    }


@dataclass
class _RunState:
    brief: ResearchBrief | None = None
    evidence: list[EvidenceRecord] = field(default_factory=list)
    summaries: list[IterationSynthesis] = field(default_factory=list)
    task_focuses: list[str] = field(default_factory=list)
    queries_run: list[str] = field(default_factory=list)
    next_iteration: int = 0
    research_done: bool = False
    agent_model_calls: int = 0
    lead_fallbacks: int = 0
    extraction_fallbacks: int = 0
    fallback_reasons: list[str] = field(default_factory=list)
    llm_totals: dict[str, Any] = field(default_factory=_empty_llm_totals)
    saw_llm_usage: bool = False

    def counters(self) -> dict[str, Any]:
        return {
            "agent_model_calls": self.agent_model_calls,
            "lead_fallbacks": self.lead_fallbacks,
            "extraction_fallbacks": self.extraction_fallbacks,
            "fallback_reasons": list(self.fallback_reasons),
            "llm_totals": dict(self.llm_totals),
            "saw_llm_usage": self.saw_llm_usage,
        }

    @classmethod
    def from_stored(cls, stored: StoredRun, include_partial: bool) -> "_RunState":
        """Rebuild loop state from a stored run.

        Evidence from an iteration that never reached synthesis is kept only
        when ``include_partial`` is set: a resumed run re-plans that
        iteration, while a report rewrite uses everything that was read.
        """
        state = cls(brief=stored.brief)
        completed = {item.index for item in stored.iterations}
        state.evidence = [
            record
            for iteration, record in stored.evidence
            if include_partial or iteration in completed
        ]
        for item in stored.iterations:
            state.summaries.append(item.synthesis)
            for task in item.plan.subagent_tasks:
                state.task_focuses.append(task.focus)
                for query in task.search_queries:
                    if query not in state.queries_run:
                        state.queries_run.append(query)
        state.research_done = bool(stored.status.get("research_done"))
        if stored.iterations:
            state.next_iteration = stored.iterations[-1].index + 1
            counters = stored.iterations[-1].counters
            state.agent_model_calls = int(counters.get("agent_model_calls", 0))
            state.lead_fallbacks = int(counters.get("lead_fallbacks", 0))
            state.extraction_fallbacks = int(counters.get("extraction_fallbacks", 0))
            state.fallback_reasons = list(counters.get("fallback_reasons") or [])
            state.llm_totals.update(counters.get("llm_totals") or {})
            state.saw_llm_usage = bool(counters.get("saw_llm_usage"))
        return state


class LeadOrchestrator:
    def __init__(
        self,
        lead_agent: LeadAgentLike,
        search_subagent: SearchSubagentLike,
        citation_agent: CitationAgentLike,
        run_store: RunStore,
        report_service: ReportServiceLike,
        runtime_settings: dict[str, Any] | None = None,
        auditor: CitationAuditorLike | None = None,
    ) -> None:
        self._lead = lead_agent
        self._search_subagent = search_subagent
        self._citation = citation_agent
        self._store = run_store
        self._report = report_service
        self._auditor = auditor
        self._runtime_settings = (
            dict(runtime_settings) if runtime_settings is not None else None
        )

    async def clarifying_questions(self, query: str) -> list[str]:
        return await self._lead.clarifying_questions(query)

    async def run(
        self,
        request: ResearchRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        run_id = new_id()
        self._store.create(run_id, request, self._runtime_settings)
        return await self._execute(
            run_id, request, _RunState(), progress_callback, "Initializing run"
        )

    async def resume(
        self,
        run_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        stored = self._store.load(run_id)
        state = _RunState.from_stored(stored, include_partial=False)
        return await self._execute(
            run_id,
            stored.request,
            state,
            progress_callback,
            f"Resuming run at iteration {state.next_iteration + 1}",
        )

    async def rewrite_report(
        self,
        run_id: str,
        detail_level: Literal["concise", "standard", "high"] | None = None,
        progress_callback: ProgressCallback | None = None,
        writer: Literal["single", "sectioned"] | None = None,
        audit: bool | None = None,
    ) -> ResearchRunResult:
        stored = self._store.load(run_id)
        request = stored.request
        if detail_level is not None:
            request = request.model_copy(update={"detail_level": detail_level})
        if writer is not None:
            request = request.model_copy(update={"writer": writer})
        if audit is not None:
            request = request.model_copy(update={"audit": audit})
        state = _RunState.from_stored(stored, include_partial=True)
        state.research_done = True
        return await self._execute(
            run_id, request, state, progress_callback, "Rewriting report from stored evidence"
        )

    async def _execute(
        self,
        run_id: str,
        request: ResearchRequest,
        state: _RunState,
        progress_callback: ProgressCallback | None,
        start_message: str,
    ) -> ResearchRunResult:
        started = time.perf_counter()

        async def emit(event: RunEvent) -> None:
            self._store.append_event(run_id, event)
            await self._emit(progress_callback, event)

        try:
            self._store.set_status(run_id, "running")
            await emit(
                RunEvent(
                    stage="bootstrap",
                    message=start_message,
                    metrics={"run_id": run_id},
                ),
            )

            lead_seen = self._lead.fallback_count

            def lead_fell_back() -> bool:
                nonlocal lead_seen
                if self._lead.fallback_count <= lead_seen:
                    return False
                state.lead_fallbacks += self._lead.fallback_count - lead_seen
                lead_seen = self._lead.fallback_count
                return True

            def absorb_llm_usage(usage: Any) -> dict[str, Any] | None:
                if not isinstance(usage, dict) or not usage:
                    return None
                state.saw_llm_usage = True
                llm_totals = state.llm_totals
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
                if state.agent_model_calls > 0:
                    updated["agent_model_calls"] = state.agent_model_calls
                return updated

            if state.brief is None and not state.research_done:
                state.agent_model_calls += 1
                state.brief = await self._lead.create_brief(request)
                brief_usage = absorb_llm_usage(
                    getattr(self._lead, "last_llm_usage", None)
                )
                if lead_fell_back():
                    await emit(
                        RunEvent(
                            stage="error",
                            message="Lead brief writer fell back to the raw query",
                            payload=self._fallback_payload("create_brief"),
                        ),
                    )
                    self._record_fallback_reason(state.fallback_reasons, "create_brief")
                self._store.save_brief(run_id, state.brief)
                brief_metrics = with_model_call_count(
                    {"sub_questions": len(state.brief.sub_questions)}
                )
                if brief_usage:
                    brief_metrics["llm_usage"] = brief_usage
                await emit(
                    RunEvent(
                        stage="plan",
                        message="Research brief ready",
                        metrics=brief_metrics,
                        payload={"sub_questions": list(state.brief.sub_questions)},
                    ),
                )

            budget_stopped = False
            for iteration in range(
                state.next_iteration,
                state.next_iteration if state.research_done else request.max_iterations,
            ):
                budget = request.max_cost_usd
                if budget is not None and state.llm_totals["cost_usd"] >= budget:
                    budget_stopped = True
                    await emit(
                        RunEvent(
                            stage="synthesize",
                            message="Cost budget reached; writing the report",
                            iteration=iteration,
                            metrics={
                                "usd_spent": round(state.llm_totals["cost_usd"], 6),
                                "max_cost_usd": budget,
                            },
                        ),
                    )
                    break
                memory_context = [
                    ("prior_task_focuses", list(state.task_focuses)),
                    ("queries_run", list(state.queries_run)),
                ]
                state.agent_model_calls += 1
                plan = await self._lead.create_iteration_plan(
                    request=request,
                    iteration=iteration,
                    prior_summaries=state.summaries,
                    memory_context=memory_context,
                    brief=state.brief,
                )
                plan_usage = absorb_llm_usage(
                    getattr(self._lead, "last_llm_usage", None)
                )
                if lead_fell_back():
                    await emit(
                        RunEvent(
                            stage="error",
                            message="Lead planner fell back to deterministic plan",
                            iteration=iteration,
                            payload=self._fallback_payload("create_iteration_plan"),
                        ),
                    )
                    self._record_fallback_reason(
                        state.fallback_reasons, "create_iteration_plan"
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

                # Planned tasks always run; continue_loop=False ends the loop
                # after them. A planner stops early by returning no tasks.
                if not plan.subagent_tasks:
                    break

                # A failed fetch leaves snippet-only evidence; its URL stays
                # eligible for another attempt.
                fetched = [item for item in state.evidence if not item.fetch_error]
                extracted_urls = {item.requested_url for item in fetched}
                extracted_urls.update(item.final_url for item in fetched if item.final_url)
                semaphore = asyncio.Semaphore(request.parallelism)
                task_total = len(plan.subagent_tasks)
                completed_tasks = 0
                completed_lock = asyncio.Lock()
                task_memos: list[dict[str, str]] = []

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
                        if trace_type in ("extract_started", "urls_ranked", "task_memo"):
                            state.agent_model_calls += 1
                        if trace_type == "task_memo" and payload.get("memo"):
                            task_memos.append(
                                {
                                    "task_id": str(payload.get("task_id", "")),
                                    "focus": str(payload.get("focus", "")),
                                    "memo": str(payload["memo"]),
                                }
                            )
                        elif trace_type == "extraction_fallback":
                            state.extraction_fallbacks += 1
                            trace_reason = payload.get("reason")
                            if trace_reason:
                                state.fallback_reasons.append(str(trace_reason)[:300])
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
                            evidence = await self._search_subagent.execute_task(
                                f"run:{run_id}",
                                task,
                                request,
                                progress_callback=on_search_trace,
                                extracted_urls=extracted_urls,
                                brief=state.brief,
                            )
                        self._store.append_evidence(run_id, iteration, evidence)
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

                state.evidence.extend(iteration_evidence)
                state.task_focuses.extend(task.focus for task in plan.subagent_tasks)
                for task in plan.subagent_tasks:
                    for query in task.search_queries:
                        if query not in state.queries_run:
                            state.queries_run.append(query)
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

                state.agent_model_calls += 1
                synthesis = await self._lead.synthesize_iteration(
                    request=request,
                    iteration=iteration,
                    iteration_evidence=[
                        item.model_dump(mode="json") for item in iteration_evidence
                    ],
                    prior_summaries=state.summaries,
                    brief=state.brief,
                    task_memos=task_memos,
                )
                synthesis_usage = absorb_llm_usage(
                    getattr(self._lead, "last_llm_usage", None)
                )
                if state.brief is not None:
                    matrix = coverage_matrix(
                        state.brief, state.evidence, build_findings(state.evidence)
                    )
                    gaps = {item.index for item in matrix if item.strength != "strong"}
                    synthesis = synthesis.model_copy(
                        update={
                            "coverage_matrix": matrix,
                            "open_sub_questions": sorted(
                                gaps | set(synthesis.open_sub_questions)
                            ),
                        }
                    )
                if lead_fell_back():
                    await emit(
                        RunEvent(
                            stage="error",
                            message="Lead synthesizer fell back to deterministic synthesis",
                            iteration=iteration,
                            payload=self._fallback_payload("synthesize_iteration"),
                        ),
                    )
                    self._record_fallback_reason(
                        state.fallback_reasons, "synthesize_iteration"
                    )
                state.summaries.append(synthesis)
                self._store.save_iteration(
                    run_id,
                    CompletedIteration(
                        index=iteration,
                        plan=plan,
                        synthesis=synthesis,
                        counters=state.counters(),
                    ),
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

                if request.depth_policy == "adaptive":
                    if iteration + 1 >= request.max_iterations:
                        break
                    if not self._adaptive_should_continue(
                        synthesis.coverage,
                        state.evidence,
                        request.max_iterations,
                        iteration,
                        open_sub_questions=len(synthesis.open_sub_questions)
                        if state.brief is not None
                        else None,
                    ):
                        break
                elif not synthesis.continue_loop:
                    break
            self._store.mark_research_done(run_id)

            citations = await self._citation.build_citations(request.query, state.evidence)
            citation_metrics = with_model_call_count({"citations": len(citations)})
            await emit(
                RunEvent(
                    stage="cite",
                    message="Citation subagent completed",
                    metrics=citation_metrics,
                ),
            )

            state.agent_model_calls += 1
            draft = await self._lead.build_final_report(
                request=request,
                iteration_summaries=state.summaries,
                evidence_payload=[item.model_dump(mode="json") for item in state.evidence],
                citations_payload=[entry.model_dump(mode="json") for entry in citations],
                brief=state.brief,
            )
            if lead_fell_back():
                await emit(
                    RunEvent(
                        stage="error",
                        message="Lead reporter fell back to deterministic report",
                        payload=self._fallback_payload("build_final_report"),
                    ),
                )
                self._record_fallback_reason(state.fallback_reasons, "build_final_report")
            rendered_report = self._report.render_result(request, draft, citations)
            report_usage = absorb_llm_usage(
                getattr(self._lead, "last_llm_usage", None)
            )
            # The sectioned writer makes an outline, section, and edit call.
            if report_usage and isinstance(report_usage.get("llm_calls"), int):
                state.agent_model_calls += max(0, report_usage["llm_calls"] - 1)
            report_metrics = with_model_call_count(
                {"report_words": len(rendered_report.markdown.split())}
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

            citation_audit: list[CitationVerdict] = []
            if self._auditor is not None and request.audit and rendered_report.citations:
                outcome = await self._auditor.audit(
                    rendered_report.markdown, rendered_report.citations, state.evidence
                )
                state.agent_model_calls += outcome.calls
                audit_usage = absorb_llm_usage(
                    {**outcome.llm_usage, "llm_calls": outcome.calls}
                    if outcome.llm_usage
                    else None
                )
                citation_audit = outcome.verdicts
                # Re-rendering drops references that lost their last marker
                # and renumbers the rest.
                rendered_report = self._report.render_result(
                    request,
                    draft.model_copy(update={"markdown": outcome.markdown}),
                    rendered_report.citations,
                )
                audit_metrics = with_model_call_count(
                    {"citation_audit": self._audit_counts(citation_audit)}
                )
                if audit_usage:
                    audit_metrics["llm_usage"] = audit_usage
                await emit(
                    RunEvent(
                        stage="report",
                        message="Citation audit completed",
                        metrics=audit_metrics,
                    ),
                )
            report_markdown = rendered_report.markdown
            report_citations = rendered_report.citations

            llm_totals = state.llm_totals
            elapsed = time.perf_counter() - started
            run_stats: dict[str, Any] = {
                "elapsed_seconds": round(elapsed, 2),
                "iterations": len(state.summaries),
                "evidence_count": len(state.evidence),
                "candidate_citation_count": len(citations),
                "citation_count": len(report_citations),
                "agent_model_calls": state.agent_model_calls,
                "agent_fallbacks": state.lead_fallbacks + state.extraction_fallbacks,
            }
            if state.fallback_reasons:
                run_stats["fallback_reasons"] = list(state.fallback_reasons)
            if state.saw_llm_usage:
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
            run_stats.update(self._quality_summary(state.evidence))
            findings = build_findings(state.evidence)
            run_stats["findings"] = len(findings)
            if budget_stopped:
                run_stats["stopped_at_cost_budget"] = True
            if citation_audit:
                run_stats["citation_audit"] = self._audit_counts(citation_audit)
            final_matrix = state.summaries[-1].coverage_matrix if state.summaries else []
            if final_matrix:
                strengths = [item.strength for item in final_matrix]
                run_stats["sub_question_coverage"] = {
                    level: strengths.count(level)
                    for level in ("strong", "moderate", "weak", "none")
                }

            result = ResearchRunResult(
                run_id=run_id,
                request=request,
                brief=state.brief,
                report_markdown=report_markdown,
                citations=report_citations,
                evidence=state.evidence,
                findings=findings,
                citation_audit=citation_audit,
                iteration_summaries=state.summaries,
                run_stats=run_stats,
            )

            self._store.save_report(
                run_id, report_markdown, report_citations, run_stats, citation_audit
            )
            self._store.set_status(run_id, "completed")
            await emit(
                RunEvent(
                    stage="complete",
                    message="Run completed",
                    metrics=result.run_stats,
                    payload={"run_id": run_id},
                )
            )
            return result
        except BaseException as exc:
            terminal_status = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            try:
                self._store.set_status(
                    run_id, terminal_status, f"{type(exc).__name__}: {exc}"[:300]
                )
            except Exception:
                logger.warning(
                    "Failed to persist terminal run state", exc_info=True
                )
            raise

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
        open_sub_questions: int | None = None,
    ) -> bool:
        if iteration + 1 >= max_iterations:
            return False
        # With a brief, the sub-question checklist decides: keep going while
        # any sub-question lacks two independent credible sources or sources
        # conflict. The model's own coverage score only decides without one.
        if open_sub_questions is not None:
            return open_sub_questions > 0 or int(
                getattr(coverage, "contradiction_count", 0) or 0
            ) > 0

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

        # No coverage (deterministic synthesis fallback): the model's vote
        # is missing, so the evidence signals alone decide.
        if coverage is None:
            if len(domains) < 3:
                return True
            if high_conf < 2:
                return True
            return False

        cov = getattr(coverage, "coverage_score", 0.5)
        severity = getattr(coverage, "open_question_severity", 0.5)
        contradictions = getattr(coverage, "contradiction_count", 0)
        should = getattr(coverage, "should_continue", True)

        if should:
            return True

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
    def _audit_counts(verdicts: list[CitationVerdict]) -> dict[str, int]:
        return {
            verdict: sum(1 for item in verdicts if item.verdict == verdict)
            for verdict in ("supported", "partial", "unsupported", "unchecked")
        }

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
        elif trace_type == "urls_ranked":
            message = f"Task {task_id} ranked sources" if task_id else "Ranked sources"
            if "candidates" in payload:
                metrics["candidates"] = payload["candidates"]
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
        elif trace_type == "task_memo":
            message = (
                f"Task {task_id} researcher finished" if task_id else "Researcher finished"
            )
            for key in ("searches", "reads"):
                if key in payload:
                    metrics[key] = payload[key]
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
