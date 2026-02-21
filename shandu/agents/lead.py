from __future__ import annotations

import json
from datetime import date
from typing import Any

from blackgeorge import Job, Worker
from pydantic import BaseModel, Field

from ..contracts import (
    FinalReportDraft,
    IterationPlan,
    IterationSynthesis,
    ReportSection,
    ResearchRequest,
    SubagentTask,
)
from ..interfaces import RuntimeExecutionLike


class _PlanPayload(BaseModel):
    goals: list[str] = Field(default_factory=list)
    subagent_tasks: list[SubagentTask] = Field(default_factory=list)
    continue_loop: bool = True
    stop_reason: str | None = None


class _SynthesisPayload(BaseModel):
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    continue_loop: bool = True
    stop_reason: str | None = None


class LeadAgent:
    def __init__(self, runtime: RuntimeExecutionLike) -> None:
        self._runtime = runtime

    async def create_iteration_plan(
        self,
        request: ResearchRequest,
        iteration: int,
        prior_summaries: list[IterationSynthesis],
        memory_context: list[tuple[str, Any]],
    ) -> IterationPlan:
        payload = {
            "query": request.query,
            "iteration": iteration + 1,
            "max_iterations": request.max_iterations,
            "parallelism": request.parallelism,
            "detail_level": request.detail_level,
            "prior_summaries": [summary.model_dump(mode="json") for summary in prior_summaries],
            "memory_context": memory_context,
        }
        worker = Worker(
            name="LeadPlanner",
            model=self._runtime.settings.model,
            instructions=(
                "You are LeadPlanner for a multi-agent research system. "
                "Return only valid structured output. "
                "Design independent subagent tasks that maximize source diversity, source quality, "
                "and evidence coverage. "
                "Avoid overlapping tasks unless the query is narrow. "
                "Each task must have a clear focus, practical search queries, and explicit expected evidence. "
                "Prefer primary sources, recent data, and high-authority publications."
            ),
        )
        job = Job(
            input=(
                "Create the next iteration plan as structured data.\n"
                "Requirements:\n"
                "- Use prior_summaries and memory_context to avoid duplicated research.\n"
                "- Return tasks that can run in parallel with distinct evidence goals.\n"
                "- Target at least requested parallelism unless the query is provably narrow.\n"
                "- Task IDs must be unique and stable strings.\n"
                "- search_queries must be high-signal and specific enough to retrieve factual evidence.\n"
                "- continue_loop=false only when enough evidence already exists to answer query well.\n"
                f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
            response_schema=_PlanPayload,
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            if report.status == "completed" and isinstance(report.data, _PlanPayload):
                tasks = self._ensure_parallel_task_count(
                    report.data.subagent_tasks,
                    request=request,
                    iteration=iteration,
                )
                return IterationPlan(
                    iteration_index=iteration,
                    goals=report.data.goals,
                    subagent_tasks=tasks,
                    continue_loop=report.data.continue_loop,
                    stop_reason=report.data.stop_reason,
                )
        except Exception:
            pass

        return IterationPlan(
            iteration_index=iteration,
            goals=[request.query],
            subagent_tasks=self._ensure_parallel_task_count([], request, iteration),
            continue_loop=True,
            stop_reason=None,
        )

    async def synthesize_iteration(
        self,
        request: ResearchRequest,
        iteration: int,
        iteration_evidence: list[dict[str, Any]],
        prior_summaries: list[IterationSynthesis],
    ) -> IterationSynthesis:
        payload = {
            "query": request.query,
            "iteration": iteration + 1,
            "max_iterations": request.max_iterations,
            "detail_level": request.detail_level,
            "iteration_evidence": iteration_evidence,
            "prior_summaries": [summary.model_dump(mode="json") for summary in prior_summaries],
        }
        worker = Worker(
            name="LeadSynthesizer",
            model=self._runtime.settings.model,
            instructions=(
                "You are LeadSynthesizer. "
                "Synthesize only from supplied evidence and prior summaries. "
                "Separate validated findings from unknowns. "
                "Avoid duplicative statements and prioritize decision-useful synthesis."
            ),
        )
        job = Job(
            input=(
                "Synthesize this iteration and decide whether another research loop is needed.\n"
                "Requirements:\n"
                "- summary should state what is known now and why.\n"
                "- key_findings should contain concrete, evidence-backed points.\n"
                "- open_questions should capture missing evidence required for confidence.\n"
                "- continue_loop=false if evidence is already sufficient or no productive next step remains.\n"
                f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
            response_schema=_SynthesisPayload,
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            if report.status == "completed" and isinstance(report.data, _SynthesisPayload):
                return IterationSynthesis(
                    summary=report.data.summary,
                    key_findings=report.data.key_findings,
                    open_questions=report.data.open_questions,
                    continue_loop=report.data.continue_loop,
                    stop_reason=report.data.stop_reason,
                )
        except Exception:
            pass

        fallback_summary = "No structured synthesis available; using deterministic fallback."
        continue_loop = iteration + 1 < request.max_iterations and bool(iteration_evidence)
        return IterationSynthesis(
            summary=fallback_summary,
            key_findings=[entry.get("snippet", "") for entry in iteration_evidence[:5]],
            open_questions=[],
            continue_loop=continue_loop,
            stop_reason=None if continue_loop else "Iteration budget reached",
        )

    async def build_final_report(
        self,
        request: ResearchRequest,
        iteration_summaries: list[IterationSynthesis],
        evidence_payload: list[dict[str, Any]],
        citations_payload: list[dict[str, Any]],
    ) -> FinalReportDraft:
        payload = {
            "query": request.query,
            "detail_level": request.detail_level,
            "iterations": [summary.model_dump(mode="json") for summary in iteration_summaries],
            "evidence": self._compact_evidence(evidence_payload),
            "citations": self._compact_citations(citations_payload),
            "today": date.today().isoformat(),
        }
        target_words = self._word_target(request.detail_level)
        worker = Worker(
            name="LeadReporter",
            model=self._runtime.settings.model,
            instructions=(
                "You are LeadReporter. "
                "Write a publication-grade long-form report that is rigorous, coherent, and source-grounded. "
                "Do not fabricate facts, numbers, or citations. "
                "Every concrete claim should be supported by provided citations when available. "
                "Use clear argument flow, explicit caveats, and balanced counterpoints."
            ),
        )
        job = Job(
            input=(
                "Write the final report directly in markdown.\n"
                f"Minimum body length: {target_words} words before References.\n"
                "Use this structure exactly:\n"
                "# <Title>\n"
                "## Executive Summary\n"
                "## Key Findings\n"
                "## Detailed Analysis\n"
                "## Risks and Counterpoints\n"
                "## Open Questions\n"
                "## References\n"
                "Use citation markers like [1], [2], ... and only cite sources provided in payload.\n"
                "Prefer coherent paragraphs over bullets in Detailed Analysis.\n"
                "Do not include internal IDs in citations.\n"
                "Keep claims calibrated: state uncertainty when evidence is limited or conflicting.\n"
                f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
            expected_output="A very long markdown report with explicit citations and references.",
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            content = getattr(report, "content", None)
            if report.status == "completed" and isinstance(content, str) and content.strip():
                markdown = content.strip()
                return FinalReportDraft(
                    title=self._extract_title(markdown, request.query),
                    executive_summary=self._extract_summary(markdown),
                    sections=[],
                    markdown=markdown,
                )
        except Exception:
            pass

        findings = [
            item
            for summary in iteration_summaries
            for item in summary.key_findings
            if isinstance(item, str) and item.strip()
        ]
        sections = [
            ReportSection(
                heading="Key Findings",
                content="\n".join(f"- {item}" for item in findings[:24])
                or "\n".join(
                    f"- {summary.summary}" for summary in iteration_summaries if summary.summary
                )
                or "No detailed findings were captured.",
            ),
            ReportSection(
                heading="Detailed Analysis",
                content="\n".join(
                    (
                        f"{idx + 1}. {entry.get('title', 'Untitled')} "
                        f"({entry.get('url', '')})\n"
                        f"{entry.get('extracted_text', '')[:900]}"
                    )
                    for idx, entry in enumerate(evidence_payload[:16])
                )
                or "No evidence records available.",
            ),
        ]
        return FinalReportDraft(
            title=request.query.strip()[:120],
            executive_summary="Model markdown generation failed; this deterministic report uses collected evidence and synthesized findings.",
            sections=sections,
        )

    @staticmethod
    def _compact_evidence(evidence_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for entry in evidence_payload:
            try:
                confidence = float(entry.get("confidence", 0.5) or 0.5)
            except (TypeError, ValueError):
                confidence = 0.5
            compact.append(
                {
                    "task_id": str(entry.get("task_id", "")),
                    "query": str(entry.get("query", "")),
                    "url": str(entry.get("url", "")),
                    "title": str(entry.get("title", "")),
                    "snippet": str(entry.get("snippet", "")),
                    "extracted_text": str(entry.get("extracted_text", ""))[:2200],
                    "confidence": confidence,
                }
            )
        return compact

    @staticmethod
    def _compact_citations(citations_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for entry in citations_payload:
            citation_id = entry.get("citation_id", 0)
            try:
                normalized_id = int(citation_id)
            except (TypeError, ValueError):
                normalized_id = 0
            compact.append(
                {
                    "citation_id": normalized_id,
                    "url": str(entry.get("url", "")),
                    "title": str(entry.get("title", "")),
                    "publisher": str(entry.get("publisher", "")),
                    "accessed_at": str(entry.get("accessed_at", "")),
                }
            )
        return compact

    @staticmethod
    def _ensure_parallel_task_count(
        tasks: list[SubagentTask],
        request: ResearchRequest,
        iteration: int,
    ) -> list[SubagentTask]:
        target = max(1, min(request.parallelism, 8))
        normalized: list[SubagentTask] = []
        seen_ids: set[str] = set()

        for idx, task in enumerate(tasks, start=1):
            focus = task.focus.strip()
            if not focus:
                continue
            candidate_id = task.task_id.strip() or f"iter_{iteration + 1}_task_{idx}"
            if candidate_id in seen_ids:
                candidate_id = f"iter_{iteration + 1}_task_{len(normalized) + 1}"
            seen_ids.add(candidate_id)
            queries = [query.strip() for query in task.search_queries if query.strip()]
            if not queries:
                queries = [focus]
            normalized.append(
                SubagentTask(
                    task_id=candidate_id,
                    focus=focus,
                    search_queries=queries,
                    expected_output=task.expected_output.strip()
                    or "High quality evidence with primary-source links",
                )
            )

        if not normalized:
            normalized = LeadAgent._fallback_tasks(request, iteration)

        facets = [
            "latest developments",
            "market landscape",
            "technical details",
            "counterarguments",
            "regional data",
            "expert analysis",
            "primary-source statements",
            "case studies",
        ]
        facet_index = 0
        while len(normalized) < target:
            base_focus = normalized[facet_index % len(normalized)].focus
            facet = facets[facet_index % len(facets)]
            task_number = len(normalized) + 1
            normalized.append(
                SubagentTask(
                    task_id=f"iter_{iteration + 1}_task_{task_number}",
                    focus=f"{base_focus} - {facet}",
                    search_queries=[f"{request.query} {facet}", base_focus],
                    expected_output="Independent evidence track with distinct sources.",
                )
            )
            facet_index += 1

        return normalized

    @staticmethod
    def _fallback_tasks(request: ResearchRequest, iteration: int) -> list[SubagentTask]:
        facets = [
            "overview",
            "current status",
            "primary sources",
            "expert commentary",
            "risks",
            "contrarian views",
            "regional angle",
            "implementation details",
        ]
        target = max(1, min(request.parallelism, 8))
        tasks: list[SubagentTask] = []
        for index in range(target):
            facet = facets[index % len(facets)]
            focus = request.query if index == 0 else f"{request.query} - {facet}"
            queries = [request.query] if index == 0 else [f"{request.query} {facet}", request.query]
            tasks.append(
                SubagentTask(
                    task_id=f"iter_{iteration + 1}_task_{index + 1}",
                    focus=focus,
                    search_queries=queries,
                    expected_output="High quality evidence with primary-source links",
                )
            )
        return tasks

    @staticmethod
    def _word_target(detail_level: str) -> int:
        if detail_level == "concise":
            return 1200
        if detail_level == "standard":
            return 2200
        return 3600

    @staticmethod
    def _extract_title(markdown: str, fallback_query: str) -> str:
        for line in markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and stripped[2:].strip():
                return stripped[2:].strip()[:180]
        return fallback_query.strip()[:180] or "Research Report"

    @staticmethod
    def _extract_summary(markdown: str) -> str:
        lines = [line.strip() for line in markdown.splitlines()]
        capture = False
        summary_lines: list[str] = []
        for line in lines:
            if line.lower().startswith("## executive summary"):
                capture = True
                continue
            if capture and line.startswith("## "):
                break
            if capture and line:
                summary_lines.append(line)
            if len(" ".join(summary_lines).split()) >= 120:
                break
        if summary_lines:
            return " ".join(summary_lines)
        for line in lines:
            if line and not line.startswith("#"):
                return line[:480]
        return "Summary unavailable."
