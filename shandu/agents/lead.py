from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from typing import Any

from blackgeorge import Job, Worker
from pydantic import BaseModel, Field

from ..contracts import (
    CoverageAssessment,
    EvidenceRecord,
    FinalReportDraft,
    Finding,
    IterationPlan,
    IterationSynthesis,
    ReportSection,
    ResearchBrief,
    ResearchRequest,
    SubagentTask,
)
from ..interfaces import RuntimeExecutionLike
from ..prompts import (
    brief_instructions,
    brief_job,
    clarifier_instructions,
    clarifier_job,
    editor_instructions,
    editor_job,
    outliner_instructions,
    outliner_job,
    planner_instructions,
    planner_job,
    reporter_expected_output,
    reporter_instructions,
    reporter_job,
    section_writer_instructions,
    section_writer_job,
    synthesizer_instructions,
    synthesizer_job,
)
from ..runtime.costing import add_usage, collect_llm_usage
from ..runtime.roles import role_model
from ..services.findings import build_findings

logger = logging.getLogger(__name__)

# Reporter evidence budget in characters (~30k tokens), leaving room for
# instructions, prior summaries, and output inside typical context windows.
_REPORTER_EVIDENCE_BUDGET = 120_000
# Floor for one record's extracted text. A record gets an equal share of the
# budget when that is larger, so a small corpus reaches the reporter whole.
_MIN_EVIDENCE_TEXT_CHARS = 2200
# A reasoning model sometimes drafts the whole report inside its reasoning
# channel and then stops, so the call completes with empty content. The report
# is the one call a run cannot do without, so that case is retried.
_REPORTER_ATTEMPTS = 3
_MAX_SUB_QUESTIONS = 8
_MAX_SECTIONS = 9
_SECTION_CONCURRENCY = 4
# The sectioned writer's total length scales with the run's research effort.
_EFFORT_LENGTH = {"low": 0.75, "medium": 1.0, "high": 1.5, "max": 2.0}
# An edit shorter than this share of the draft lost content, not repetition.
_EDIT_MIN_RATIO = 0.6
_MAX_CLARIFYING_QUESTIONS = 3


def _failure_reason(report: Any, label: str) -> str:
    errors = getattr(report, "errors", None) or []
    detail = "; ".join(str(item) for item in errors if str(item).strip())
    if detail:
        return detail[:300]
    return f"{label} status={getattr(report, 'status', 'unknown')}"


class _PlanPayload(BaseModel):
    goals: list[str] = Field(default_factory=list)
    subagent_tasks: list[SubagentTask] = Field(default_factory=list)
    continue_loop: bool = True
    stop_reason: str | None = None


class _OutlineSection(BaseModel):
    heading: str
    purpose: str = ""
    sub_questions: list[int] = Field(default_factory=list)
    finding_ids: list[int] = Field(default_factory=list)
    target_words: int = 400


class _OutlinePayload(BaseModel):
    title: str
    sections: list[_OutlineSection] = Field(default_factory=list)


class _ClarifierPayload(BaseModel):
    questions: list[str] = Field(default_factory=list)


class _SynthesisPayload(BaseModel):
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    continue_loop: bool = True
    stop_reason: str | None = None
    coverage_score: float = Field(default=0.5, ge=0.0, le=1.0)
    open_question_severity: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction_count: int = Field(default=0, ge=0)
    recency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    coverage_should_continue: bool = True
    open_sub_questions: list[int] = Field(default_factory=list)


def _brief_payload(brief: ResearchBrief | None) -> dict[str, Any] | None:
    return brief.model_dump(mode="json") if brief is not None else None


class LeadAgent:
    def __init__(
        self, runtime: RuntimeExecutionLike, writer_model: str | None = None
    ) -> None:
        self._runtime = runtime
        self._writer_model = writer_model
        self.fallback_count = 0
        self.last_fallback_reason: str | None = None
        self.last_llm_usage: dict[str, Any] | None = None

    async def create_brief(self, request: ResearchRequest) -> ResearchBrief:
        payload = {
            "query": request.query,
            "clarifications": request.clarifications,
            "since": request.since.isoformat() if request.since else None,
            "include_domains": request.include_domains,
            "exclude_domains": request.exclude_domains,
            "today": date.today().isoformat(),
        }
        worker = Worker(
            name="LeadBriefWriter",
            model=role_model(self._runtime.settings, "planner"),
            instructions=brief_instructions(),
        )
        job = Job(input=brief_job(payload), response_schema=ResearchBrief)
        self.last_llm_usage = None
        try:
            report = await self._runtime.desk.arun(worker, job)
            self.last_llm_usage = collect_llm_usage(report)
            if report.status == "completed" and isinstance(report.data, ResearchBrief):
                sub_questions = [
                    question.strip()
                    for question in report.data.sub_questions
                    if question.strip()
                ][:_MAX_SUB_QUESTIONS]
                if sub_questions:
                    return report.data.model_copy(update={"sub_questions": sub_questions})
                reason = "brief writer returned no sub-questions"
            else:
                reason = _failure_reason(report, "brief writer")
            logger.warning("Lead brief call did not complete: %s", reason)
        except Exception:
            reason = "brief writer raised before completing"
            logger.warning("Lead brief call failed", exc_info=True)

        self.fallback_count += 1
        self.last_fallback_reason = reason
        return ResearchBrief(scope=request.query, sub_questions=[request.query])

    async def clarifying_questions(self, query: str) -> list[str]:
        worker = Worker(
            name="LeadClarifier",
            model=role_model(self._runtime.settings, "planner"),
            instructions=clarifier_instructions(),
        )
        job = Job(input=clarifier_job(query), response_schema=_ClarifierPayload)
        try:
            report = await self._runtime.desk.arun(worker, job)
        except Exception:
            logger.warning("Clarifier call failed", exc_info=True)
            return []
        if report.status != "completed" or not isinstance(report.data, _ClarifierPayload):
            logger.warning(
                "Clarifier call did not complete: %s", _failure_reason(report, "clarifier")
            )
            return []
        questions = [item.strip() for item in report.data.questions if item.strip()]
        return questions[:_MAX_CLARIFYING_QUESTIONS]

    async def create_iteration_plan(
        self,
        request: ResearchRequest,
        iteration: int,
        prior_summaries: list[IterationSynthesis],
        memory_context: list[tuple[str, Any]],
        brief: ResearchBrief | None = None,
    ) -> IterationPlan:
        payload = {
            "query": request.query,
            "brief": _brief_payload(brief),
            "iteration": iteration + 1,
            "max_iterations": request.max_iterations,
            "parallelism": request.parallelism,
            "detail_level": request.detail_level,
            "prior_summaries": [summary.model_dump(mode="json") for summary in prior_summaries],
            "memory_context": memory_context,
        }
        worker = Worker(
            name="LeadPlanner",
            model=role_model(self._runtime.settings, "planner"),
            instructions=planner_instructions(),
        )
        job = Job(
            input=planner_job(payload),
            response_schema=_PlanPayload,
        )
        self.last_llm_usage = None
        try:
            report = await self._runtime.desk.arun(worker, job)
            self.last_llm_usage = collect_llm_usage(report)
            if report.status == "completed" and isinstance(report.data, _PlanPayload):
                tasks = self._normalize_tasks(
                    report.data.subagent_tasks,
                    request=request,
                    iteration=iteration,
                    sub_question_count=len(brief.sub_questions) if brief else 0,
                )
                # An empty plan is the planner's stop signal; tasks that all
                # lack a focus are a malformed plan.
                if tasks or not report.data.subagent_tasks:
                    return IterationPlan(
                        iteration_index=iteration,
                        goals=report.data.goals,
                        subagent_tasks=tasks,
                        continue_loop=report.data.continue_loop,
                        stop_reason=report.data.stop_reason,
                    )
                reason = "planner returned no usable tasks"
            else:
                reason = _failure_reason(report, "planner")
            logger.warning("Lead planner call did not complete: %s", reason)
        except Exception:
            reason = "planner raised before completing"
            logger.warning("Lead planner call failed", exc_info=True)

        self.fallback_count += 1
        self.last_fallback_reason = reason

        return IterationPlan(
            iteration_index=iteration,
            goals=[request.query],
            subagent_tasks=self._fallback_tasks(request, iteration),
            continue_loop=True,
            stop_reason=None,
        )

    async def synthesize_iteration(
        self,
        request: ResearchRequest,
        iteration: int,
        iteration_evidence: list[dict[str, Any]],
        prior_summaries: list[IterationSynthesis],
        brief: ResearchBrief | None = None,
        task_memos: list[dict[str, str]] | None = None,
    ) -> IterationSynthesis:
        payload = {
            "query": request.query,
            "brief": _brief_payload(brief),
            "task_memos": task_memos or [],
            "iteration": iteration + 1,
            "max_iterations": request.max_iterations,
            "detail_level": request.detail_level,
            "iteration_evidence": iteration_evidence,
            "prior_summaries": [summary.model_dump(mode="json") for summary in prior_summaries],
        }
        worker = Worker(
            name="LeadSynthesizer",
            model=role_model(self._runtime.settings, "synthesizer"),
            instructions=synthesizer_instructions(),
        )
        job = Job(
            input=synthesizer_job(payload),
            response_schema=_SynthesisPayload,
        )
        self.last_llm_usage = None
        try:
            report = await self._runtime.desk.arun(worker, job)
            self.last_llm_usage = collect_llm_usage(report)
            if report.status == "completed" and isinstance(report.data, _SynthesisPayload):
                return IterationSynthesis(
                    summary=report.data.summary,
                    key_findings=report.data.key_findings,
                    open_questions=report.data.open_questions,
                    continue_loop=report.data.continue_loop,
                    stop_reason=report.data.stop_reason,
                    coverage=CoverageAssessment(
                        coverage_score=report.data.coverage_score,
                        open_question_severity=report.data.open_question_severity,
                        contradiction_count=report.data.contradiction_count,
                        recency_score=report.data.recency_score,
                        should_continue=report.data.coverage_should_continue,
                    ),
                    open_sub_questions=sorted(
                        {
                            index
                            for index in report.data.open_sub_questions
                            if brief is not None and 1 <= index <= len(brief.sub_questions)
                        }
                    ),
                )
            reason = _failure_reason(report, "synthesizer")
            logger.warning("Lead synthesizer call did not complete: %s", reason)
        except Exception:
            reason = "synthesizer raised before completing"
            logger.warning("Lead synthesizer call failed", exc_info=True)

        self.fallback_count += 1
        self.last_fallback_reason = reason

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
        brief: ResearchBrief | None = None,
    ) -> FinalReportDraft:
        self.last_llm_usage = None
        if request.writer == "sectioned":
            sectioned = await self._build_sectioned_report(
                request, iteration_summaries, evidence_payload, citations_payload, brief
            )
            if sectioned is not None:
                return sectioned
            logger.warning("Sectioned writer produced no report; using the single writer")
        payload = {
            "query": request.query,
            "brief": _brief_payload(brief),
            "detail_level": request.detail_level,
            "iterations": [summary.model_dump(mode="json") for summary in iteration_summaries],
            "evidence": self._compact_evidence(evidence_payload, citations_payload),
            "citations": self._compact_citations(citations_payload),
            "today": date.today().isoformat(),
        }
        target_words = self._word_target(request.detail_level)
        markdown, reason = await self._markdown_call(
            "LeadReporter", "reporter", reporter_instructions(), reporter_job(payload, target_words)
        )
        if markdown is not None:
            return self._draft_from_markdown(markdown, request.query)
        logger.warning("Lead reporter call did not complete: %s", reason)

        self.fallback_count += 1
        self.last_fallback_reason = reason

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

    async def _markdown_call(
        self, name: str, label: str, instructions: str, job_input: str
    ) -> tuple[str | None, str]:
        """Run a writing call and return its markdown, or None and the reason.

        A reasoning model sometimes drafts the whole text inside its reasoning
        channel and completes with empty content, so that case is retried.
        """
        worker = Worker(
            name=name,
            model=self._writer_model or role_model(self._runtime.settings, "writer"),
            instructions=instructions,
        )
        reason = f"{label} returned empty content"
        try:
            for _ in range(_REPORTER_ATTEMPTS):
                job = Job(input=job_input, expected_output=reporter_expected_output())
                report = await self._runtime.desk.arun(worker, job)
                self.last_llm_usage = add_usage(
                    self.last_llm_usage, collect_llm_usage(report)
                )
                if report.status != "completed":
                    return None, _failure_reason(report, label)
                content = getattr(report, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip(), ""
        except Exception:
            logger.warning("%s call failed", name, exc_info=True)
            return None, f"{label} raised before completing"
        return None, reason

    async def _build_sectioned_report(
        self,
        request: ResearchRequest,
        iteration_summaries: list[IterationSynthesis],
        evidence_payload: list[dict[str, Any]],
        citations_payload: list[dict[str, Any]],
        brief: ResearchBrief | None,
    ) -> FinalReportDraft | None:
        """Outline, draft each section from its own evidence, then edit.

        Returns None when no outline or no section comes back, so the caller
        falls back to the single-call writer.
        """
        records = [EvidenceRecord.model_validate(entry) for entry in evidence_payload]
        citation_ids = self._citation_ids(citations_payload)
        findings = build_findings(records)

        def cited(finding: Finding) -> list[int]:
            return sorted(
                {
                    citation_ids[evidence_id]
                    for evidence_id in finding.evidence_ids
                    if evidence_id in citation_ids
                }
            )

        finding_rows = [
            {
                "id": index,
                "statement": finding.statement,
                "sub_questions": finding.sub_questions,
                "citation_ids": cited(finding),
                "independent_sources": len(finding.domains),
                "confidence": round(finding.confidence, 2),
            }
            for index, finding in enumerate(findings, start=1)
        ]
        total_words = int(
            self._word_target(request.detail_level) * _EFFORT_LENGTH[request.effort]
        )
        outline_worker = Worker(
            name="LeadOutliner",
            model=self._writer_model or role_model(self._runtime.settings, "writer"),
            instructions=outliner_instructions(),
        )
        outline_job = Job(
            input=outliner_job(
                {
                    "query": request.query,
                    "brief": _brief_payload(brief),
                    "iterations": [
                        {"summary": summary.summary, "key_findings": summary.key_findings}
                        for summary in iteration_summaries
                    ],
                    "findings": finding_rows,
                    "today": date.today().isoformat(),
                },
                total_words,
            ),
            response_schema=_OutlinePayload,
        )
        try:
            report = await self._runtime.desk.arun(outline_worker, outline_job)
        except Exception:
            logger.warning("Outline call failed", exc_info=True)
            return None
        self.last_llm_usage = add_usage(self.last_llm_usage, collect_llm_usage(report))
        if report.status != "completed" or not isinstance(report.data, _OutlinePayload):
            logger.warning("Outline call did not complete: %s", _failure_reason(report, "outliner"))
            return None
        outline = report.data
        sections = [section for section in outline.sections if section.heading.strip()]
        if not sections:
            return None
        semaphore = asyncio.Semaphore(_SECTION_CONCURRENCY)
        sub_questions = brief.sub_questions if brief is not None else []

        async def write(section: _OutlineSection) -> str | None:
            chosen = [
                findings[index - 1]
                for index in dict.fromkeys(section.finding_ids)
                if 1 <= index <= len(findings)
            ] or [
                finding
                for finding in findings
                if set(finding.sub_questions) & set(section.sub_questions)
            ]
            supporting = {evidence_id for finding in chosen for evidence_id in finding.evidence_ids}
            # Evidence stored without claims yields no findings; such a
            # section draws on the whole budgeted evidence set instead.
            section_evidence = [
                entry for entry in evidence_payload if entry.get("evidence_id") in supporting
            ] or evidence_payload
            section_citations = {
                citation_ids.get(str(entry.get("evidence_id"))) for entry in section_evidence
            }
            payload = {
                "query": request.query,
                "heading": section.heading.strip(),
                "purpose": section.purpose,
                "answers": [
                    sub_questions[index - 1]
                    for index in section.sub_questions
                    if 1 <= index <= len(sub_questions)
                ],
                "target_words": section.target_words,
                "findings": [
                    {
                        "statement": finding.statement,
                        "quotes": finding.quotes[:3],
                        "citation_ids": cited(finding),
                    }
                    for finding in chosen
                ],
                "evidence": self._compact_evidence(section_evidence, citations_payload),
                # Only the ledger entries this section's evidence can cite.
                "citations": self._compact_citations(
                    [
                        entry
                        for entry in citations_payload
                        if entry.get("citation_id") in section_citations
                    ]
                ),
                "today": date.today().isoformat(),
            }
            async with semaphore:
                markdown, reason = await self._markdown_call(
                    "SectionWriter",
                    "section writer",
                    section_writer_instructions(),
                    section_writer_job(payload),
                )
            if markdown is None:
                logger.warning("Section %r was not written: %s", section.heading, reason)
            return markdown

        drafts = await asyncio.gather(*(write(section) for section in sections[:_MAX_SECTIONS]))
        written = [draft for draft in drafts if draft]
        if not written:
            return None
        title = outline.title.strip() or request.query.strip()[:180]
        draft = "\n\n".join(written)
        edited, reason = await self._markdown_call(
            "LeadEditor",
            "editor",
            editor_instructions(),
            editor_job({"title": title, "brief": _brief_payload(brief), "draft": draft}),
        )
        headings = re.findall(r"^##\s+.+$", draft, flags=re.MULTILINE)
        if (
            edited is not None
            and all(heading in edited for heading in headings)
            and len(edited.split()) >= _EDIT_MIN_RATIO * len(draft.split())
        ):
            markdown = edited
        else:
            logger.warning(
                "Editor output rejected (%s); keeping the section drafts",
                reason or "it dropped sections or content",
            )
            markdown = f"# {title}\n\n{draft}"
        return self._draft_from_markdown(markdown, request.query)

    @classmethod
    def _draft_from_markdown(cls, markdown: str, query: str) -> FinalReportDraft:
        return FinalReportDraft(
            title=cls._extract_title(markdown, query),
            executive_summary=cls._extract_summary(markdown),
            sections=[],
            markdown=markdown,
        )

    @staticmethod
    def _citation_ids(citations_payload: list[dict[str, Any]]) -> dict[str, int]:
        citation_by_evidence: dict[str, int] = {}
        for candidate in citations_payload:
            try:
                citation_id = int(candidate.get("citation_id", 0))
            except (TypeError, ValueError):
                continue
            if citation_id <= 0:
                continue
            for evidence_id in candidate.get("evidence_ids") or []:
                if evidence_id:
                    citation_by_evidence.setdefault(str(evidence_id), citation_id)
        return citation_by_evidence

    @staticmethod
    def _compact_evidence(
        evidence_payload: list[dict[str, Any]],
        citations_payload: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        citation_by_evidence = LeadAgent._citation_ids(citations_payload)
        text_chars = max(
            _MIN_EVIDENCE_TEXT_CHARS,
            _REPORTER_EVIDENCE_BUDGET // max(1, len(evidence_payload)),
        )
        compact: list[dict[str, Any]] = []
        scores: list[float] = []
        for entry in evidence_payload:
            try:
                confidence = float(entry.get("confidence", 0.5) or 0.5)
            except (TypeError, ValueError):
                confidence = 0.5
            try:
                credibility = float(entry.get("credibility_score"))
            except (TypeError, ValueError):
                credibility = 0.5
            scores.append(credibility * confidence)
            compact.append(
                {
                    "task_id": str(entry.get("task_id", "")),
                    "citation_id": citation_by_evidence.get(
                        str(entry.get("evidence_id", ""))
                    ),
                    "query": str(entry.get("query", "")),
                    "url": str(entry.get("requested_url", entry.get("url", ""))),
                    "domain": str(entry.get("domain", "") or ""),
                    "title": str(entry.get("title", "")),
                    "snippet": str(entry.get("snippet", "")),
                    "extracted_text": str(entry.get("extracted_text", ""))[:text_chars],
                    "confidence": confidence,
                    "published_at": entry.get("published_at"),
                    "source_class": entry.get("source_class"),
                    "credibility_score": entry.get("credibility_score"),
                    "quality_flags": entry.get("quality_flags") or [],
                }
            )
        return LeadAgent._apply_evidence_budget(compact, scores)

    @staticmethod
    def _apply_evidence_budget(
        compact: list[dict[str, Any]], scores: list[float]
    ) -> list[dict[str, Any]]:
        sizes = [
            len(json.dumps(record, ensure_ascii=False, default=str))
            for record in compact
        ]
        # Cited records claim the budget first, then the rest by score.
        order = sorted(
            range(len(compact)),
            key=lambda index: (compact[index].get("citation_id") is None, -scores[index]),
        )
        kept: set[int] = set()
        used = 0
        for index in order:
            if used + sizes[index] <= _REPORTER_EVIDENCE_BUDGET:
                kept.add(index)
                used += sizes[index]
        return [record for index, record in enumerate(compact) if index in kept]

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
    def _normalize_tasks(
        tasks: list[SubagentTask],
        request: ResearchRequest,
        iteration: int,
        sub_question_count: int = 0,
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
                    sub_questions=sorted(
                        {
                            index
                            for index in task.sub_questions
                            if 1 <= index <= sub_question_count
                        }
                    ),
                    scholarly=task.scholarly,
                )
            )

        return normalized[:target]

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
