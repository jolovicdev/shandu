from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceClass = Literal[
    "primary",
    "official",
    "peer_reviewed",
    "journalism",
    "technical_reference",
    "corporate",
    "expert_commentary",
    "community",
    "personal_blog",
    "advocacy_marketing",
    "aggregator",
    "social_profile",
    "unknown",
]

# quality_flags vocabulary: undated (no publication date), no_author (anonymous),
# self_published (publisher and subject are the same party), secondhand
# (summarizes primary work it does not contain), promotional (exists to persuade
# or sell), snippet_only (evidence is a search snippet, page not extracted),
# unassessed (no model source-quality assessment happened for this record).
QualityFlag = Literal[
    "undated",
    "no_author",
    "self_published",
    "secondhand",
    "promotional",
    "snippet_only",
    "unassessed",
]


class ResearchRequest(BaseModel):
    query: str
    max_iterations: int = Field(default=2, ge=1, le=8)
    parallelism: int = Field(default=3, ge=1, le=8)
    detail_level: Literal["concise", "standard", "high"] = "high"
    depth_policy: Literal["adaptive", "fixed"] = "adaptive"
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    max_pages_per_task: int = Field(default=3, ge=1, le=10)


class SubagentTask(BaseModel):
    task_id: str
    focus: str
    search_queries: list[str] = Field(default_factory=list)
    expected_output: str = ""


class IterationPlan(BaseModel):
    iteration_index: int
    goals: list[str] = Field(default_factory=list)
    subagent_tasks: list[SubagentTask] = Field(default_factory=list)
    continue_loop: bool = True
    stop_reason: str | None = None


class EvidenceRecord(BaseModel):
    evidence_id: str
    task_id: str
    query: str
    requested_url: str
    final_url: str | None = None
    domain: str | None = None
    title: str
    snippet: str
    extracted_text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    fetched_at: datetime | None = None
    published_at: str | None = None
    source_type: str | None = None
    extraction_method: str = ""
    relevance_score: float | None = None
    credibility_score: float | None = None
    source_class: SourceClass | None = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    fetch_error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CitationEntry(BaseModel):
    citation_id: int
    evidence_ids: list[str] = Field(default_factory=list)
    url: str
    title: str
    publisher: str
    accessed_at: str


class MemoryNote(BaseModel):
    key: str
    scope: str
    value: Any
    author: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoverageAssessment(BaseModel):
    coverage_score: float = Field(default=0.5, ge=0.0, le=1.0)
    open_question_severity: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction_count: int = Field(default=0, ge=0)
    recency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    should_continue: bool = True


class IterationSynthesis(BaseModel):
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    continue_loop: bool = True
    stop_reason: str | None = None
    coverage: CoverageAssessment | None = None


class ReportSection(BaseModel):
    heading: str
    content: str


class FinalReportDraft(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    markdown: str | None = None


class AISearchSource(BaseModel):
    title: str
    url: str
    snippet: str
    text_excerpt: str = ""


class AISearchResult(BaseModel):
    query: str
    answer_markdown: str
    sources: list[AISearchSource] = Field(default_factory=list)
    run_stats: dict[str, Any] = Field(default_factory=dict)


class RunEvent(BaseModel):
    stage: Literal[
        "bootstrap",
        "plan",
        "search",
        "synthesize",
        "cite",
        "report",
        "complete",
        "error",
    ]
    message: str
    iteration: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchRunResult(BaseModel):
    run_id: str
    request: ResearchRequest
    report_markdown: str
    citations: list[CitationEntry] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    iteration_summaries: list[IterationSynthesis] = Field(default_factory=list)
    run_stats: dict[str, Any] = Field(default_factory=dict)
