from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from .contracts import (
    AISearchResult,
    CitationEntry,
    EvidenceRecord,
    FinalReportDraft,
    IterationPlan,
    IterationSynthesis,
    ResearchRequest,
    ResearchRunResult,
    RunEvent,
    SubagentTask,
)

DetailLevel = Literal["concise", "standard", "high"]
DepthPolicy = Literal["adaptive", "fixed"]


class DeskLike(Protocol):
    async def arun(self, worker: Any, job: Any) -> Any: ...


class RuntimeSettingsLike(Protocol):
    model: str


class RuntimeExecutionLike(Protocol):
    settings: Any
    desk: Any


class RuntimeInspectLike(Protocol):
    def inspect_run(self, run_id: str) -> dict[str, object]: ...


class SearchHitLike(Protocol):
    url: str
    title: str
    snippet: str


class ScrapedPageLike(Protocol):
    url: str
    title: str
    text: str


class SearchServiceLike(Protocol):
    async def search(self, query: str, max_results: int) -> Sequence[SearchHitLike]: ...


class ScrapeServiceLike(Protocol):
    async def scrape_many(self, urls: list[str]) -> Sequence[ScrapedPageLike]: ...


class LeadAgentLike(Protocol):
    async def create_iteration_plan(
        self,
        request: ResearchRequest,
        iteration: int,
        prior_summaries: list[IterationSynthesis],
        memory_context: list[tuple[str, Any]],
    ) -> IterationPlan: ...

    async def synthesize_iteration(
        self,
        request: ResearchRequest,
        iteration: int,
        iteration_evidence: list[dict[str, Any]],
        prior_summaries: list[IterationSynthesis],
    ) -> IterationSynthesis: ...

    async def build_final_report(
        self,
        request: ResearchRequest,
        iteration_summaries: list[IterationSynthesis],
        evidence_payload: list[dict[str, Any]],
        citations_payload: list[dict[str, Any]],
    ) -> FinalReportDraft: ...


class SearchSubagentLike(Protocol):
    async def execute_task(
        self,
        run_scope: str,
        task: SubagentTask,
        request: ResearchRequest,
    ) -> list[EvidenceRecord]: ...


class CitationAgentLike(Protocol):
    async def build_citations(
        self,
        query: str,
        evidence: list[EvidenceRecord],
    ) -> list[CitationEntry]: ...


class ReportServiceLike(Protocol):
    def render(
        self,
        request: ResearchRequest,
        draft: FinalReportDraft,
        citations: list[CitationEntry],
    ) -> str: ...


ProgressCallback = Callable[[RunEvent], Any]


class OrchestratorLike(Protocol):
    async def run(
        self,
        request: ResearchRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult: ...


class AISearchServiceLike(Protocol):
    async def search(
        self,
        query: str,
        max_results: int = 8,
        max_pages: int = 3,
        detail_level: DetailLevel = "standard",
    ) -> AISearchResult: ...
