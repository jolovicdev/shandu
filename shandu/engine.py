from __future__ import annotations

import asyncio
from typing import Any
from collections.abc import AsyncIterator, Callable

from .agents import CitationAgent, LeadAgent, SearchSubagent
from .contracts import AISearchResult, ResearchRequest, ResearchRunResult, RunEvent
from .interfaces import (
    AISearchServiceLike,
    DetailLevel,
    OrchestratorLike,
    RuntimeInspectLike,
)
from .orchestration import LeadOrchestrator
from .runtime import get_async_runner
from .runtime.bootstrap import get_bootstrap
from .services import AISearchService, MemoryService, ReportService, ScrapeService, SearchService

ProgressCallback = Callable[[RunEvent], Any]


class ShanduEngine:
    def __init__(
        self,
        runtime: RuntimeInspectLike,
        orchestrator: OrchestratorLike,
        ai_search_service: AISearchServiceLike,
    ) -> None:
        self._runtime = runtime
        self._orchestrator = orchestrator
        self._ai_search = ai_search_service

    @classmethod
    def from_config(cls) -> "ShanduEngine":
        runtime = get_bootstrap()
        search_service = SearchService()
        scrape_service = ScrapeService()
        memory_service = MemoryService(runtime.memory_store)
        report_service = ReportService()

        lead = LeadAgent(runtime)
        subagent = SearchSubagent(runtime, search_service, scrape_service)
        citation = CitationAgent(runtime)
        orchestrator = LeadOrchestrator(
            lead_agent=lead,
            search_subagent=subagent,
            citation_agent=citation,
            memory_service=memory_service,
            report_service=report_service,
        )
        ai_search_service = AISearchService(runtime, search_service, scrape_service)
        return cls(
            runtime=runtime,
            orchestrator=orchestrator,
            ai_search_service=ai_search_service,
        )

    async def run(
        self,
        request: ResearchRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        return await self._orchestrator.run(request, progress_callback)

    def run_sync(
        self,
        request: ResearchRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        runner = get_async_runner()
        return runner.run(self.run(request, progress_callback))

    async def stream(self, request: ResearchRequest) -> AsyncIterator[RunEvent]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        finished = asyncio.Event()
        error: Exception | None = None

        async def on_event(event: RunEvent) -> None:
            await queue.put(event)

        async def worker() -> None:
            nonlocal error
            try:
                await self.run(request, on_event)
            except Exception as exc:
                error = exc
                await queue.put(RunEvent(stage="error", message=str(exc)))
            finally:
                finished.set()

        task = asyncio.create_task(worker())
        while not finished.is_set() or not queue.empty():
            event = await queue.get()
            yield event
        await task
        if error is not None:
            raise error

    def inspect_run(self, run_id: str) -> dict[str, object]:
        return self._runtime.inspect_run(run_id)

    async def ai_search(
        self,
        query: str,
        max_results: int = 8,
        max_pages: int = 3,
        detail_level: DetailLevel = "standard",
    ) -> AISearchResult:
        return await self._ai_search.search(
            query=query,
            max_results=max_results,
            max_pages=max_pages,
            detail_level=detail_level,
        )

    def ai_search_sync(
        self,
        query: str,
        max_results: int = 8,
        max_pages: int = 3,
        detail_level: DetailLevel = "standard",
    ) -> AISearchResult:
        runner = get_async_runner()
        return runner.run(
            self.ai_search(
                query=query,
                max_results=max_results,
                max_pages=max_pages,
                detail_level=detail_level,
            )
        )
