from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager

from .agents import CitationAgent, LeadAgent, SearchSubagent
from .agents.auditor import CitationAuditor
from .agents.researcher import ResearcherAgent, ResearcherByRequest
from .contracts import AISearchResult, ResearchRequest, ResearchRunResult, RunEvent
from .interfaces import (
    AISearchServiceLike,
    DetailLevel,
    OrchestratorLike,
    RuntimeInspectLike,
)
from .orchestration import LeadOrchestrator
from .runtime import get_async_runner
from .runtime.bootstrap import (
    acquire_bootstrap_run,
    get_bootstrap,
    release_bootstrap_run,
)
from .services import AISearchService, ReportService, ScrapeService, SearchService
from .services.scholarly import ScholarlySearchService
from .services.run_store import RunNotFoundError, RunStore, RunSummary

ProgressCallback = Callable[[RunEvent], Any]


@contextmanager
def _bootstrap_run() -> Iterator[None]:
    """Keep the runtime open for the duration of one call."""
    acquire_bootstrap_run()
    try:
        yield
    finally:
        release_bootstrap_run()


class ShanduEngine:
    def __init__(
        self,
        runtime: RuntimeInspectLike,
        orchestrator: OrchestratorLike,
        ai_search_service: AISearchServiceLike,
        run_store: RunStore,
        scrape_service: ScrapeService | None = None,
        scholarly_service: ScholarlySearchService | None = None,
        search_service: SearchService | None = None,
    ) -> None:
        self._runtime = runtime
        self._orchestrator = orchestrator
        self._ai_search = ai_search_service
        self._run_store = run_store
        # Services holding HTTP sessions, closed with the engine.
        self._closables = [
            service
            for service in (scrape_service, scholarly_service, search_service)
            if service is not None
        ]

    @classmethod
    def from_config(cls, writer_model: str | None = None) -> "ShanduEngine":
        runtime = get_bootstrap()
        search_service = SearchService()
        scrape_service = ScrapeService()
        scholarly_service = ScholarlySearchService()
        run_store = RunStore(Path(runtime.settings.storage_dir) / "runs")
        report_service = ReportService()

        lead = LeadAgent(runtime, writer_model=writer_model)
        subagent = ResearcherByRequest(
            pipeline=SearchSubagent(
                runtime, search_service, scrape_service, scholarly_service
            ),
            agentic=ResearcherAgent(
                runtime, search_service, scrape_service, scholarly_service
            ),
        )
        citation = CitationAgent()
        orchestrator = LeadOrchestrator(
            lead_agent=lead,
            search_subagent=subagent,
            citation_agent=citation,
            run_store=run_store,
            report_service=report_service,
            runtime_settings=asdict(runtime.settings),
            auditor=CitationAuditor(runtime),
        )
        ai_search_service = AISearchService(runtime, search_service, scrape_service)
        return cls(
            runtime=runtime,
            orchestrator=orchestrator,
            ai_search_service=ai_search_service,
            run_store=run_store,
            scrape_service=scrape_service,
            scholarly_service=scholarly_service,
            search_service=search_service,
        )

    async def run(
        self,
        request: ResearchRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        with _bootstrap_run():
            return await self._orchestrator.run(request, progress_callback)

    def run_sync(
        self,
        request: ResearchRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        runner = get_async_runner()
        return runner.run(self.run(request, progress_callback))

    async def clarifying_questions(self, query: str) -> list[str]:
        with _bootstrap_run():
            return await self._orchestrator.clarifying_questions(query)

    async def resume(
        self,
        run_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        with _bootstrap_run():
            return await self._orchestrator.resume(run_id, progress_callback)

    async def rewrite_report(
        self,
        run_id: str,
        overrides: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchRunResult:
        with _bootstrap_run():
            return await self._orchestrator.rewrite_report(
                run_id, overrides, progress_callback
            )

    def list_runs(self) -> list[RunSummary]:
        return self._run_store.list_runs()

    def load_run(self, run_id: str) -> ResearchRunResult:
        return self._run_store.load_result(run_id)

    def report_path(self, run_id: str) -> Path:
        return self._run_store.report_path(run_id)

    async def stream(self, request: ResearchRequest) -> AsyncIterator[RunEvent]:
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
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
                # Sentinel guarantees the consumer wakes even when run() raises a
                # BaseException (e.g. CancelledError) that the except above skips.
                await queue.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await task
        if error is not None:
            raise error

    def inspect_run(self, run_id: str) -> dict[str, object]:
        """Describe a research run, or a single model call by its desk run id."""
        try:
            stored = self._run_store.load(run_id)
        except RunNotFoundError:
            return self._runtime.inspect_run(run_id)
        result = self._run_store.load_result(run_id)
        return {
            "exists": True,
            "run_id": run_id,
            "status": stored.status.get("status", "unknown"),
            "created_at": stored.status.get("created_at", ""),
            "updated_at": stored.status.get("updated_at", ""),
            "error": stored.status.get("error"),
            "input": stored.request.model_dump(mode="json"),
            "settings": self._run_store.settings(run_id),
            "output": None,
            "output_json": {"run_stats": result.run_stats},
            "events": self._run_store.events(run_id),
        }

    async def ai_search(
        self,
        query: str,
        max_results: int = 8,
        max_pages: int = 3,
        detail_level: DetailLevel = "standard",
    ) -> AISearchResult:
        with _bootstrap_run():
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

    async def aclose(self) -> None:
        for service in self._closables:
            await service.aclose()

    def close(self) -> None:
        if self._closables:
            get_async_runner().run(self.aclose())
