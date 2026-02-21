from __future__ import annotations

import asyncio

from shandu.contracts import AISearchResult, ResearchRequest, ResearchRunResult, RunEvent
from shandu.engine import ShanduEngine


class FakeOrchestrator:
    async def run(self, request, progress_callback=None):
        if progress_callback is not None:
            await progress_callback(RunEvent(stage="bootstrap", message="start"))
            await progress_callback(RunEvent(stage="complete", message="done", payload={"run_id": "r1"}))
        return ResearchRunResult(
            run_id="r1",
            request=request,
            report_markdown="# r1",
            citations=[],
            evidence=[],
            iteration_summaries=[],
            run_stats={"iterations": 1, "evidence_count": 0, "citation_count": 0},
        )


class FakeRuntime:
    def inspect_run(self, run_id):
        return {"exists": True, "run_id": run_id, "status": "completed", "events": []}


class FakeAISearchService:
    async def search(self, query, max_results=8, max_pages=3, detail_level="standard"):
        del max_results, max_pages, detail_level
        return AISearchResult(query=query, answer_markdown="# answer", sources=[])


def test_engine_stream_emits_events() -> None:
    engine = ShanduEngine(
        runtime=FakeRuntime(),
        orchestrator=FakeOrchestrator(),
        ai_search_service=FakeAISearchService(),
    )
    request = ResearchRequest(query="q")

    async def collect():
        events = []
        async for event in engine.stream(request):
            events.append(event)
        return events

    events = asyncio.run(collect())
    assert [event.stage for event in events] == ["bootstrap", "complete"]


def test_engine_inspect_passthrough() -> None:
    engine = ShanduEngine(
        runtime=FakeRuntime(),
        orchestrator=FakeOrchestrator(),
        ai_search_service=FakeAISearchService(),
    )
    payload = engine.inspect_run("abc")
    assert payload["run_id"] == "abc"


def test_engine_ai_search_passthrough() -> None:
    engine = ShanduEngine(
        runtime=FakeRuntime(),
        orchestrator=FakeOrchestrator(),
        ai_search_service=FakeAISearchService(),
    )
    result = engine.ai_search_sync("markets")
    assert result.query == "markets"
