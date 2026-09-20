from __future__ import annotations

import asyncio

import pytest

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


class RaisingOrchestrator:
    async def run(self, request, progress_callback=None):
        del request
        if progress_callback is not None:
            await progress_callback(RunEvent(stage="bootstrap", message="start"))
        raise RuntimeError("boom")


class _Boom(BaseException):
    pass


class BaseErrorOrchestrator:
    async def run(self, request, progress_callback=None):
        del request, progress_callback
        raise _Boom()


def test_engine_stream_propagates_worker_error() -> None:
    engine = ShanduEngine(
        runtime=FakeRuntime(),
        orchestrator=RaisingOrchestrator(),
        ai_search_service=FakeAISearchService(),
    )

    async def collect():
        async for _event in engine.stream(ResearchRequest(query="q")):
            pass

    try:
        asyncio.run(collect())
        raise AssertionError("expected RuntimeError to propagate")
    except RuntimeError as exc:
        assert "boom" in str(exc)


class BlockingOrchestrator:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def run(self, request, progress_callback=None):
        del request
        if progress_callback is not None:
            await progress_callback(RunEvent(stage="bootstrap", message="start"))
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def test_engine_run_releases_bootstrap_on_error(monkeypatch) -> None:
    import shandu.runtime.bootstrap as boot

    calls: list[str] = []
    real_acquire = boot.acquire_bootstrap_run
    real_release = boot.release_bootstrap_run

    def acquire() -> None:
        calls.append("acquire")
        real_acquire()

    def release() -> None:
        calls.append("release")
        real_release()

    monkeypatch.setattr("shandu.engine.acquire_bootstrap_run", acquire)
    monkeypatch.setattr("shandu.engine.release_bootstrap_run", release)

    engine = ShanduEngine(
        runtime=FakeRuntime(),
        orchestrator=RaisingOrchestrator(),
        ai_search_service=FakeAISearchService(),
    )

    async def main() -> None:
        try:
            await engine.run(ResearchRequest(query="q"))
        except RuntimeError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected RuntimeError to propagate")

    before = boot._active_runs
    asyncio.run(main())

    assert calls == ["acquire", "release"]
    assert boot._active_runs == before


def test_engine_stream_cancels_worker_on_aclose() -> None:
    orchestrator = BlockingOrchestrator()
    engine = ShanduEngine(
        runtime=FakeRuntime(),
        orchestrator=orchestrator,
        ai_search_service=FakeAISearchService(),
    )

    async def main() -> None:
        stream = engine.stream(ResearchRequest(query="q"))
        first = await stream.__anext__()
        assert first.stage == "bootstrap"
        aclose_task = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(orchestrator.cancelled.wait(), timeout=5.0)
        await asyncio.wait_for(aclose_task, timeout=5.0)
        current = asyncio.current_task()
        unfinished = [
            task for task in asyncio.all_tasks() if task is not current and not task.done()
        ]
        assert unfinished == []

    asyncio.run(main())


def test_engine_stream_does_not_hang_on_base_exception() -> None:
    engine = ShanduEngine(
        runtime=FakeRuntime(),
        orchestrator=BaseErrorOrchestrator(),
        ai_search_service=FakeAISearchService(),
    )

    async def collect():
        async for _event in engine.stream(ResearchRequest(query="q")):
            pass

    async def main():
        with pytest.raises(_Boom):
            await asyncio.wait_for(collect(), timeout=5.0)

    asyncio.run(main())
