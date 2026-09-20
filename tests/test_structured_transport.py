from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import litellm
from blackgeorge import Desk, Job, Worker
from pydantic import BaseModel

from shandu.runtime.cost_tracker import CostTracker


class _Answer(BaseModel):
    summary: str
    score: float


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage={"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    )


def _run(desk: Desk, worker: Worker, job: Job):
    return asyncio.run(desk.arun(worker, job))


def test_structured_call_forwards_generation_controls_and_meters(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion(json.dumps({"summary": "ok", "score": 0.9}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    tracker = CostTracker()
    desk = Desk(
        model="fake/model",
        temperature=0.2,
        max_tokens=16384,
        num_retries=2,
        storage_dir=str(tmp_path),
    )
    try:
        desk.event_bus.subscribe("llm.completed", tracker.handle_event)
        report = _run(
            desk, Worker(name="probe"), Job(input="answer", response_schema=_Answer)
        )
    finally:
        desk.close()

    assert report.status == "completed"
    assert isinstance(report.data, _Answer)
    assert len(calls) == 1
    assert calls[0].get("temperature") == 0.2
    assert calls[0].get("max_tokens") == 16384
    assert calls[0].get("num_retries") == 2
    snapshot = tracker.snapshot()
    assert snapshot.llm_calls == 1
    assert snapshot.total_tokens == 160


def test_structured_retry_path_forwards_controls_and_meters(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []
    script = ["not json", json.dumps({"summary": "recovered", "score": 0.5})]

    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion(script[min(len(calls) - 1, len(script) - 1)])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    tracker = CostTracker()
    desk = Desk(
        model="fake/model",
        temperature=0.2,
        max_tokens=16384,
        num_retries=2,
        structured_output_retries=3,
        storage_dir=str(tmp_path),
    )
    try:
        desk.event_bus.subscribe("llm.completed", tracker.handle_event)
        report = _run(
            desk, Worker(name="probe"), Job(input="answer", response_schema=_Answer)
        )
    finally:
        desk.close()

    assert report.status == "completed"
    assert isinstance(report.data, _Answer)
    assert len(calls) >= 2
    for kwargs in calls:
        assert kwargs.get("temperature") == 0.2
        assert kwargs.get("max_tokens") == 16384
        assert kwargs.get("num_retries") == 2
    snapshot = tracker.snapshot()
    assert snapshot.llm_calls >= 1
    assert snapshot.total_tokens >= 160
