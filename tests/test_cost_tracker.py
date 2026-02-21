from __future__ import annotations

from types import SimpleNamespace

from shandu.runtime.cost_tracker import CostSnapshot, CostTracker


def test_cost_tracker_accumulates_llm_completed_payload() -> None:
    tracker = CostTracker()
    tracker.handle_event(
        SimpleNamespace(
            type="llm.completed",
            payload={
                "cost": "0.0125",
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "total_tokens": 200,
            },
        )
    )
    tracker.handle_event(
        SimpleNamespace(
            type="llm.completed",
            payload={
                "prompt_tokens": 50,
                "completion_tokens": 30,
            },
        )
    )

    snap = tracker.snapshot()
    assert snap.llm_calls == 2
    assert snap.cost_events == 1
    assert round(snap.total_cost_usd, 4) == 0.0125
    assert snap.total_tokens == 280


def test_cost_tracker_delta_since_baseline() -> None:
    tracker = CostTracker()
    baseline = CostSnapshot()
    tracker.handle_event(
        SimpleNamespace(
            type="llm.completed",
            payload={"cost": 0.1, "total_tokens": 600},
        )
    )

    delta = tracker.delta_since(baseline)
    assert delta.llm_calls == 1
    assert delta.cost_events == 1
    assert round(delta.total_cost_usd, 3) == 0.1
    assert delta.total_tokens == 600
