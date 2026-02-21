from __future__ import annotations

from rich.console import Console

from shandu.contracts import ResearchRequest, RunEvent
from shandu.ui.rich_frontend import ShanduUI


def test_dashboard_renders_sections() -> None:
    console = Console(record=True, width=160)
    ui = ShanduUI(console=console)

    request = ResearchRequest(query="Distributed retrieval", max_iterations=2, parallelism=2)
    snapshot = ui.new_snapshot(request, model="deepseek/deepseek-chat")
    snapshot.apply(RunEvent(stage="plan", message="Plan ready", iteration=0, metrics={"tasks": 2}))
    snapshot.apply(RunEvent(stage="search", message="Search complete", iteration=0, metrics={"evidence": 4}))

    console.print(ui.dashboard(snapshot))
    output = console.export_text()

    assert "Control Plane" in output
    assert "Execution Timeline" in output
    assert "Run Metrics" in output
    assert "System Topology" in output


def test_event_line_renders_stage_and_task() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    event = RunEvent(
        stage="search",
        message="Task iter_1_task_1 started",
        iteration=0,
        metrics={"task_index": 1, "task_total": 4},
        payload={"task_id": "iter_1_task_1"},
    )
    line = ui.event_line(event)
    assert "SEARCH" in line.plain
    assert "iter_1_task_1" in line.plain


def test_event_line_renders_trace_query_and_url() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    event = RunEvent(
        stage="search",
        message="Task iter_1_task_1 query completed",
        iteration=0,
        metrics={"trace_type": "query_completed", "hits": 3},
        payload={
            "task_id": "iter_1_task_1",
            "query": "future of multimodal agents",
            "url": "https://example.com/article",
            "urls": ["https://example.com/article"],
        },
    )
    line = ui.event_line(event)
    assert "query_completed" in line.plain
    assert "future of multimodal agents" in line.plain
    assert "https://example.com/article" in line.plain
