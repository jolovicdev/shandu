from __future__ import annotations

from pathlib import Path

from shandu.contracts import RunEvent
from shandu.ui.gradio_app import GuiRunState, _persist_report_markdown


def test_gradio_task_status_not_completed_on_trace_completed_message() -> None:
    state = GuiRunState(query="q")
    state.apply_event(
        RunEvent(
            stage="search",
            message="Task t1 started",
            payload={"task_id": "t1", "focus": "f"},
        )
    )
    state.apply_event(
        RunEvent(
            stage="search",
            message="Task t1 query completed",
            metrics={"trace_type": "query_completed", "hits": 3},
            payload={"task_id": "t1", "query": "abc"},
        )
    )

    task = state.task_rows["t1"]
    assert task["Status"] == "running"


def test_gradio_task_status_completed_only_on_final_task_event() -> None:
    state = GuiRunState(query="q")
    state.apply_event(
        RunEvent(
            stage="search",
            message="Task t1 started",
            payload={"task_id": "t1"},
        )
    )
    state.apply_event(
        RunEvent(
            stage="search",
            message="Task t1 completed",
            payload={"task_id": "t1"},
        )
    )

    task = state.task_rows["t1"]
    assert task["Status"] == "completed"


def test_persist_report_markdown_writes_export_file() -> None:
    path = _persist_report_markdown("run-xyz", "# Title\n\nBody")
    assert path is not None
    file_path = Path(path)
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8").startswith("# Title")
