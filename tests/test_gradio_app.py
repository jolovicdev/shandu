from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import gradio as gr

import shandu.ui.gradio.app as app_module
from shandu.config import config
from shandu.contracts import RunEvent
from shandu.ui.gradio import layout as layout_module
from shandu.ui.gradio_app import GuiRunState, _persist_report_markdown
from shandu.ui.gradio.layout import _outputs, build_gui


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


def test_gradio_model_calls_update_from_live_events() -> None:
    state = GuiRunState(query="q")
    state.apply_event(
        RunEvent(
            stage="plan",
            message="Iteration 1 plan ready",
            metrics={"agent_model_calls": 1},
        )
    )

    assert "Model calls" in state.lane_html()
    assert "<dd>1</dd>" in state.lane_html()


def test_persist_report_markdown_writes_export_file() -> None:
    path = _persist_report_markdown("run-xyz", "# Title\n\nBody")
    assert path is not None
    file_path = Path(path)
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8").startswith("# Title")


def test_gradio_run_output_contract_stays_aligned() -> None:
    state = GuiRunState(query="q")
    assert len(state.render(running=False).as_tuple()) == 10
    assert len(_outputs(state=state, running=False, download_path=None)) == 11


def test_run_generator_close_stops_worker(monkeypatch) -> None:
    started = threading.Event()
    cancelled = threading.Event()

    class BlockingEngine:
        @classmethod
        def from_config(cls) -> BlockingEngine:
            return cls()

        async def run(self, request, progress_callback=None) -> None:
            del request
            started.set()
            if progress_callback is not None:
                progress_callback(RunEvent(stage="bootstrap", message="start"))
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        layout_module, "save_configuration", lambda **kwargs: "saved"
    )
    monkeypatch.setattr(layout_module, "ShanduEngine", BlockingEngine)

    gen = layout_module._run_action(
        "q", "m", "", "", 0.2, 100, 1, 1, "high", "adaptive", 5, 3
    )
    next(gen)
    next(gen)
    assert started.is_set()
    gen.close()

    assert cancelled.wait(timeout=5)


def test_gui_stop_button_cancels_run() -> None:
    demo = build_gui()
    run_ids = {
        index
        for index, dep in demo.fns.items()
        if getattr(dep.fn, "__name__", "") == "_run_action"
    }
    assert len(run_ids) == 1
    run_id = next(iter(run_ids))
    stoppers = [
        dep for dep in demo.fns.values() if run_id in (dep.cancels or [])
    ]
    assert len(stoppers) == 1
    buttons = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Button) and block.value == "Stop"
    ]
    assert len(buttons) == 1


def test_save_and_run_share_runtime_concurrency_group() -> None:
    demo = build_gui()
    groups = {}
    for dep in demo.fns.values():
        name = getattr(dep.fn, "__name__", "")
        if name in ("_save_action", "_run_action"):
            groups[name] = (dep.concurrency_id, dep.concurrency_limit)

    assert groups["_save_action"] == ("shandu_runtime", 1)
    assert groups["_run_action"] == ("shandu_runtime", 1)


def test_launch_gui_allows_export_dir(monkeypatch, tmp_path) -> None:
    launched: dict = {}

    class FakeDemo:
        def launch(self, **kwargs) -> None:
            launched.update(kwargs)

    monkeypatch.setattr(app_module, "build_gui", lambda: FakeDemo())
    monkeypatch.setitem(
        config._config["runtime"], "storage_dir", str(tmp_path / "storage")
    )
    app_module.launch_gui()

    assert launched.get("allowed_paths") == [str(tmp_path / "storage" / "exports")]
