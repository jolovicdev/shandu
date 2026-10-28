from __future__ import annotations

import asyncio
import inspect
import re
import threading
from datetime import date
from types import SimpleNamespace

import gradio as gr
import pytest

import shandu.ui.gradio.app as app_module
from shandu.config import config
from shandu.contracts import RunEvent
from shandu.runtime.calls import ProviderRejectedError
from shandu.ui.gradio import layout as layout_module
from shandu.ui.gradio_app import GuiRunState
from shandu.ui.gradio.layout import _outputs, build_gui
from shandu.ui.gradio.settings import GuiDefaults


def test_gradio_task_status_not_completed_on_trace_completed_message() -> None:
    state = GuiRunState()
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
    state = GuiRunState()
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
    state = GuiRunState()
    state.apply_event(
        RunEvent(
            stage="plan",
            message="Iteration 1 plan ready",
            metrics={"agent_model_calls": 1},
        )
    )

    strip = state.strip_html(running=True)
    assert "model calls" in strip
    assert "<b>1</b>" in strip


def test_gradio_run_output_contract_stays_aligned() -> None:
    state = GuiRunState()
    assert len(state.render(running=False).as_tuple()) == 7
    assert len(_outputs(state=state, running=False, download_path=None)) == 13


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

    monkeypatch.setattr(layout_module, "ShanduEngine", BlockingEngine)

    gen = layout_module._run_action("q")
    next(gen)
    next(gen)
    assert started.is_set()
    gen.close()

    assert cancelled.wait(timeout=5)


class _BlockingEngine:
    started = threading.Event()
    cancelled = threading.Event()

    @classmethod
    def from_config(cls) -> _BlockingEngine:
        return cls()

    async def run(self, request, progress_callback=None) -> None:
        del request
        if progress_callback is not None:
            progress_callback(RunEvent(stage="bootstrap", message="start"))
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def close(self) -> None:
        pass


def test_gui_stop_button_calls_the_stop_handler() -> None:
    demo = build_gui()
    stop_ids = {
        block._id
        for block in demo.blocks.values()
        if isinstance(block, gr.Button) and block.value == "Stop"
    }
    stoppers = [
        dep
        for dep in demo.fns.values()
        if getattr(dep.fn, "__name__", "") == "_stop_action"
        and {target for target, _ in dep.targets} & stop_ids
    ]
    assert len(stop_ids) == 1
    assert len(stoppers) == 1
    assert not stoppers[0].queue


def test_stop_ends_the_sessions_run_as_stopped(monkeypatch) -> None:
    monkeypatch.setattr(layout_module, "ShanduEngine", _BlockingEngine)
    _BlockingEngine.cancelled.clear()
    page = SimpleNamespace(session_hash="s1")

    gen = layout_module._run_action("q", http_request=page)
    next(gen)
    next(gen)
    layout_module._stop_action(page)

    assert _BlockingEngine.cancelled.wait(timeout=5)
    final = list(gen)[-1]
    assert "<span class='stage'>Stopped</span>" in final[0]
    assert "s1" not in layout_module._ACTIVE_RUNS


def test_rejected_api_key_reads_as_a_settings_problem(monkeypatch) -> None:
    class RejectingEngine(_BlockingEngine):
        async def run(self, request, progress_callback=None) -> None:
            raise ProviderRejectedError("key", "Your api key: ****3f9a is invalid")

    monkeypatch.setattr(layout_module, "ShanduEngine", RejectingEngine)

    final = list(layout_module._run_action("q"))[-1]

    assert "rejected the API key. Check API key and Key variable in Settings." in final[0]
    assert "ProviderRejectedError: The model provider rejected the API key" in final[0]


def test_error_hints_leave_other_errors_as_they_are() -> None:
    assert layout_module._error_message(FileNotFoundError("runs/a/report.md")) == "runs/a/report.md"


def test_empty_question_asks_for_one_without_a_failed_run() -> None:
    outputs = next(layout_module._run_action("  "))

    assert "Type a research question first." in outputs[0]
    assert len(outputs) == 13


_SAVED = GuiDefaults(
    model="m",
    api_key_env="",
    temperature=0.2,
    max_tokens=100,
    max_iterations=2,
    parallelism=3,
    detail_level="standard",
    depth_policy="adaptive",
    max_results_per_query=8,
    max_pages_per_task=6,
)


def _capture_request(monkeypatch, *args) -> object:
    captured = {}

    class CapturingEngine(_BlockingEngine):
        async def run(self, request, progress_callback=None) -> None:
            captured["request"] = request
            raise RuntimeError("captured")

    def refuse_to_save(**kwargs):
        raise AssertionError("a run must not save the settings")

    monkeypatch.setattr(layout_module, "save_configuration", refuse_to_save)
    monkeypatch.setattr(layout_module, "load_defaults", lambda: _SAVED)
    monkeypatch.setattr(layout_module, "ShanduEngine", CapturingEngine)
    list(layout_module._run_action("q", *args))
    return captured["request"]


def test_run_options_reach_the_request(monkeypatch) -> None:
    request = _capture_request(
        monkeypatch,
        "low",
        "agentic",
        "concise",
        "$0.5",
        False,
        "nature.com, https://www.who.int/news",
        "example.com",
        "2025-01-31",
    )

    assert (request.max_iterations, request.parallelism, request.max_pages_per_task) == (1, 2, 3)
    assert (request.max_results_per_query, request.max_searches_per_task) == (6, 3)
    assert (request.researcher, request.audit, request.max_cost_usd) == ("agentic", False, 0.5)
    assert (request.effort, request.detail_level) == ("low", "concise")
    assert request.include_domains == ["nature.com", "who.int"]
    assert request.exclude_domains == ["example.com"]
    assert request.since == date(2025, 1, 31)


def test_effort_from_settings_keeps_the_saved_settings(monkeypatch) -> None:
    request = _capture_request(monkeypatch, "settings", "pipeline", "settings", "", True, "", "", "")

    assert (request.max_iterations, request.parallelism, request.max_pages_per_task) == (2, 3, 6)
    assert (request.max_results_per_query, request.detail_level) == (8, "standard")
    assert (request.researcher, request.audit, request.max_cost_usd) == ("pipeline", True, None)


def test_a_malformed_since_date_asks_for_a_date() -> None:
    outputs = next(layout_module._run_action("q", since_value="31/01/2025"))

    assert "takes a date like 2025-01-31" in outputs[0]
    assert len(outputs) == 13


@pytest.mark.parametrize("amount", ["abc", "0"])
def test_a_max_cost_that_is_not_a_positive_amount_is_refused(amount: str) -> None:
    outputs = next(layout_module._run_action("q", max_cost_value=amount))

    assert "Research budget takes an amount above zero" in outputs[0]


def test_a_site_that_is_not_a_site_name_is_named_by_its_label() -> None:
    outputs = next(layout_module._run_action("q", include_value="nature"))

    assert "Check Only these sites: &#x27;nature&#x27; is not a site name" in outputs[0]


def test_the_check_before_a_run_stops_only_a_run_it_would_refuse() -> None:
    with pytest.raises(gr.Error, match="Type a research question first."):
        layout_module._check_options(" ", "settings", "pipeline", "settings", "", True, "", "", "")

    layout_module._check_options("q", "settings", "pipeline", "settings", "", True, "", "", "")


def test_run_button_sends_every_run_option() -> None:
    demo = build_gui()
    run = next(dep for dep in demo.fns.values() if getattr(dep.fn, "__name__", "") == "_run_action")
    handler_inputs = [
        name for name in inspect.signature(layout_module._run_action).parameters if name != "http_request"
    ]

    assert len(run.inputs) == len(handler_inputs)


def test_save_and_run_share_runtime_concurrency_group() -> None:
    demo = build_gui()
    groups = {}
    for dep in demo.fns.values():
        name = getattr(dep.fn, "__name__", "")
        if name in ("_save_action", "_run_action"):
            groups[name] = (dep.concurrency_id, dep.concurrency_limit)

    assert groups["_save_action"] == ("shandu_runtime", 1)
    assert groups["_run_action"] == ("shandu_runtime", 1)


def test_launch_gui_allows_runs_dir(monkeypatch, tmp_path) -> None:
    launched: dict = {}

    class FakeDemo:
        def launch(self, **kwargs) -> None:
            launched.update(kwargs)

    monkeypatch.setattr(app_module, "build_gui", lambda: FakeDemo())
    monkeypatch.setitem(
        config._config["runtime"], "storage_dir", str(tmp_path / "storage")
    )
    app_module.launch_gui()

    assert launched.get("allowed_paths") == [str(tmp_path / "storage" / "runs")]


def test_gradio_strip_shows_citation_audit_counts() -> None:
    from shandu.contracts import ResearchRequest, ResearchRunResult

    state = GuiRunState()
    state.apply_result(
        ResearchRunResult(
            run_id="r1",
            request=ResearchRequest(query="q"),
            report_markdown="# R",
            run_stats={
                "citation_audit": {"supported": 36, "partial": 0, "unsupported": 2, "unchecked": 1}
            },
        )
    )
    state.stage = "complete"

    strip = state.strip_html(running=False)

    assert "citation checks <b>36 supported · 2 removed · 1 unchecked</b>" in strip


def test_gradio_evidence_counts_every_task_while_running() -> None:
    state = GuiRunState()
    for task_id, evidence in (("t1", 3), ("t2", 4)):
        state.apply_event(
            RunEvent(
                stage="search",
                message=f"Task {task_id} completed",
                metrics={"evidence": evidence},
                payload={"task_id": task_id},
            )
        )

    assert "evidence <b>7</b>" in state.strip_html(running=True)


def test_gradio_stopped_run_marks_open_subagents_stopped() -> None:
    state = GuiRunState()
    for task_id in ("t1", "t2"):
        state.apply_event(
            RunEvent(stage="search", message=f"Task {task_id} started", payload={"task_id": task_id})
        )
    state.apply_event(
        RunEvent(stage="search", message="Task t1 completed", payload={"task_id": "t1"})
    )

    state.apply_stopped()

    assert [row["Status"] for row in state.task_rows.values()] == ["completed", "stopped"]
    assert "<span class='stage'>Stopped</span>" in state.strip_html(running=False)


def test_gradio_feed_lines_use_words_not_raw_fields() -> None:
    state = GuiRunState()
    state.apply_event(
        RunEvent(
            stage="search",
            message="Task iter_1_task_1 query completed",
            metrics={"trace_type": "query_completed", "hits": 6, "query": "abc"},
            payload={"task_id": "iter_1_task_1", "query": "abc"},
        )
    )

    feed = state.event_feed_html()

    assert "<span class='meta'>results 6 · “abc”</span>" in feed
    assert "iter_1_task_1" not in feed


def test_recent_report_click_opens_the_report_the_page_shows(monkeypatch, tmp_path) -> None:
    from shandu.contracts import ResearchRequest
    from shandu.services.run_store import RunStore

    monkeypatch.setitem(config._config["runtime"], "storage_dir", str(tmp_path))
    store = RunStore(tmp_path / "runs")
    for run_id, question in (("a", "first question"), ("b", "second question")):
        store.create(run_id, ResearchRequest(query=question), None)
        store.save_report(run_id, f"# {question}", [], {})
    shown = layout_module._library_samples(layout_module.list_recent_reports())
    store.create("c", ResearchRequest(query="newer question"), None)
    store.save_report("c", "# newer question", [], {})

    outputs = layout_module._open_report(shown[0])

    assert [sample[-1] for sample in shown] == ["b", "a"]
    assert "second question" in outputs[1]
    # The question box keeps whatever the user is typing.
    assert len(outputs) == 13
    assert re.fullmatch(r"\d{2} \w{3} \d{4}, \d{2}:\d{2}", shown[0][2])


def test_library_rows_show_the_start_and_cut_long_titles_at_a_word(monkeypatch, tmp_path) -> None:
    import json
    from datetime import datetime, timezone

    from shandu.contracts import ResearchRequest
    from shandu.services.run_store import RunStore

    monkeypatch.setitem(config._config["runtime"], "storage_dir", str(tmp_path))
    store = RunStore(tmp_path / "runs")
    question = "How do grid operators estimate inertia across interconnected regions in real time?"
    store.create("a", ResearchRequest(query=question), None)
    store.save_report("a", "# a", [], {})
    status_path = tmp_path / "runs" / "a" / "status.json"
    status = json.loads(status_path.read_text())
    status["created_at"] = "2026-01-02T03:04:00+00:00"
    status_path.write_text(json.dumps(status))

    (title, _, started, _) = layout_module._library_samples(layout_module.list_recent_reports())[0]

    assert started == datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc).astimezone().strftime(
        "%d %b %Y, %H:%M"
    )
    assert len(title) <= 60 and title.endswith(" …")
    assert question.startswith(title.removesuffix(" …"))


def test_a_report_gone_from_disk_says_so() -> None:
    outputs = layout_module._open_report(["t", "", "", "gone"])

    assert "Saved report gone is no longer on disk." in outputs[0]


def test_each_visit_reads_the_saved_settings(monkeypatch) -> None:
    monkeypatch.setattr(layout_module, "load_defaults", lambda: _SAVED)

    loaded = layout_module._load_settings()

    assert loaded[4:6] == (2, 3)
    assert "<code>m</code>" in loaded[-1]


def test_gui_needs_no_heartbeat_connection() -> None:
    assert build_gui().config["connect_heartbeat"] is False


def test_each_visit_loads_settings_and_each_run_is_checked_first() -> None:
    demo = build_gui()
    handlers = {getattr(dep.fn, "__name__", "") for dep in demo.fns.values()}

    assert {"_load_settings", "_check_options", "_run_action"} <= handlers
