from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from ...contracts import ResearchRequest, ResearchRunResult, RunEvent
from ...engine import ShanduEngine
from .constants import (
    CITATION_HEADERS,
    DEPTH_POLICIES,
    DETAIL_LEVELS,
    TIMELINE_HEADERS,
    TRACE_HEADERS,
)
from .settings import (
    load_defaults,
    persist_report_markdown,
    resolved_depth_policy,
    resolved_detail_level,
    save_configuration,
)
from .state import GuiRunState


def build_gui() -> gr.Blocks:
    defaults = load_defaults()
    initial_state = GuiRunState(query="")

    with gr.Blocks(
        title="Shandu",
        fill_width=True,
        elem_classes=["shandu-shell"],
    ) as demo:
        gr.HTML(
            """
            <header class="shandu-topbar">
              <div class="shandu-brand">
                <div class="shandu-brand-kicker">Research Control Room</div>
                <h1>Shandu</h1>
                <p>Multi-agent research runs with live search, scrape, citation, and cost telemetry.</p>
              </div>
              <div class="shandu-header-metrics">
                <div><span>Default Model</span><b>configured</b></div>
                <div><span>Mode</span><b>queued</b></div>
                <div><span>Output</span><b>report.md</b></div>
              </div>
            </header>
            """
        )

        with gr.Column(elem_classes=["shandu-run-panel"]):
            gr.Markdown("## Mission")
            query = gr.Textbox(
                label="Research query",
                lines=4,
                placeholder="Compare three cloud GPU providers for startups in 2026, including pricing, availability, and risk.",
                show_label=True,
            )
            with gr.Row(elem_classes=["shandu-run-actions"]):
                run_button = gr.Button("Start Research", variant="primary", size="lg")
            gr.Examples(
                examples=[
                    "Map likely labor-market shifts in Southeast Europe by 2035 and justify assumptions.",
                    "Compare open-source browser automation frameworks in 2026 for reliability and speed.",
                    "Which AI agent frameworks are strongest for enterprise workflow automation in 2026?",
                ],
                inputs=query,
                label="Quick starts",
            )

        with gr.Accordion(
            "Runtime settings", open=False, elem_classes=["shandu-config-panel"]
        ):
            model = gr.Textbox(label="Model", value=defaults.model)
            with gr.Row():
                api_key_env = gr.Textbox(
                    label="Key env var", value=defaults.api_key_env
                )
                api_key_value = gr.Textbox(label="Key value", type="password")
            with gr.Row():
                temperature = gr.Slider(
                    0.0,
                    1.0,
                    value=defaults.temperature,
                    step=0.05,
                    label="Temperature",
                )
                max_tokens = gr.Number(
                    value=defaults.max_tokens, label="Max tokens", precision=0
                )
            with gr.Row():
                max_iterations = gr.Slider(
                    1, 8, value=defaults.max_iterations, step=1, label="Iterations"
                )
                parallelism = gr.Slider(
                    1, 8, value=defaults.parallelism, step=1, label="Parallelism"
                )
            with gr.Row():
                detail_level = gr.Dropdown(
                    choices=list(DETAIL_LEVELS),
                    value=defaults.detail_level
                    if defaults.detail_level in DETAIL_LEVELS
                    else "high",
                    label="Detail",
                )
                depth_policy = gr.Dropdown(
                    choices=list(DEPTH_POLICIES),
                    value=defaults.depth_policy
                    if defaults.depth_policy in DEPTH_POLICIES
                    else "adaptive",
                    label="Depth",
                )
            with gr.Row():
                max_results_per_query = gr.Slider(
                    1,
                    20,
                    value=defaults.max_results_per_query,
                    step=1,
                    label="Results/query",
                )
                max_pages_per_task = gr.Slider(
                    1,
                    10,
                    value=defaults.max_pages_per_task,
                    step=1,
                    label="Pages/task",
                )
            save_button = gr.Button("Save Runtime", variant="secondary")
            save_message = gr.Markdown(
                "Loaded from local Shandu config.",
                elem_classes=["shandu-save-note"],
            )

        status = gr.HTML(
            initial_state.status_html(running=False),
            elem_classes=["shandu-html-reset"],
        )
        lane_view = gr.HTML(
            initial_state.lane_html(),
            elem_classes=["shandu-html-reset"],
        )

        with gr.Column(elem_classes=["shandu-main-stack"]):
            with gr.Column(elem_classes=["shandu-report-panel"]):
                with gr.Row(elem_classes=["shandu-section-bar"]):
                    gr.HTML(
                        "<h2 class='shandu-section-title'>Report</h2>",
                        elem_classes=["shandu-html-reset"],
                    )
                    download_report = gr.DownloadButton(
                        "Download report",
                        visible=False,
                        elem_classes=["shandu-download-action"],
                    )
                report = gr.Markdown(
                    initial_state.report_markdown,
                    elem_classes=["shandu-report-copy"],
                )
                with gr.Accordion(
                    "Citations", open=True, elem_classes=["shandu-citations-panel"]
                ):
                    citations = gr.Dataframe(
                        headers=CITATION_HEADERS,
                        datatype=["number", "str", "str", "str", "str"],
                        wrap=True,
                        interactive=False,
                    )
            with gr.Column(elem_classes=["shandu-telemetry-panel"]):
                gr.HTML(
                    "<h2 class='shandu-section-title'>Telemetry</h2>",
                    elem_classes=["shandu-html-reset"],
                )
                with gr.Row(elem_classes=["shandu-telemetry-grid"]):
                    with gr.Column(scale=5):
                        event_feed = gr.HTML(
                            initial_state.event_feed_html(),
                            elem_classes=["shandu-html-reset"],
                        )
                    with gr.Column(scale=7, elem_classes=["shandu-task-panel"]):
                        tasks = gr.HTML(
                            initial_state.task_board_html(),
                            elem_classes=["shandu-html-reset"],
                        )

        with gr.Tabs(elem_classes=["shandu-tabs"]):
            with gr.Tab("Timeline"):
                timeline = gr.Dataframe(
                    headers=TIMELINE_HEADERS,
                    datatype=["str", "str", "str", "str", "str"],
                    wrap=True,
                    interactive=False,
                    max_height=520,
                )
            with gr.Tab("Search + Scrape"):
                traces = gr.Dataframe(
                    headers=TRACE_HEADERS,
                    datatype=["str", "str", "str", "str", "str", "str"],
                    wrap=True,
                    interactive=False,
                    max_height=560,
                )
            with gr.Tab("Advanced"):
                with gr.Accordion(
                    "Debug payloads", open=False, elem_classes=["shandu-advanced"]
                ):
                    metrics = gr.JSON(label="Runtime Metrics", value={})
                    payload = gr.JSON(label="Run Payload", value={})

        config_inputs: list[Any] = [
            model,
            api_key_env,
            api_key_value,
            temperature,
            max_tokens,
            max_iterations,
            parallelism,
            detail_level,
            depth_policy,
            max_results_per_query,
            max_pages_per_task,
        ]

        save_button.click(
            fn=_save_action,
            inputs=config_inputs,
            outputs=[save_message],
        )

        run_button.click(
            fn=_run_action,
            inputs=[query] + config_inputs,
            outputs=[
                status,
                lane_view,
                event_feed,
                timeline,
                tasks,
                traces,
                metrics,
                report,
                citations,
                payload,
                download_report,
            ],
        )

    demo.queue(default_concurrency_limit=1, max_size=12)
    return demo


def _save_action(
    model_value: str,
    api_env_value: str,
    api_value: str,
    temp_value: float,
    token_value: float,
    iter_value: float,
    par_value: float,
    detail_value: str,
    depth_value: str,
    results_value: float,
    pages_value: float,
) -> str:
    return save_configuration(
        model=model_value,
        api_key_env=api_env_value,
        api_key_value=api_value,
        temperature=float(temp_value),
        max_tokens=token_value,
        max_iterations=int(iter_value),
        parallelism=int(par_value),
        detail_level=detail_value,
        depth_policy=depth_value,
        max_results_per_query=int(results_value),
        max_pages_per_task=int(pages_value),
    )


def _run_action(
    query_value: str,
    model_value: str,
    api_env_value: str,
    api_value: str,
    temp_value: float,
    token_value: float,
    iter_value: float,
    par_value: float,
    detail_value: str,
    depth_value: str,
    results_value: float,
    pages_value: float,
):
    text = query_value.strip()
    if not text:
        state = GuiRunState(query="")
        state.apply_error("Query is required.")
        yield _outputs(state=state, running=False, download_path=None)
        return

    save_configuration(
        model=model_value,
        api_key_env=api_env_value,
        api_key_value=api_value,
        temperature=float(temp_value),
        max_tokens=token_value,
        max_iterations=int(iter_value),
        parallelism=int(par_value),
        detail_level=detail_value,
        depth_policy=depth_value,
        max_results_per_query=int(results_value),
        max_pages_per_task=int(pages_value),
    )

    request = ResearchRequest(
        query=text,
        max_iterations=int(iter_value),
        parallelism=int(par_value),
        detail_level=resolved_detail_level(detail_value),
        depth_policy=resolved_depth_policy(depth_value),
        max_results_per_query=int(results_value),
        max_pages_per_task=int(pages_value),
    )
    state = GuiRunState(query=text)
    state.stage = "bootstrap"
    yield _outputs(state=state, running=True, download_path=None)

    event_queue: queue.Queue[RunEvent | None] = queue.Queue()
    result_box: dict[str, Any] = {}
    error_box: dict[str, str] = {}

    def on_event(event: RunEvent) -> None:
        event_queue.put(event)

    def run_worker() -> None:
        engine = None
        try:
            engine = ShanduEngine.from_config()
            result_box["result"] = engine.run_sync(request, progress_callback=on_event)
        except Exception as exc:
            error_box["error"] = str(exc)
        finally:
            if engine is not None:
                engine.close()
            event_queue.put(None)

    threading.Thread(target=run_worker, daemon=True).start()

    while True:
        done = _apply_next_event_batch(event_queue=event_queue, state=state)
        if done:
            break
        yield _outputs(state=state, running=True, download_path=None)

    if "error" in error_box:
        state.apply_error(error_box["error"])
        yield _outputs(state=state, running=False, download_path=None)
        return

    result = result_box.get("result")
    if isinstance(result, ResearchRunResult):
        state.apply_result(result)
        state.stage = "complete"
        yield _outputs(
            state=state,
            running=False,
            download_path=persist_report_markdown(
                result.run_id, result.report_markdown
            ),
        )
        return

    state.apply_error("Run did not return a valid result.")
    yield _outputs(state=state, running=False, download_path=None)


def _apply_next_event_batch(
    event_queue: queue.Queue[RunEvent | None], state: GuiRunState
) -> bool:
    event = event_queue.get()
    if event is None:
        return True
    state.apply_event(event)
    while True:
        try:
            queued_event = event_queue.get_nowait()
        except queue.Empty:
            return False
        if queued_event is None:
            return True
        state.apply_event(queued_event)


def _outputs(
    state: GuiRunState, running: bool, download_path: str | None
) -> tuple[Any, ...]:
    return (*state.render(running=running).as_tuple(), _download_update(download_path))


def _download_update(path: str | None) -> Any:
    if path:
        return gr.update(label=f"Download {Path(path).name}", value=path, visible=True)
    return gr.update(value=None, visible=False)
