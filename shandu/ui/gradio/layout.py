from __future__ import annotations

import queue
import threading
from concurrent import futures
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from ...contracts import ResearchRequest, ResearchRunResult, RunEvent
from ...config import infer_api_key_env_name
from ...engine import ShanduEngine
from ...runtime import get_async_runner
from ...services.report import RecentReport, list_recent_reports
from ...services.run_store import default_run_store
from .constants import DEPTH_POLICIES, DETAIL_LEVELS
from .settings import (
    load_defaults,
    resolved_depth_policy,
    resolved_detail_level,
    save_configuration,
)
from .state import GuiRunState, _field, format_cost

EXAMPLE_LABELS = [
    "Microplastics in the brain?",
    "Browser automation in 2026?",
    "Balkan jobs by 2035?",
]

EXAMPLES = [
    "What does current evidence say about microplastics crossing the blood-brain barrier?",
    "Compare open-source browser automation frameworks in 2026 on reliability and speed.",
    "How are Southeast European labor markets likely to shift by 2035, and on what assumptions?",
]


# Stop reaches a run through its session: Gradio's own cancel waits on the
# worker thread, which is blocked until the engine sends its next event.
_ACTIVE_RUNS: dict[str, threading.Event] = {}

_ERROR_HINTS = (
    (
        "AuthenticationError",
        "The model provider rejected the API key. Check Key env var and Key value in Settings.",
    ),
    ("RateLimitError", "The model provider is rate limiting requests. Wait a minute, then run again."),
    ("NotFoundError", "The model provider does not know this model. Check Model in Settings."),
    ("APIConnectionError", "Shandu could not reach the model provider. Check your connection."),
)

EMPTY_QUESTION = "<p class='shandu-hint'>Type a research question first.</p>"


def _masthead(model: str) -> str:
    model_line = f"<code>{escape(model)}</code>" if model else ""
    return (
        "<header class='shandu-masthead'>"
        "<div>"
        "<h1>Shandu</h1>"
        "<p>Ask a question. Agents search the web, read the pages, weigh each "
        "source, and write a report with citations.</p>"
        "</div>"
        f"{model_line}"
        "</header>"
    )


FIRST_RUN = """
<section class='shandu-first-run'>
  <h3>What a run does</h3>
  <ol>
    <li>Splits the question into separate lines of inquiry.</li>
    <li>Runs subagents in parallel, each searching and reading pages.</li>
    <li>Scores every source for credibility, authorship, and date.</li>
    <li>Writes the report and builds the references from what it read.</li>
  </ol>
</section>
"""


def _library_samples(reports: list[RecentReport]) -> list[list[str]]:
    samples: list[list[str]] = []
    for report in reports:
        facts = []
        if report.citations:
            facts.append(f"{report.citations} sources")
        if report.usd_spent:
            facts.append(format_cost(report.usd_spent))
        samples.append([report.title, " · ".join(facts), report.saved_at, report.run_id])
    return samples


def _archive_strip(report: RecentReport) -> str:
    fields = [
        _field("from", report.saved_at),
        _field("sources", report.citations or ""),
        _field("cost", format_cost(report.usd_spent) if report.usd_spent else ""),
        _field("run", report.run_id[:8], extra_class="run"),
    ]
    return (
        "<section class='shandu-strip is-complete'>"
        "<span class='stage'>Saved report</span>"
        f"{''.join(fields)}"
        "</section>"
    )


def build_gui() -> gr.Blocks:
    defaults = load_defaults()
    initial_state = GuiRunState()

    with gr.Blocks(
        title="Shandu",
        fill_width=True,
        elem_classes=["shandu-shell"],
    ) as demo:
        with gr.Sidebar(
            label="Settings",
            open=False,
            position="right",
            width=340,
            elem_classes=["shandu-settings"],
        ):
            gr.HTML("<h3>Settings</h3>", elem_classes=["shandu-plain"])
            model = gr.Textbox(label="Model", value=defaults.model)
            api_key_env = gr.Textbox(
                label="Key env var",
                value=defaults.api_key_env,
                placeholder=infer_api_key_env_name(defaults.model),
                info="Environment variable that holds the API key. Blank uses the provider's usual name.",
            )
            api_key_value = gr.Textbox(
                label="Key value",
                type="password",
                info="Saved to your local config. Blank keeps the current key.",
            )
            temperature = gr.Slider(
                0.0, 1.0, value=defaults.temperature, step=0.05, label="Temperature"
            )
            max_tokens = gr.Number(
                value=defaults.max_tokens, label="Max tokens", precision=0
            )
            max_iterations = gr.Slider(
                1, 8, value=defaults.max_iterations, step=1, label="Iterations"
            )
            parallelism = gr.Slider(
                1, 8, value=defaults.parallelism, step=1, label="Parallel subagents"
            )
            detail_level = gr.Dropdown(
                choices=list(DETAIL_LEVELS),
                value=defaults.detail_level
                if defaults.detail_level in DETAIL_LEVELS
                else "high",
                label="Detail",
                info="Report length: about 1,200, 2,200 or 3,600 words.",
            )
            depth_policy = gr.Dropdown(
                choices=list(DEPTH_POLICIES),
                value=defaults.depth_policy
                if defaults.depth_policy in DEPTH_POLICIES
                else "adaptive",
                label="Depth",
                info="Adaptive keeps searching while a sub-question lacks two credible "
                "sources. Fixed lets the lead agent decide when to stop.",
            )
            max_results_per_query = gr.Slider(
                1, 20, value=defaults.max_results_per_query, step=1, label="Results per query"
            )
            max_pages_per_task = gr.Slider(
                1, 10, value=defaults.max_pages_per_task, step=1, label="Pages per task"
            )
            save_button = gr.Button("Save", variant="secondary")
            save_message = gr.Markdown("", elem_classes=["shandu-note"])

        gr.HTML(_masthead(defaults.model), elem_classes=["shandu-plain"])

        with gr.Column(elem_classes=["shandu-composer"]):
            query = gr.Textbox(
                label="Research question",
                show_label=False,
                lines=3,
                placeholder="Ask a research question. The more specific the question, the better the report.",
                elem_classes=["shandu-question"],
            )
            with gr.Row(elem_classes=["shandu-actions"]):
                examples = gr.Examples(
                    examples=EXAMPLES,
                    example_labels=EXAMPLE_LABELS,
                    inputs=query,
                    label="",
                    elem_id="shandu-examples",
                )
                run_button = gr.Button("Research", variant="primary")
                stop_button = gr.Button("Stop", variant="secondary", visible=False)

        strip = gr.HTML(
            initial_state.strip_html(running=False), elem_classes=["shandu-plain"]
        )

        with gr.Column(elem_classes=["shandu-report"]):
            doc_title = gr.HTML(
                initial_state.title_html(), elem_classes=["shandu-plain"]
            )
            contents = gr.HTML(
                initial_state.contents_html(), elem_classes=["shandu-plain"]
            )
            with gr.Row(elem_classes=["shandu-report-head"]):
                download_report = gr.DownloadButton(
                    "Download markdown",
                    visible=False,
                    elem_classes=["shandu-download"],
                )
            report = gr.Markdown(
                initial_state.report_markdown, elem_classes=["shandu-prose"]
            )

        with gr.Accordion(
            "Activity",
            open=True,
            visible=False,
            elem_classes=["shandu-activity-toggle"],
        ) as activity_panel:
            with gr.Row(elem_classes=["shandu-activity"]):
                event_feed = gr.HTML(
                    initial_state.event_feed_html(), elem_classes=["shandu-plain"]
                )
                tasks = gr.HTML(
                    initial_state.task_board_html(), elem_classes=["shandu-plain"]
                )

        with gr.Accordion(
            "Run data", open=False, visible=False, elem_classes=["shandu-debug"]
        ) as debug_panel:
            payload = gr.JSON(value={}, show_label=False)

        recent = list_recent_reports()
        first_run = gr.HTML(FIRST_RUN, visible=not recent, elem_classes=["shandu-plain"])
        # Each sample carries its run id in a hidden last column. Keeping the
        # ids in a gr.State instead would turn on Gradio's heartbeat streams,
        # and this page opens enough of them to fill the browser's six
        # connections per host, which stalls every request after that.
        library = gr.Dataset(
            label="Recent reports",
            components=[
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
            ],
            samples=_library_samples(recent),
            samples_per_page=8,
            type="values",
            visible=bool(recent),
            elem_classes=["shandu-library"],
        )

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
            concurrency_id="shandu_runtime",
            concurrency_limit=1,
        )

        run_event = gr.on(
            triggers=[run_button.click, query.submit],
            fn=_run_action,
            inputs=[query] + config_inputs,
            outputs=[
                strip,
                doc_title,
                contents,
                event_feed,
                tasks,
                report,
                payload,
                download_report,
                debug_panel,
                activity_panel,
                stop_button,
                examples.dataset,
                run_button,
            ],
            concurrency_id="shandu_runtime",
            concurrency_limit=1,
        )
        run_event.then(fn=_refresh_library, outputs=[library, first_run])
        demo.load(fn=_refresh_library, outputs=[library, first_run])
        stop_button.click(fn=_stop_action, queue=False)
        library.click(
            fn=_open_report,
            inputs=[library],
            outputs=[
                strip,
                doc_title,
                contents,
                event_feed,
                tasks,
                report,
                payload,
                download_report,
                debug_panel,
                activity_panel,
                stop_button,
                examples.dataset,
                run_button,
                query,
            ],
            concurrency_id="shandu_runtime",
            concurrency_limit=1,
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
    http_request: gr.Request | None = None,
):
    text = query_value.strip()
    if not text:
        yield (EMPTY_QUESTION, *(gr.skip(),) * 12)
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
    state = GuiRunState(max_iterations=int(iter_value))
    state.stage = "bootstrap"
    yield _outputs(state=state, running=True, download_path=None)

    event_queue: queue.Queue[RunEvent | None] = queue.Queue()
    result_box: dict[str, Any] = {}
    error_box: dict[str, str] = {}
    stop_event = threading.Event()
    session = http_request.session_hash if http_request is not None else None
    if session:
        _ACTIVE_RUNS[session] = stop_event

    def on_event(event: RunEvent) -> None:
        event_queue.put(event)

    def run_worker() -> None:
        engine = None
        try:
            engine = ShanduEngine.from_config()
            future = get_async_runner().submit(
                engine.run(request, progress_callback=on_event)
            )
            while True:
                if stop_event.is_set():
                    future.cancel()
                try:
                    result_box["result"] = future.result(timeout=0.1)
                except futures.TimeoutError:
                    continue
                break
        except futures.CancelledError:
            result_box["stopped"] = True
        except Exception as exc:
            error_box["error"] = _error_message(exc)
            error_box["detail"] = f"{type(exc).__name__}: {exc}"
        finally:
            if engine is not None:
                engine.close()
            event_queue.put(None)

    threading.Thread(target=run_worker, daemon=True).start()

    try:
        while True:
            done = _apply_next_event_batch(event_queue=event_queue, state=state)
            if done:
                break
            yield _outputs(state=state, running=True, download_path=None)
    finally:
        stop_event.set()
        if session and _ACTIVE_RUNS.get(session) is stop_event:
            del _ACTIVE_RUNS[session]

    if result_box.get("stopped"):
        state.apply_stopped()
        yield _outputs(state=state, running=False, download_path=None)
        return

    if "error" in error_box:
        state.apply_error(error_box["error"], detail=error_box["detail"])
        yield _outputs(state=state, running=False, download_path=None)
        return

    result = result_box.get("result")
    if isinstance(result, ResearchRunResult):
        state.apply_result(result)
        state.stage = "complete"
        yield _outputs(
            state=state,
            running=False,
            download_path=str(default_run_store().report_path(result.run_id)),
        )
        return

    state.apply_error("Run did not return a valid result.")
    yield _outputs(state=state, running=False, download_path=None)


def _stop_action(http_request: gr.Request) -> None:
    stop_event = _ACTIVE_RUNS.get(http_request.session_hash)
    if stop_event is not None:
        stop_event.set()


def _error_message(exc: Exception) -> str:
    for name, message in _ERROR_HINTS:
        if type(exc).__name__ == name or f"litellm.{name}" in str(exc):
            return message
    return str(exc)


def _refresh_library() -> tuple[Any, Any]:
    recent = list_recent_reports()
    return (
        gr.update(samples=_library_samples(recent), visible=bool(recent)),
        gr.update(visible=not recent),
    )


def _open_report(sample: list[str]) -> tuple[Any, ...]:
    reports = {report.run_id: report for report in list_recent_reports()}
    report = reports.get(sample[-1]) if sample else None
    if report is None:
        return (gr.skip(),) * 14
    archived = GuiRunState()
    try:
        markdown = Path(report.path).read_text(encoding="utf-8")
    except OSError:
        archived.apply_error(f"Saved report {report.run_id} is no longer on disk.")
        return (
            archived.strip_html(running=False),
            "",
            "",
            "",
            "",
            "",
            {},
            gr.update(value=None, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.skip(),
        )
    archived.report_markdown = markdown
    return (
        _archive_strip(report),
        archived.title_html(),
        archived.contents_html(),
        "",
        "",
        archived.body_markdown(),
        {},
        _download_update(report.path),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        report.title,
    )


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
    # Accordion `open` updates from a streaming handler loop the Gradio client,
    # so visibility is the only state these panels take from a run.
    has_run = state.event_count > 0 or bool(state.errors)
    return (
        *state.render(running=running).as_tuple(),
        _download_update(download_path),
        gr.update(visible=has_run),
        gr.update(visible=has_run),
        gr.update(visible=running),
        gr.update(visible=not running),
        gr.update(visible=not running),
    )


def _download_update(path: str | None) -> Any:
    if path:
        return gr.update(value=path, visible=True)
    return gr.update(value=None, visible=False)
