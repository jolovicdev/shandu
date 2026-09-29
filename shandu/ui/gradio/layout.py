from __future__ import annotations

import queue
import threading
from concurrent import futures
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from ...contracts import ResearchRequest, ResearchRunResult, RunEvent
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
from .state import GuiRunState, format_cost

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


def _library_samples() -> list[list[str]]:
    samples: list[list[str]] = []
    for report in list_recent_reports(limit=6):
        facts = []
        if report.citations:
            facts.append(f"{report.citations} citations")
        if report.usd_spent:
            facts.append(format_cost(report.usd_spent))
        samples.append([report.title, " · ".join(facts), report.saved_at])
    return samples


def _archive_strip(report: RecentReport) -> str:
    fields = [f"<span class='field'>from <b>{escape(report.saved_at)}</b></span>"]
    if report.citations:
        fields.append(f"<span class='field'>citations <b>{report.citations}</b></span>")
    if report.usd_spent:
        fields.append(
            f"<span class='field'>cost <b>{format_cost(report.usd_spent)}</b></span>"
        )
    fields.append(
        f"<span class='field run'>run <b>{escape(report.run_id[:8])}</b></span>"
    )
    return (
        "<section class='shandu-strip is-complete'>"
        "<span class='stage'>Saved report</span>"
        f"{''.join(fields)}"
        "</section>"
    )


def build_gui() -> gr.Blocks:
    defaults = load_defaults()
    initial_state = GuiRunState(query="")

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
            api_key_env = gr.Textbox(label="Key env var", value=defaults.api_key_env)
            api_key_value = gr.Textbox(label="Key value", type="password")
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
            )
            depth_policy = gr.Dropdown(
                choices=list(DEPTH_POLICIES),
                value=defaults.depth_policy
                if defaults.depth_policy in DEPTH_POLICIES
                else "adaptive",
                label="Depth",
            )
            max_results_per_query = gr.Slider(
                1, 20, value=defaults.max_results_per_query, step=1, label="Results per query"
            )
            max_pages_per_task = gr.Slider(
                1, 10, value=defaults.max_pages_per_task, step=1, label="Pages per task"
            )
            save_button = gr.Button("Save", variant="secondary")
            save_message = gr.Markdown(
                "Saved to your local Shandu config.",
                elem_classes=["shandu-note"],
            )

        gr.HTML(_masthead(defaults.model), elem_classes=["shandu-plain"])

        with gr.Column(elem_classes=["shandu-composer"]):
            query = gr.Textbox(
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

        first_run = gr.HTML(
            FIRST_RUN, visible=not _library_samples(), elem_classes=["shandu-plain"]
        )
        library = gr.Dataset(
            label="Recent reports",
            components=[
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
                gr.Textbox(visible=False),
            ],
            samples=_library_samples(),
            samples_per_page=6,
            type="index",
            visible=bool(_library_samples()),
            elem_classes=["shandu-library"],
        )

        with gr.Accordion(
            "Run data", open=False, visible=False, elem_classes=["shandu-debug"]
        ) as debug_panel:
            payload = gr.JSON(value={}, show_label=False)

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

        run_event = run_button.click(
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
            ],
            concurrency_id="shandu_runtime",
            concurrency_limit=1,
        )
        run_event.then(fn=_refresh_library, outputs=[library, first_run])
        stop_button.click(fn=None, inputs=None, outputs=None, cancels=[run_event])
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
):
    text = query_value.strip()
    if not text:
        state = GuiRunState(query="")
        state.apply_error("A research question is required.")
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
    state = GuiRunState(query=text, max_iterations=int(iter_value))
    state.stage = "bootstrap"
    yield _outputs(state=state, running=True, download_path=None)

    event_queue: queue.Queue[RunEvent | None] = queue.Queue()
    result_box: dict[str, Any] = {}
    error_box: dict[str, str] = {}
    stop_event = threading.Event()

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
            error_box["error"] = "Run cancelled."
        except Exception as exc:
            error_box["error"] = str(exc)
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
            download_path=str(default_run_store().report_path(result.run_id)),
        )
        return

    state.apply_error("Run did not return a valid result.")
    yield _outputs(state=state, running=False, download_path=None)


def _refresh_library() -> tuple[Any, Any]:
    samples = _library_samples()
    return (
        gr.update(samples=samples, visible=bool(samples)),
        gr.update(visible=not samples),
    )


def _open_report(index: int) -> tuple[Any, ...]:
    reports = list_recent_reports(limit=6)
    if not 0 <= index < len(reports):
        return (gr.skip(),) * 13
    report = reports[index]
    archived = GuiRunState(query=report.title)
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
    )


def _download_update(path: str | None) -> Any:
    if path:
        return gr.update(value=path, visible=True)
    return gr.update(value=None, visible=False)
