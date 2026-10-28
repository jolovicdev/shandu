from __future__ import annotations

import queue
import textwrap
import threading
from concurrent import futures
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr
from pydantic import ValidationError

from ...contracts import EFFORT_PRESETS, ResearchRequest, ResearchRunResult, RunEvent
from ...config import infer_api_key_env_name
from ...engine import ShanduEngine
from ...runtime import get_async_runner
from ...runtime.calls import ProviderRejectedError
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

_REJECTION_HINTS = {
    "key": "The model provider rejected the API key. Check API key and Key variable in Settings.",
    "model": "The model provider does not know this model. Check Model in Settings.",
    "credit": "The model provider account is out of credit. Add credit with the provider, then run again.",
}
_ERROR_HINTS = (
    ("RateLimitError", "The model provider is rate limiting requests. Wait a minute, then run again."),
    ("APIConnectionError", "Shandu could not reach the model provider. Check your connection."),
)


def _hint(message: str) -> str:
    return f"<p class='shandu-hint'>{escape(message)}</p>"


EFFORT_CHOICES = [
    ("Saved settings", "settings"),
    *(
        (
            f"{name.capitalize()}: {preset['max_iterations']} "
            f"iteration{'' if preset['max_iterations'] == 1 else 's'}",
            name,
        )
        for name, preset in EFFORT_PRESETS.items()
    ),
]
RESEARCHER_CHOICES = [
    ("Pipeline: faster, cheaper", "pipeline"),
    ("Agentic: follows links, slower", "agentic"),
]
LENGTH_CHOICES = [
    ("Saved settings", "settings"),
    ("Concise (1,200 words)", "concise"),
    ("Standard (2,200 words)", "standard"),
    ("Long (3,600 words)", "high"),
]
OPTIONS_NOTE = (
    "Low to Max replace Iterations, Parallel subagents, Results per query and Pages per "
    "task from Settings for this run. Research stops starting iterations once the budget "
    "is spent; the report and its citation check come after, so the total can end higher. "
    "The agentic researcher follows links to primary sources, so it takes longer and costs more."
)
# The labels a request field has on this page, for validation messages.
_OPTION_LABELS = {
    "include_domains": "Only these sites",
    "exclude_domains": "Skip these sites",
    "max_cost_usd": "Research budget",
}
# A run stops when its page goes away, so leaving mid-run asks first.
_WARN_ON_LEAVE_JS = """
() => {
  window.addEventListener("beforeunload", (event) => {
    if (document.querySelector(".shandu-strip.is-running")) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
}
"""
# Runs in the browser before the run starts and hands the inputs on. A
# Gradio update cannot close an accordion the user opened, so Options fold
# with a click, and the scroll waits for the page to settle after the fold.
_START_RUN_JS = """
(...values) => {
  document.querySelector(".shandu-options button.label-wrap.open")?.click();
  document.title = "Researching · Shandu";
  setTimeout(() => {
    document.getElementById("shandu-run")?.scrollIntoView({behavior: "smooth", block: "start"});
  }, 100);
  return values;
}
"""
_SHOW_REPORT_JS = """
() => {
  const title = document.querySelector(".shandu-strip.is-complete")
    ? document.querySelector(".shandu-doc-title")
    : null;
  document.title = title ? `Done: ${title.textContent} · Shandu` : "Shandu";
  title?.scrollIntoView({behavior: "smooth", block: "start"});
}
"""


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
        # The list's cell cuts text at 60 characters, mid-word.
        title = textwrap.shorten(report.title, 60, placeholder=" …")
        samples.append([title, " · ".join(facts), report.started_at, report.run_id])
    return samples


def _archive_strip(report: RecentReport) -> str:
    fields = [
        _field("from", report.started_at),
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
            gr.Markdown("Runs use these once they are saved.", elem_classes=["shandu-note"])
            model = gr.Textbox(label="Model", value=defaults.model)
            api_key_value = gr.Textbox(
                label="API key",
                type="password",
                info="Saved in ~/.shandu/config.json on this computer. Blank keeps the saved key.",
            )
            api_key_env = gr.Textbox(
                label="Key variable",
                value=defaults.api_key_env,
                placeholder=infer_api_key_env_name(defaults.model),
                info="The environment variable the key goes in. Blank uses the provider's usual one.",
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

        masthead = gr.HTML(_masthead(defaults.model), elem_classes=["shandu-plain"])

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
            with gr.Accordion("Options", open=False, elem_classes=["shandu-options"]):
                with gr.Row(elem_classes=["shandu-option-row"]):
                    effort = gr.Dropdown(choices=EFFORT_CHOICES, value="settings", label="Effort")
                    researcher = gr.Dropdown(
                        choices=RESEARCHER_CHOICES, value="pipeline", label="Researcher"
                    )
                    length = gr.Dropdown(
                        choices=LENGTH_CHOICES, value="settings", label="Report length"
                    )
                    max_cost = gr.Textbox(label="Research budget (USD)", placeholder="No limit")
                with gr.Row(elem_classes=["shandu-option-row"]):
                    include_sites = gr.Textbox(
                        label="Only these sites", placeholder="nature.com, who.int"
                    )
                    exclude_sites = gr.Textbox(
                        label="Skip these sites", placeholder="example.com"
                    )
                    since = gr.Textbox(label="Pages dated since", placeholder="YYYY-MM-DD")
                audit = gr.Checkbox(value=True, label="Check every citation against its sources")
                gr.Markdown(OPTIONS_NOTE, elem_classes=["shandu-note"])

        strip = gr.HTML(
            initial_state.strip_html(running=False),
            elem_id="shandu-run",
            elem_classes=["shandu-plain"],
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
            outputs=[save_message, masthead],
            concurrency_id="shandu_runtime",
            concurrency_limit=1,
        )
        # The page is built once per process, so each visit reads the
        # settings saved since.
        demo.load(
            fn=_load_settings,
            outputs=[
                model,
                api_key_env,
                temperature,
                max_tokens,
                max_iterations,
                parallelism,
                detail_level,
                depth_policy,
                max_results_per_query,
                max_pages_per_task,
                masthead,
            ],
        )
        demo.load(fn=None, js=_WARN_ON_LEAVE_JS)

        option_inputs: list[Any] = [
            effort,
            researcher,
            length,
            max_cost,
            audit,
            include_sites,
            exclude_sites,
            since,
        ]

        # Checking the options first shows a mistake as a message, where it
        # is seen wherever the page is scrolled, without starting a run.
        checked = gr.on(
            triggers=[run_button.click, query.submit],
            fn=_check_options,
            inputs=[query] + option_inputs,
        )
        run_event = checked.success(
            fn=_run_action,
            js=_START_RUN_JS,
            inputs=[query] + option_inputs,
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
        run_event.then(fn=_refresh_library, outputs=[library, first_run]).then(
            fn=None, js=_SHOW_REPORT_JS
        )
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
            ],
            concurrency_id="shandu_runtime",
            concurrency_limit=1,
        ).then(fn=_refresh_library, outputs=[library, first_run])

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
) -> tuple[str, str]:
    message = save_configuration(
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
    return message, _masthead(load_defaults().model)


def _load_settings() -> tuple[Any, ...]:
    defaults = load_defaults()
    return (
        defaults.model,
        gr.update(
            value=defaults.api_key_env, placeholder=infer_api_key_env_name(defaults.model)
        ),
        defaults.temperature,
        defaults.max_tokens,
        defaults.max_iterations,
        defaults.parallelism,
        resolved_detail_level(defaults.detail_level),
        resolved_depth_policy(defaults.depth_policy),
        defaults.max_results_per_query,
        defaults.max_pages_per_task,
        _masthead(defaults.model),
    )


def _request_from_options(
    query_value: str,
    effort_value: str,
    researcher_value: str,
    length_value: str,
    max_cost_value: str,
    audit_value: bool,
    include_value: str,
    exclude_value: str,
    since_value: str,
) -> ResearchRequest:
    """The request a run would make, on the saved settings and this run's
    options. A ValueError carries the message that names what to fix."""
    text = (query_value or "").strip()
    if not text:
        raise ValueError("Type a research question first.")

    since_text = (since_value or "").strip()
    try:
        since_day = date.fromisoformat(since_text) if since_text else None
    except ValueError:
        raise ValueError("Pages dated since takes a date like 2025-01-31.") from None

    cost_text = (max_cost_value or "").strip().removeprefix("$")
    try:
        max_cost_usd = float(cost_text) if cost_text else None
    except ValueError:
        max_cost_usd = -1.0
    if max_cost_usd is not None and not max_cost_usd > 0:
        raise ValueError(
            "Research budget takes an amount above zero, like 0.50. Leave it blank for no limit."
        )

    saved = load_defaults()
    preset = EFFORT_PRESETS.get(effort_value or "", {})
    try:
        return ResearchRequest(
            query=text,
            max_iterations=preset.get("max_iterations", saved.max_iterations),
            parallelism=preset.get("parallelism", saved.parallelism),
            detail_level=resolved_detail_level(
                saved.detail_level if length_value == "settings" else length_value
            ),
            depth_policy=resolved_depth_policy(saved.depth_policy),
            max_results_per_query=preset.get(
                "max_results_per_query", saved.max_results_per_query
            ),
            max_pages_per_task=preset.get("max_pages_per_task", saved.max_pages_per_task),
            max_searches_per_task=preset.get("max_searches_per_task", 6),
            effort=effort_value if effort_value in EFFORT_PRESETS else "medium",
            researcher="agentic" if researcher_value == "agentic" else "pipeline",
            audit=bool(audit_value),
            max_cost_usd=max_cost_usd,
            include_domains=[include_value or ""],
            exclude_domains=[exclude_value or ""],
            since=since_day,
        )
    except ValidationError as exc:
        problem = exc.errors()[0]
        name = str(problem["loc"][0]) if problem["loc"] else "options"
        label = _OPTION_LABELS.get(name, name.replace("_", " "))
        message = str(problem["msg"]).removeprefix("Value error, ")
        raise ValueError(f"Check {label}: {message}.") from None


def _check_options(*values: Any) -> None:
    try:
        _request_from_options(*values)
    except ValueError as exc:
        raise gr.Error(str(exc), print_exception=False) from None


def _run_action(
    query_value: str,
    effort_value: str = "settings",
    researcher_value: str = "pipeline",
    length_value: str = "settings",
    max_cost_value: str = "",
    audit_value: bool = True,
    include_value: str = "",
    exclude_value: str = "",
    since_value: str = "",
    http_request: gr.Request | None = None,
):
    try:
        request = _request_from_options(
            query_value,
            effort_value,
            researcher_value,
            length_value,
            max_cost_value,
            audit_value,
            include_value,
            exclude_value,
            since_value,
        )
    except ValueError as exc:
        yield (_hint(str(exc)), *(gr.skip(),) * 12)
        return

    state = GuiRunState(max_iterations=request.max_iterations)
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
        gr.Info("The report is ready below.")
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
    if isinstance(exc, ProviderRejectedError):
        return _REJECTION_HINTS[exc.kind]
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
    run_id = sample[-1] if sample else ""
    reports = {report.run_id: report for report in list_recent_reports()}
    report = reports.get(run_id)
    archived = GuiRunState()
    try:
        if report is None:
            raise FileNotFoundError(run_id)
        markdown = Path(report.path).read_text(encoding="utf-8")
    except OSError:
        archived.apply_error(f"Saved report {run_id[:8]} is no longer on disk.")
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
