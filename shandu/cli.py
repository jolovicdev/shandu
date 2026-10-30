from __future__ import annotations

import math
import os
import shutil
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import click
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from . import __version__
from .config import api_key_env_to_keep, config, infer_api_key_env_name
from .contracts import EFFORT_PRESETS, ReportLength, ResearchRequest, ResearchRunResult, RunEvent
from .engine import ShanduEngine
from .interfaces import DepthPolicy, DetailLevel
from .runtime import get_async_runner, reset_bootstrap
from .runtime.calls import ProviderRejectedError
from .services.run_store import RunNotFoundError
from .ui import ShanduUI

ui = ShanduUI()
console = ui.console
err_console = Console(theme=ui.theme, stderr=True)

_DETAIL_LEVELS: tuple[DetailLevel, ...] = ("concise", "standard", "high")
_REPORT_LENGTHS: tuple[ReportLength, ...] = ("auto", "concise", "standard", "high")
_DEPTH_POLICIES: tuple[DepthPolicy, ...] = ("adaptive", "fixed")


def _resolve_detail_level(value: str | None, fallback: DetailLevel) -> DetailLevel:
    if value is None:
        return fallback
    if value in _DETAIL_LEVELS:
        return cast(DetailLevel, value)
    return fallback


def _resolve_report_length(value: str | None) -> ReportLength:
    if value in _REPORT_LENGTHS:
        return cast(ReportLength, value)
    return "auto"


def _length_label(level: str) -> str:
    return "length fits the evidence" if level == "auto" else f"{level} detail"


def _resolve_depth_policy(value: str | None, fallback: DepthPolicy) -> DepthPolicy:
    if value is None:
        return fallback
    if value in _DEPTH_POLICIES:
        return cast(DepthPolicy, value)
    return fallback


def _bound(field: str, name: str) -> Any:
    for constraint in ResearchRequest.model_fields[field].metadata:
        if hasattr(constraint, name):
            return getattr(constraint, name)
    return None


def _count_range(field: str) -> click.IntRange:
    return click.IntRange(_bound(field, "ge"), _bound(field, "le"))


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _finite(ctx: click.Context, param: click.Parameter, value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise click.BadParameter("give an amount in USD, like 0.50.")
    return value


class _Day(click.ParamType):
    name = "YYYY-MM-DD"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> datetime:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            self.fail("Use YYYY-MM-DD, like 2025-01-31.", param, ctx)


_COMMAND_ORDER = (
    "run", "aisearch", "resume", "report", "list", "show", "inspect", "gui", "info", "configure", "clean",
)


class _OrderedGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(
            self.commands,
            key=lambda name: _COMMAND_ORDER.index(name) if name in _COMMAND_ORDER else len(_COMMAND_ORDER),
        )


@click.group(
    cls=_OrderedGroup,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.version_option(__version__, prog_name="shandu")
def cli() -> None:
    """Deep research from the terminal.

    Searches the web and papers, reads the pages, and writes a report with
    citations. Start with: shandu run "your question"
    """
    legacy = Path.cwd() / ".blackgeorge"
    storage = Path(str(config.get("runtime", "storage_dir")))
    if legacy.is_dir() and legacy.resolve() != storage:
        err_console.print(
            ui.warning(
                f"Found legacy storage in {legacy}; runs now live in {storage}."
            ),
            soft_wrap=True,
        )


@cli.command()
def info() -> None:
    """Show the model, key status, and storage in use."""
    api_key_env = config.get_api_key_env_name()
    key_in_env = bool(api_key_env and os.getenv(api_key_env))
    key_in_config = bool(str(config.get("api", "api_key", "")).strip())
    if api_key_env:
        key_rows = [
            ("API key env", api_key_env),
            ("API key", "set" if (key_in_env or key_in_config) else "not set"),
        ]
    else:
        key_rows = [
            ("API key env", "provider-managed"),
            ("API key", "provider-managed"),
        ]
    role_models = dict(config.get("api", "models", {}) or {})
    rows = [
        ("version", __version__),
        ("model", config.get("api", "model")),
        *((f"model ({role})", model) for role, model in sorted(role_models.items())),
        ("temperature", config.get("api", "temperature")),
        ("max tokens", config.get("api", "max_tokens")),
        *key_rows,
        ("search backend", config.get("search", "backend")),
        ("storage dir", config.get("runtime", "storage_dir")),
        ("default iterations", config.get("orchestration", "max_iterations")),
        ("default subagents", config.get("orchestration", "parallelism")),
    ]
    console.print(ui.key_values("Configuration", [(label, str(value)) for label, value in rows]))


@cli.command()
def configure() -> None:
    """Choose the model, save its API key, and set run defaults."""
    previous_model = str(config.get("api", "model"))
    model = click.prompt("Default model", default=previous_model)
    inferred_env_name = infer_api_key_env_name(model)
    saved_env_name = api_key_env_to_keep(
        str(config.get("api", "api_key_env", "")).strip(), previous_model, model
    )
    api_key_env = click.prompt(
        "API key env var name", default=saved_env_name or inferred_env_name
    ).strip()
    existing_key = str(config.get("api", "api_key", "")).strip()
    key_in_env = bool(os.getenv(api_key_env))
    if existing_key or key_in_env:
        console.print(
            f"[muted]{escape(api_key_env)} already available (env or saved config). "
            "Leave value empty to keep current key.[/]"
        )
    api_key = click.prompt(
        "API key value",
        default="",
        show_default=False,
        hide_input=True,
    ).strip()
    temperature = click.prompt(
        "Temperature", default=float(config.get("api", "temperature", 0.2)), type=float
    )
    max_tokens = click.prompt(
        "Max tokens", default=int(config.get("api", "max_tokens", 32768)), type=int
    )
    max_iterations = click.prompt(
        "Default max iterations",
        default=int(config.get("orchestration", "max_iterations", 2)),
        type=int,
    )
    parallelism = click.prompt(
        "Default subagents",
        default=int(config.get("orchestration", "parallelism", 3)),
        type=int,
    )

    config.set("api", "model", model)
    config.set("api", "api_key_env", api_key_env)
    if api_key:
        config.set("api", "api_key", api_key)
    config.set("api", "temperature", temperature)
    config.set("api", "max_tokens", max_tokens)
    config.set("orchestration", "max_iterations", max_iterations)
    config.set("orchestration", "parallelism", parallelism)
    config.save()
    reset_bootstrap()
    config.apply_provider_api_key()
    console.print(ui.success("Configuration saved."))


@cli.command("gui")
@click.option("--host", default="127.0.0.1", show_default=True, help="Address to listen on.")
@click.option("--port", default=7860, type=int, show_default=True, help="Port to listen on.")
@click.option("--share", is_flag=True, help="Create a public gradio share URL.")
@click.option("--auth", default=None, help="Basic auth credentials as user:pass.")
@click.option(
    "--browser/--no-browser", default=False, show_default=True, help="Open the GUI in a browser tab."
)
def gui_command(host: str, port: int, share: bool, auth: str | None, browser: bool) -> None:
    """Open the web GUI."""
    credentials = _parse_gui_auth(auth)
    if share and credentials is None:
        raise click.ClickException("--share requires --auth user:pass.")
    if credentials is None and host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            ui.warning(f"GUI on {host} without --auth is open to the network.")
        )

    try:
        from .ui.gradio_app import launch_gui
    except Exception as exc:
        raise click.ClickException(f"Failed to initialize GUI: {exc}") from None

    console.print(f"[brand]Launching Shandu GUI[/] [muted]http://{escape(host)}:{port}[/]")
    try:
        launch_gui(host=host, port=port, share=share, inbrowser=browser, auth=credentials)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from None
    except Exception as exc:
        if isinstance(exc, OSError) and "empty port" in str(exc):
            raise click.ClickException(
                f"Port {port} is in use. Shandu may already be running at http://{host}:{port}. "
                f"Use `shandu gui --port {port + 1}`."
            ) from None
        raise click.ClickException(f"GUI runtime error: {exc}") from None


def _parse_gui_auth(value: str | None) -> tuple[str, str] | None:
    if value is None:
        return None
    user, separator, password = value.partition(":")
    if not separator or not user or not password:
        raise click.ClickException("--auth must be user:pass.")
    return user, password


@cli.command("run")
@click.argument("query", nargs=-1, metavar="QUERY...")
@click.option(
    "--max-iterations",
    default=None,
    type=_count_range("max_iterations"),
    help="Iterations before writing the report.",
)
@click.option(
    "--parallelism",
    default=None,
    type=_count_range("parallelism"),
    help="Subagents that run at once in each iteration.",
)
@click.option(
    "--detail-level",
    default=None,
    show_default=_resolve_report_length(str(config.get("orchestration", "detail_level", "auto"))),
    type=click.Choice(_REPORT_LENGTHS),
    help="Report length. auto sizes it to the question and the evidence; concise, "
    "standard, and high also cap it at about 1,200, 2,200, or 3,600 words.",
)
@click.option(
    "--depth-policy",
    default=None,
    show_default=_resolve_depth_policy(str(config.get("orchestration", "depth_policy", "adaptive")), "adaptive"),
    type=click.Choice(["adaptive", "fixed"]),
    help="adaptive keeps searching while a sub-question lacks two credible sources, "
    "and stops once an iteration adds no finding or credible source to any sub-question; "
    "fixed lets the lead agent decide.",
)
@click.option(
    "--max-results-per-query",
    default=None,
    type=_count_range("max_results_per_query"),
    help="Search results ranked per query.",
)
@click.option(
    "--max-pages-per-task",
    default=None,
    type=_count_range("max_pages_per_task"),
    help="Pages each subagent reads.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save the report, or with --json-output the full result, to this file.",
)
@click.option("--json-output", is_flag=True, help="Emit the full result as JSON.")
@click.option(
    "--verbose",
    is_flag=True,
    help="Show every event with its raw metrics, and a recap of every stage when the run ends.",
)
@click.option(
    "--clarify", is_flag=True, help="Answer up to three clarifying questions first."
)
@click.option(
    "--include-domain", "include_domains", multiple=True, help="Only use this site; repeatable."
)
@click.option(
    "--exclude-domain", "exclude_domains", multiple=True, help="Never use this site; repeatable."
)
@click.option(
    "--since",
    type=_Day(),
    default=None,
    help="Skip pages dated before this day.",
)
@click.option(
    "--effort",
    default=None,
    show_default="medium",
    type=click.Choice(list(EFFORT_PRESETS)),
    help="Research budget preset; explicit budget flags override it.",
)
@click.option(
    "--researcher",
    default="pipeline",
    show_default=True,
    type=click.Choice(["pipeline", "agentic"]),
    help="How each subagent gathers evidence.",
)
@click.option("--no-audit", is_flag=True, help="Skip checking citations against sources.")
@click.option(
    "--max-cost",
    type=click.FloatRange(min=_bound("max_cost_usd", "gt"), min_open=True),
    callback=_finite,
    default=None,
    help="Start no new iteration once this many USD are spent, then write the report. "
    "Writing and the citation check come after, so the total can end higher.",
)
def run_command(
    query: tuple[str, ...],
    max_iterations: int | None,
    parallelism: int | None,
    detail_level: str | None,
    depth_policy: str | None,
    max_results_per_query: int | None,
    max_pages_per_task: int | None,
    output: str | None,
    json_output: bool,
    verbose: bool,
    clarify: bool,
    include_domains: tuple[str, ...],
    exclude_domains: tuple[str, ...],
    since: datetime | None,
    effort: str | None,
    researcher: str,
    no_audit: bool,
    max_cost: float | None,
) -> None:
    """Research a question and write a report with citations."""
    question = " ".join(query).strip()
    if not question:
        raise click.UsageError(
            'Give a question to research, like: shandu run "What caused the 2024 CrowdStrike outage?"'
        )
    diag = err_console if (json_output and not output) else console
    default_depth = _resolve_depth_policy(
        str(config.get("orchestration", "depth_policy", "adaptive")),
        "adaptive",
    )
    preset = EFFORT_PRESETS.get(effort or "", {})

    def budget(flag: int | None, key: str, fallback: int) -> int:
        if flag is not None:
            return flag
        if key in preset:
            return preset[key]
        return int(config.get("orchestration", key, fallback))

    try:
        request = ResearchRequest(
            query=question,
            max_iterations=budget(max_iterations, "max_iterations", 2),
            parallelism=budget(parallelism, "parallelism", 3),
            detail_level=_resolve_report_length(
                detail_level or str(config.get("orchestration", "detail_level", "auto"))
            ),
            depth_policy=_resolve_depth_policy(depth_policy, default_depth),
            max_results_per_query=budget(max_results_per_query, "max_results_per_query", 8),
            max_pages_per_task=budget(max_pages_per_task, "max_pages_per_task", 6),
            max_searches_per_task=preset.get("max_searches_per_task", 6),
            effort=cast(Literal["low", "medium", "high", "max"], effort or "medium"),
            include_domains=list(include_domains),
            exclude_domains=list(exclude_domains),
            since=since.date() if since else None,
            researcher=cast(Literal["pipeline", "agentic"], researcher),
            audit=not no_audit,
            max_cost_usd=max_cost,
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise click.ClickException(f"Invalid research options: {problems}") from exc

    engine = ShanduEngine.from_config()
    if clarify:
        try:
            clarifications = _ask_clarifications(engine, question)
        except ProviderRejectedError as exc:
            engine.close()
            raise click.ClickException(f"{exc}\n{_rejection_fix(exc)}") from None
        request = request.model_copy(update={"clarifications": clarifications})
    diag.print(
        ui.header(
            request.query,
            [
                str(config.get("api", "model")),
                _count(request.max_iterations, "iteration"),
                _count(request.parallelism, "subagent"),
                f"{request.researcher} researcher",
                _length_label(request.detail_level),
            ],
        )
    )
    result = _run_to_completion(
        engine,
        request,
        lambda on_event: engine.run(request, progress_callback=on_event),
        diag,
        verbose,
    )
    _deliver(engine, result, output, json_output, diag)


def _ask_clarifications(engine: ShanduEngine, query: str) -> list[str]:
    questions = get_async_runner().run(engine.clarifying_questions(query))
    answers: list[str] = []
    for question in questions:
        answer = click.prompt(question, default="", show_default=False, err=True).strip()
        if answer:
            answers.append(f"{question} {answer}")
    return answers


def _run_to_completion(
    engine: ShanduEngine,
    request: ResearchRequest,
    start: Callable[[Callable[[RunEvent], None]], Coroutine[Any, Any, ResearchRunResult]],
    diag: Console,
    verbose: bool = False,
    again: str = "resume",
) -> ResearchRunResult:
    snapshot = ui.new_snapshot(request, str(config.get("api", "model")))

    def on_event(event: RunEvent) -> None:
        snapshot.apply(event)
        ui.print_event(event, snapshot, diag, verbose)

    future = get_async_runner().submit(start(on_event))
    try:
        result = future.result()
    except KeyboardInterrupt:
        future.cancel()
        try:
            future.result()
        except Exception:
            pass
        _print_cancelled(diag, snapshot.run_id, again)
        raise SystemExit(130) from None
    except RunNotFoundError as exc:
        raise click.ClickException(f"Run {exc} not found.") from None
    except ProviderRejectedError as exc:
        steps = [str(exc), _rejection_fix(exc)]
        if snapshot.run_id != "pending":
            steps.append(_resume_hint(snapshot.run_id, again))
        raise click.ClickException("\n".join(steps)) from None
    finally:
        engine.close()

    if verbose:
        diag.print(ui.recap(snapshot))
    return result


def _deliver(
    engine: ShanduEngine,
    result: ResearchRunResult,
    output: str | None,
    json_output: bool,
    diag: Console,
) -> None:
    diag.print(ui.result_summary(result))

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if json_output:
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        else:
            path.write_text(result.report_markdown, encoding="utf-8")
        diag.print(ui.success(f"Output saved to {path}"), soft_wrap=True)
    elif json_output:
        console.print_json(result.model_dump_json(indent=2))
    else:
        diag.print(ui.document("Final Report", result.report_markdown))

    if not output:
        report_path = engine.report_path(result.run_id)
        if report_path.is_file():
            diag.print(ui.success(f"Report saved to {report_path}"), soft_wrap=True)


def _resolve_run_id(engine: ShanduEngine, run_id: str) -> str:
    """The run named by its full id or by the first characters `shandu list` shows.

    An id that matches no run comes back unchanged: `inspect` also looks up
    single model calls by their own ids, and the caller reports the rest as
    unknown.
    """
    run_id = run_id.strip()
    if not run_id:
        engine.close()
        raise click.UsageError("Give a run id. `shandu list` shows them.")
    matches = [run.run_id for run in engine.list_runs() if run.run_id.startswith(run_id)]
    if len(matches) > 1 and run_id not in matches:
        engine.close()
        shown = ", ".join(match[:8] for match in matches[:5])
        more = f" and {len(matches) - 5} more" if len(matches) > 5 else ""
        raise click.ClickException(
            f"Run id {run_id} matches {len(matches)} runs: {shown}{more}. Type more characters."
        )
    return matches[0] if len(matches) == 1 else run_id


def _no_such_run(run_id: str) -> click.ClickException:
    return click.ClickException(f"No run starts with {run_id}. See `shandu list`.")


def _resume_hint(run_id: str, again: str = "resume") -> str:
    return f"Continue with: shandu {again} {run_id[:8]}"


def _rejection_fix(exc: ProviderRejectedError) -> str:
    if exc.kind == "key":
        variable = config.get_api_key_env_name() or "the provider's key variable"
        return f"Set the key with `shandu configure` or in {variable}; `shandu info` shows what is in use."
    if exc.kind == "model":
        return "Check the model name with `shandu info`; `shandu configure` or SHANDU_MODEL changes it."
    return "Add credit with the model provider."


def _print_cancelled(diag: Console, run_id: str, again: str = "resume") -> None:
    # "pending" is a snapshot that has seen no event, so no run exists to resume.
    if run_id == "pending":
        diag.print("Run cancelled.")
        return
    diag.print(f"Run cancelled. Progress is saved as run {run_id[:8]}.")
    diag.print(_resume_hint(run_id, again))


def _load_request(engine: ShanduEngine, run_id: str) -> ResearchRequest:
    try:
        return engine.load_run(run_id).request
    except RunNotFoundError:
        engine.close()
        raise _no_such_run(run_id) from None


@cli.command("resume")
@click.argument("run_id")
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save the report, or with --json-output the full result, to this file.",
)
@click.option("--json-output", is_flag=True, help="Emit the full result as JSON.")
@click.option(
    "--verbose",
    is_flag=True,
    help="Show every event with its raw metrics, and a recap of every stage when the run ends.",
)
def resume_command(
    run_id: str, output: str | None, json_output: bool, verbose: bool
) -> None:
    """Continue a stopped run after its last completed iteration."""
    diag = err_console if (json_output and not output) else console
    engine = ShanduEngine.from_config()
    run_id = _resolve_run_id(engine, run_id)
    request = _load_request(engine, run_id)
    diag.print(
        ui.header(request.query, [f"resuming {run_id[:8]}", str(config.get("api", "model"))])
    )
    result = _run_to_completion(
        engine,
        request,
        lambda on_event: engine.resume(run_id, progress_callback=on_event),
        diag,
        verbose,
    )
    _deliver(engine, result, output, json_output, diag)


@cli.command("report", short_help="Rewrite a stored run's report without searching.")
@click.argument("run_id")
@click.option(
    "--detail-level",
    default=None,
    type=click.Choice(_REPORT_LENGTHS),
    help="Report length, as for run. Defaults to the run's own.",
)
@click.option("--model", default=None, help="Model for the report writer.")
@click.option("--no-audit", is_flag=True, help="Skip checking citations against sources.")
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save the report, or with --json-output the full result, to this file.",
)
@click.option("--json-output", is_flag=True, help="Emit the full result as JSON.")
def report_command(
    run_id: str,
    detail_level: str | None,
    model: str | None,
    no_audit: bool,
    output: str | None,
    json_output: bool,
) -> None:
    """Rewrite a run's report from its stored evidence, without new searches."""
    diag = err_console if (json_output and not output) else console
    engine = ShanduEngine.from_config(writer_model=model)
    run_id = _resolve_run_id(engine, run_id)
    request = _load_request(engine, run_id)
    overrides: dict[str, Any] = {}
    if detail_level is not None:
        overrides["detail_level"] = detail_level
    if no_audit:
        overrides["audit"] = False
    diag.print(
        ui.header(
            request.query,
            [
                f"rewriting report {run_id[:8]}",
                model or str(config.get("api", "model")),
                _length_label(detail_level or request.detail_level),
            ],
        )
    )
    result = _run_to_completion(
        engine,
        request,
        lambda on_event: engine.rewrite_report(run_id, overrides, progress_callback=on_event),
        diag,
        again="report",
    )
    _deliver(engine, result, output, json_output, diag)


@cli.command("list")
@click.option(
    "--limit",
    default=20,
    type=click.IntRange(min=1),
    show_default=True,
    help="How many runs to show, newest first.",
)
def list_command(limit: int) -> None:
    """List stored research runs, newest first."""
    engine = ShanduEngine.from_config()
    try:
        runs = engine.list_runs()[:limit]
    finally:
        engine.close()
    if not runs:
        console.print(ui.warning("No runs stored yet."))
        return
    console.print(ui.run_list(runs))


@cli.command("show")
@click.argument("run_id")
def show_command(run_id: str) -> None:
    """Print a stored run's summary and report."""
    engine = ShanduEngine.from_config()
    run_id = _resolve_run_id(engine, run_id)
    try:
        result = engine.load_run(run_id)
        status = next((run.status for run in engine.list_runs() if run.run_id == run_id), "unknown")
    except RunNotFoundError:
        raise _no_such_run(run_id) from None
    finally:
        engine.close()
    if result.report_markdown:
        console.print(ui.result_summary(result))
        console.print(ui.document("Final Report", result.report_markdown))
        return
    console.print(ui.key_values("Run summary", [("run", run_id[:8]), ("status", status)]))
    console.print(ui.warning("This run has no report yet."))
    if status in ("cancelled", "failed"):
        console.print(_resume_hint(run_id))


@cli.command("aisearch")
@click.argument("query", nargs=-1, metavar="QUERY...")
@click.option("--max-results", default=8, type=int, show_default=True, help="Search results to choose pages from.")
@click.option("--max-pages", default=3, type=int, show_default=True, help="Pages to read.")
@click.option(
    "--detail-level",
    default="standard",
    show_default=True,
    type=click.Choice(["concise", "standard", "high"]),
    help="Answer length.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save the answer, or with --json-output the full result, to this file.",
)
@click.option("--json-output", is_flag=True, help="Emit the full result as JSON.")
def ai_search_command(
    query: tuple[str, ...],
    max_results: int,
    max_pages: int,
    detail_level: str,
    output: str | None,
    json_output: bool,
) -> None:
    """Answer a question from one round of web search, with citations."""
    question = " ".join(query).strip()
    if not question:
        raise click.UsageError(
            'Give a question to answer, like: shandu aisearch "Which browsers support WebGPU?"'
        )
    diag = err_console if (json_output and not output) else console
    diag.print(
        ui.header(
            question,
            [str(config.get("api", "model")), "quick answer", f"{max_pages} pages"],
        )
    )
    engine = ShanduEngine.from_config()
    try:
        result = engine.ai_search_sync(
            query=question,
            max_results=max_results,
            max_pages=max_pages,
            detail_level=_resolve_detail_level(detail_level, "standard"),
        )
    except ProviderRejectedError as exc:
        raise click.ClickException(f"{exc}\n{_rejection_fix(exc)}") from None
    finally:
        engine.close()

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if json_output:
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        else:
            path.write_text(result.answer_markdown, encoding="utf-8")
        diag.print(ui.success(f"Output saved to {path}"), soft_wrap=True)
        return

    if json_output:
        console.print_json(result.model_dump_json(indent=2))
        return

    diag.print(ui.document("Answer", result.answer_markdown))
    diag.print(ui.sources(result))


@cli.command()
@click.argument("run_id")
def inspect(run_id: str) -> None:
    """Show a stored run's status, timestamps, event count, and stats."""
    engine = ShanduEngine.from_config()
    run_id = _resolve_run_id(engine, run_id)
    payload = engine.inspect_run(run_id)
    if not payload.get("exists"):
        raise _no_such_run(run_id)
    console.print(ui.inspect_summary(payload))


@cli.command()
@click.option("--force", is_flag=True, help="Delete without asking first.")
@click.option(
    "--all", "include_runs", is_flag=True, help="Also delete stored runs and their reports."
)
def clean(force: bool, include_runs: bool) -> None:
    """Delete runtime caches. Stored runs are kept unless --all is given."""
    runtime_dir = Path(
        str(config.get("runtime", "storage_dir"))
    ).expanduser().resolve()
    if not runtime_dir.exists():
        console.print(ui.warning("No runtime artifacts found."))
        return

    refusal = _refuse_clean_target(runtime_dir)
    if refusal is not None:
        raise click.ClickException(refusal)

    runs_dir = runtime_dir / "runs"
    run_count = sum(1 for entry in runs_dir.iterdir() if entry.is_dir()) if runs_dir.is_dir() else 0
    if include_runs:
        prompt = f"Delete {runtime_dir} and its {run_count} stored runs?"
    else:
        prompt = f"Delete caches in {runtime_dir}? Its {run_count} stored runs are kept."
    if not force and not click.confirm(prompt):
        console.print(ui.warning("Cleanup cancelled."))
        return

    if include_runs:
        shutil.rmtree(runtime_dir)
    else:
        for entry in runtime_dir.iterdir():
            if entry == runs_dir:
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    console.print(ui.success("Runtime artifacts removed."))


_CLEAN_MARKERS = ("blackgeorge.db", "memory.db", "runs")


def _refuse_clean_target(target: Path) -> str | None:
    if not target.is_dir():
        return f"Refusing to delete {target}: not a directory."
    home = Path(os.path.expanduser("~")).resolve()
    if target == home:
        return f"Refusing to delete home directory {target}."
    if target == Path(target.anchor):
        return f"Refusing to delete filesystem root {target}."
    if not any((target / marker).exists() for marker in _CLEAN_MARKERS):
        return (
            f"Refusing to delete {target}: no storage marker "
            f"({' or '.join(_CLEAN_MARKERS)}) found."
        )
    return None


if __name__ == "__main__":
    cli()
