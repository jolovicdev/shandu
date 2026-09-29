from __future__ import annotations

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
from .config import config, infer_api_key_env_name
from .contracts import ResearchRequest, ResearchRunResult, RunEvent
from .engine import ShanduEngine
from .interfaces import DepthPolicy, DetailLevel
from .runtime import get_async_runner, reset_bootstrap
from .services.run_store import RunNotFoundError
from .ui import ShanduUI

ui = ShanduUI()
console = ui.console
err_console = Console(theme=ui.theme, stderr=True)

_DETAIL_LEVELS: tuple[DetailLevel, ...] = ("concise", "standard", "high")
_DEPTH_POLICIES: tuple[DepthPolicy, ...] = ("adaptive", "fixed")


def _resolve_detail_level(value: str | None, fallback: DetailLevel) -> DetailLevel:
    if value is None:
        return fallback
    if value in _DETAIL_LEVELS:
        return cast(DetailLevel, value)
    return fallback


def _resolve_depth_policy(value: str | None, fallback: DepthPolicy) -> DepthPolicy:
    if value is None:
        return fallback
    if value in _DEPTH_POLICIES:
        return cast(DepthPolicy, value)
    return fallback


@click.group()
def cli() -> None:
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
    api_key_env = config.get_api_key_env_name()
    key_in_env = bool(api_key_env and os.getenv(api_key_env))
    key_in_config = bool(str(config.get("api", "api_key", "")).strip())
    if api_key_env:
        key_rows = [
            ("API Key Env", api_key_env),
            ("API Key", "set" if (key_in_env or key_in_config) else "not set"),
        ]
    else:
        key_rows = [
            ("API Key Env", "provider-managed"),
            ("API Key", "provider-managed"),
        ]
    rows = [
        ("Version", __version__),
        ("Model", config.get("api", "model")),
        ("Temperature", config.get("api", "temperature")),
        ("Max Tokens", config.get("api", "max_tokens")),
        *key_rows,
        ("Search Backend", config.get("search", "backend")),
        ("Storage Dir", config.get("runtime", "storage_dir")),
        ("Default Iterations", config.get("orchestration", "max_iterations")),
        ("Default Parallelism", config.get("orchestration", "parallelism")),
    ]
    console.print(ui.key_values("Configuration", [(label, str(value)) for label, value in rows]))


@cli.command()
def configure() -> None:
    model = click.prompt("Default model", default=config.get("api", "model"))
    inferred_env_name = infer_api_key_env_name(model)
    api_key_env = click.prompt(
        "API key env var name",
        default=str(config.get("api", "api_key_env", "")).strip() or inferred_env_name,
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
        "Default parallelism",
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
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7860, type=int, show_default=True)
@click.option("--share", is_flag=True, help="Create a public gradio share URL.")
@click.option("--auth", default=None, help="Basic auth credentials as user:pass.")
@click.option("--browser/--no-browser", default=False, show_default=True)
def gui_command(host: str, port: int, share: bool, auth: str | None, browser: bool) -> None:
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
        console.print(ui.error(f"Failed to initialize GUI: {exc}"))
        return

    console.print(f"[brand]Launching Shandu GUI[/] [muted]http://{escape(host)}:{port}[/]")
    try:
        launch_gui(host=host, port=port, share=share, inbrowser=browser, auth=credentials)
    except RuntimeError as exc:
        console.print(ui.error(str(exc)))
    except Exception as exc:
        console.print(ui.error(f"GUI runtime error: {exc}"))


def _parse_gui_auth(value: str | None) -> tuple[str, str] | None:
    if value is None:
        return None
    user, separator, password = value.partition(":")
    if not separator or not user or not password:
        raise click.ClickException("--auth must be user:pass.")
    return user, password


@cli.command("run")
@click.argument("query")
@click.option("--max-iterations", default=None, type=int)
@click.option("--parallelism", default=None, type=int)
@click.option("--detail-level", default=None, type=click.Choice(["concise", "standard", "high"]))
@click.option("--depth-policy", default=None, type=click.Choice(["adaptive", "fixed"]))
@click.option("--max-results-per-query", default=None, type=int)
@click.option("--max-pages-per-task", default=None, type=int)
@click.option("--output", default=None)
@click.option("--json-output", is_flag=True)
@click.option("--verbose", is_flag=True)
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
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Skip pages dated before this day (YYYY-MM-DD).",
)
@click.option(
    "--researcher",
    default="pipeline",
    show_default=True,
    type=click.Choice(["pipeline", "agentic"]),
    help="How each task gathers evidence.",
)
def run_command(
    query: str,
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
    researcher: str,
) -> None:
    diag = err_console if (json_output and not output) else console
    default_detail = _resolve_detail_level(
        str(config.get("orchestration", "detail_level", "high")),
        "high",
    )
    default_depth = _resolve_depth_policy(
        str(config.get("orchestration", "depth_policy", "adaptive")),
        "adaptive",
    )
    try:
        request = ResearchRequest(
            query=query,
            max_iterations=max_iterations
            if max_iterations is not None
            else int(config.get("orchestration", "max_iterations", 2)),
            parallelism=parallelism
            if parallelism is not None
            else int(config.get("orchestration", "parallelism", 3)),
            detail_level=_resolve_detail_level(detail_level, default_detail),
            depth_policy=_resolve_depth_policy(depth_policy, default_depth),
            max_results_per_query=max_results_per_query
            if max_results_per_query is not None
            else int(config.get("orchestration", "max_results_per_query", 8)),
            max_pages_per_task=max_pages_per_task
            if max_pages_per_task is not None
            else int(config.get("orchestration", "max_pages_per_task", 6)),
            include_domains=list(include_domains),
            exclude_domains=list(exclude_domains),
            since=since.date() if since else None,
            researcher=cast(Literal["pipeline", "agentic"], researcher),
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise click.ClickException(f"Invalid research options: {problems}") from exc

    engine = ShanduEngine.from_config()
    if clarify:
        request = request.model_copy(
            update={"clarifications": _ask_clarifications(engine, query)}
        )
    diag.print(
        ui.header(
            request.query,
            [
                str(config.get("api", "model")),
                f"{request.max_iterations} iterations",
                f"{request.parallelism} tasks",
                f"{request.researcher} researcher",
                f"{request.detail_level} detail",
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
) -> ResearchRunResult:
    snapshot = ui.new_snapshot(request, str(config.get("api", "model")))

    def on_event(event: RunEvent) -> None:
        snapshot.apply(event)
        ui.print_event(event, diag)

    future = get_async_runner().submit(start(on_event))
    try:
        result = future.result()
    except KeyboardInterrupt:
        future.cancel()
        try:
            future.result()
        except Exception:
            pass
        diag.print("Run cancelled.")
        raise SystemExit(130) from None
    except RunNotFoundError as exc:
        raise click.ClickException(f"Run {exc} not found.") from None
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


def _load_request(engine: ShanduEngine, run_id: str) -> ResearchRequest:
    try:
        return engine.load_run(run_id).request
    except RunNotFoundError:
        engine.close()
        raise click.ClickException(f"Run {run_id} not found.") from None


@cli.command("resume")
@click.argument("run_id")
@click.option("--output", default=None)
@click.option("--json-output", is_flag=True)
@click.option("--verbose", is_flag=True)
def resume_command(
    run_id: str, output: str | None, json_output: bool, verbose: bool
) -> None:
    """Continue a stopped run after its last completed iteration."""
    diag = err_console if (json_output and not output) else console
    engine = ShanduEngine.from_config()
    request = _load_request(engine, run_id)
    diag.print(
        ui.header(request.query, [f"resuming {run_id}", str(config.get("api", "model"))])
    )
    result = _run_to_completion(
        engine,
        request,
        lambda on_event: engine.resume(run_id, progress_callback=on_event),
        diag,
        verbose,
    )
    _deliver(engine, result, output, json_output, diag)


@cli.command("report")
@click.argument("run_id")
@click.option("--detail-level", default=None, type=click.Choice(["concise", "standard", "high"]))
@click.option("--model", default=None, help="Model for the report writer.")
@click.option("--output", default=None)
@click.option("--json-output", is_flag=True)
def report_command(
    run_id: str,
    detail_level: str | None,
    model: str | None,
    output: str | None,
    json_output: bool,
) -> None:
    """Rewrite a run's report from its stored evidence, without new searches."""
    diag = err_console if (json_output and not output) else console
    engine = ShanduEngine.from_config(writer_model=model)
    request = _load_request(engine, run_id)
    level = cast(DetailLevel | None, detail_level)
    diag.print(
        ui.header(
            request.query,
            [
                f"rewriting report {run_id}",
                model or str(config.get("api", "model")),
                f"{level or request.detail_level} detail",
            ],
        )
    )
    result = _run_to_completion(
        engine,
        request,
        lambda on_event: engine.rewrite_report(
            run_id, level, progress_callback=on_event
        ),
        diag,
    )
    _deliver(engine, result, output, json_output, diag)


@cli.command("list")
@click.option("--limit", default=20, type=int, show_default=True)
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
    try:
        result = engine.load_run(run_id)
    except RunNotFoundError:
        raise click.ClickException(f"Run {run_id} not found.") from None
    finally:
        engine.close()
    console.print(ui.result_summary(result))
    if result.report_markdown:
        console.print(ui.document("Final Report", result.report_markdown))
    else:
        console.print(ui.warning("This run has no report yet."))


@cli.command("aisearch")
@click.argument("query")
@click.option("--max-results", default=8, type=int)
@click.option("--max-pages", default=3, type=int)
@click.option("--detail-level", default="standard", type=click.Choice(["concise", "standard", "high"]))
@click.option("--output", default=None)
@click.option("--json-output", is_flag=True)
def ai_search_command(
    query: str,
    max_results: int,
    max_pages: int,
    detail_level: str,
    output: str | None,
    json_output: bool,
) -> None:
    diag = err_console if (json_output and not output) else console
    diag.print(
        ui.header(
            query,
            [str(config.get("api", "model")), "quick answer", f"{max_pages} pages"],
        )
    )
    engine = ShanduEngine.from_config()
    try:
        result = engine.ai_search_sync(
            query=query,
            max_results=max_results,
            max_pages=max_pages,
            detail_level=_resolve_detail_level(detail_level, "standard"),
        )
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
    engine = ShanduEngine.from_config()
    payload = engine.inspect_run(run_id)
    if not payload.get("exists"):
        console.print(ui.warning(f"Run {run_id} not found."))
        return
    console.print(ui.inspect_summary(payload))


@cli.command()
@click.option("--force", is_flag=True)
def clean(force: bool) -> None:
    runtime_dir = Path(
        str(config.get("runtime", "storage_dir"))
    ).expanduser().resolve()
    if not runtime_dir.exists():
        console.print(ui.warning("No runtime artifacts found."))
        return

    refusal = _refuse_clean_target(runtime_dir)
    if refusal is not None:
        raise click.ClickException(refusal)

    if not force and not click.confirm(f"Delete {runtime_dir}?"):
        console.print(ui.warning("Cleanup cancelled."))
        return

    shutil.rmtree(runtime_dir)
    console.print(ui.success("Runtime artifacts removed."))


_CLEAN_MARKERS = ("blackgeorge.db", "memory.db")


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
