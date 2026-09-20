from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import cast

import click
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from .config import config, infer_api_key_env_name
from .contracts import ResearchRequest, RunEvent
from .engine import ShanduEngine
from .interfaces import DepthPolicy, DetailLevel
from .runtime import get_async_runner, reset_bootstrap
from .services import persist_report_markdown
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


def _run_query_line(query: str) -> str:
    return f"[brand]Running:[/] [accent]{escape(query)}[/]"


@click.group()
def cli() -> None:
    pass


@cli.command()
def info() -> None:
    ui.print_banner()
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
        ("Model", config.get("api", "model")),
        ("Temperature", config.get("api", "temperature")),
        ("Max Tokens", config.get("api", "max_tokens")),
        *key_rows,
        ("Storage Dir", config.get("runtime", "storage_dir")),
        ("Default Iterations", config.get("orchestration", "max_iterations")),
        ("Default Parallelism", config.get("orchestration", "parallelism")),
    ]
    table = ui.inspect_panel({"run_id": "config", "status": "active", "created_at": "-", "updated_at": "-", "events": []})
    console.print(table)
    for label, value in rows:
        console.print(f"[label]{label}:[/] [accent]{escape(str(value))}[/]")


@cli.command()
def configure() -> None:
    ui.print_banner()
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
    ui.print_banner()
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
) -> None:
    diag = err_console if (json_output and not output) else console
    ui.print_banner(diag)
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
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        raise click.ClickException(f"Invalid research options: {problems}") from exc

    engine = ShanduEngine.from_config()
    snapshot = ui.new_snapshot(request, str(config.get("api", "model")))

    def on_event(event: RunEvent) -> None:
        snapshot.apply(event)
        diag.print(ui.event_line(event))

    diag.print(_run_query_line(request.query))
    future = get_async_runner().submit(
        engine.run(request, progress_callback=on_event)
    )
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
    finally:
        engine.close()

    if verbose:
        diag.print(ui.dashboard(snapshot))

    diag.print(ui.result_panels(result))

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if json_output:
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        else:
            path.write_text(result.report_markdown, encoding="utf-8")
        diag.print(ui.success(f"Output saved to {path}"))
    elif json_output:
        console.print_json(result.model_dump_json(indent=2))
    else:
        diag.print(ui.markdown_panel("Final Report", result.report_markdown))

    if not output:
        export_path = persist_report_markdown(result.run_id, result.report_markdown)
        if export_path:
            diag.print(ui.success(f"Report saved to {export_path}"))


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
    ui.print_banner(diag)
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
        diag.print(ui.success(f"Output saved to {path}"))
        return

    if json_output:
        console.print_json(result.model_dump_json(indent=2))
        return

    diag.print(ui.markdown_panel("AISearch Answer", result.answer_markdown))
    diag.print(ui.ai_sources_panel(result))


@cli.command()
@click.argument("run_id")
def inspect(run_id: str) -> None:
    ui.print_banner()
    engine = ShanduEngine.from_config()
    payload = engine.inspect_run(run_id)
    if not payload.get("exists"):
        console.print(ui.warning(f"Run {run_id} not found."))
        return
    console.print(ui.inspect_panel(payload))


@cli.command()
@click.option("--force", is_flag=True)
def clean(force: bool) -> None:
    ui.print_banner()
    runtime_dir = Path(
        str(config.get("runtime", "storage_dir", ".blackgeorge"))
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
