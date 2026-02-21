from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import cast

import click

from .config import config, infer_api_key_env_name
from .contracts import ResearchRequest, RunEvent
from .engine import ShanduEngine
from .interfaces import DepthPolicy, DetailLevel
from .ui import ShanduUI

ui = ShanduUI()
console = ui.console

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
    ui.print_banner()


@cli.command()
def info() -> None:
    api_key_env = config.get_api_key_env_name()
    key_in_env = bool(os.getenv(api_key_env))
    key_in_config = bool(str(config.get("api", "api_key", "")).strip())
    rows = [
        ("Model", config.get("api", "model")),
        ("Temperature", config.get("api", "temperature")),
        ("Max Tokens", config.get("api", "max_tokens")),
        ("API Key Env", api_key_env),
        ("API Key", "set" if (key_in_env or key_in_config) else "not set"),
        ("Storage Dir", config.get("runtime", "storage_dir")),
        ("Default Iterations", config.get("orchestration", "max_iterations")),
        ("Default Parallelism", config.get("orchestration", "parallelism")),
    ]
    table = ui.inspect_panel({"run_id": "config", "status": "active", "created_at": "-", "updated_at": "-", "events": []})
    console.print(table)
    for label, value in rows:
        console.print(f"[label]{label}:[/] [accent]{value}[/]")


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
            f"[muted]{api_key_env} already available (env or saved config). "
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
        "Max tokens", default=int(config.get("api", "max_tokens", 8192)), type=int
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
    config.apply_provider_api_key()
    console.print(ui.success("Configuration saved."))


@cli.command("run")
@click.argument("query")
@click.option("--max-iterations", default=None, type=int)
@click.option("--parallelism", default=None, type=int)
@click.option("--detail-level", default=None, type=click.Choice(["concise", "standard", "high"]))
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
    max_results_per_query: int | None,
    max_pages_per_task: int | None,
    output: str | None,
    json_output: bool,
    verbose: bool,
) -> None:
    default_detail = _resolve_detail_level(
        str(config.get("orchestration", "detail_level", "high")),
        "high",
    )
    default_depth = _resolve_depth_policy(
        str(config.get("orchestration", "depth_policy", "adaptive")),
        "adaptive",
    )
    request = ResearchRequest(
        query=query,
        max_iterations=max_iterations
        if max_iterations is not None
        else int(config.get("orchestration", "max_iterations", 2)),
        parallelism=parallelism
        if parallelism is not None
        else int(config.get("orchestration", "parallelism", 3)),
        detail_level=_resolve_detail_level(detail_level, default_detail),
        depth_policy=default_depth,
        max_results_per_query=max_results_per_query
        if max_results_per_query is not None
        else int(config.get("orchestration", "max_results_per_query", 5)),
        max_pages_per_task=max_pages_per_task
        if max_pages_per_task is not None
        else int(config.get("orchestration", "max_pages_per_task", 3)),
    )

    engine = ShanduEngine.from_config()
    snapshot = ui.new_snapshot(request, str(config.get("api", "model")))

    def on_event(event: RunEvent) -> None:
        snapshot.apply(event)
        console.print(ui.event_line(event))

    console.print(f"[brand]Running:[/] [accent]{request.query}[/]")
    result = engine.run_sync(request, progress_callback=on_event)

    if verbose:
        console.print(ui.dashboard(snapshot))

    console.print(ui.result_panels(result))

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if json_output:
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        else:
            path.write_text(result.report_markdown, encoding="utf-8")
        console.print(ui.success(f"Output saved to {path}"))
    elif json_output:
        console.print_json(result.model_dump_json(indent=2))
    else:
        console.print(ui.markdown_panel("Final Report", result.report_markdown))


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
    engine = ShanduEngine.from_config()
    result = engine.ai_search_sync(
        query=query,
        max_results=max_results,
        max_pages=max_pages,
        detail_level=_resolve_detail_level(detail_level, "standard"),
    )

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if json_output:
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        else:
            path.write_text(result.answer_markdown, encoding="utf-8")
        console.print(ui.success(f"Output saved to {path}"))
        return

    if json_output:
        console.print_json(result.model_dump_json(indent=2))
        return

    console.print(ui.markdown_panel("AISearch Answer", result.answer_markdown))
    console.print(ui.ai_sources_panel(result))


@cli.command()
@click.argument("run_id")
def inspect(run_id: str) -> None:
    engine = ShanduEngine.from_config()
    payload = engine.inspect_run(run_id)
    if not payload.get("exists"):
        console.print(ui.warning(f"Run {run_id} not found."))
        return
    console.print(ui.inspect_panel(payload))


@cli.command()
@click.option("--force", is_flag=True)
def clean(force: bool) -> None:
    runtime_dir = Path(str(config.get("runtime", "storage_dir", ".blackgeorge")))
    config_dir = Path(os.path.expanduser("~/.shandu/cache"))

    targets = [runtime_dir, config_dir]
    existing = [str(path) for path in targets if path.exists()]
    if not existing:
        console.print(ui.warning("No runtime artifacts found."))
        return

    if not force and not click.confirm(f"Delete {', '.join(existing)}?"):
        console.print(ui.warning("Cleanup cancelled."))
        return

    for path in targets:
        if path.exists():
            shutil.rmtree(path)

    console.print(ui.success("Runtime artifacts removed."))


if __name__ == "__main__":
    cli()
