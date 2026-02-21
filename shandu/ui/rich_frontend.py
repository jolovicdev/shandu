from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from textwrap import shorten

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ..contracts import AISearchResult, ResearchRequest, ResearchRunResult, RunEvent


@dataclass
class RunSnapshot:
    request: ResearchRequest
    model: str
    run_id: str = "pending"
    current_stage: str = "bootstrap"
    current_message: str = "Waiting"
    iteration: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    events: list[RunEvent] = field(default_factory=list)

    def apply(self, event: RunEvent) -> None:
        self.events.append(event)
        self.current_stage = event.stage
        self.current_message = event.message
        self.iteration = event.iteration or self.iteration
        if event.metrics:
            self.metrics.update(event.metrics)
        if event.payload.get("run_id"):
            self.run_id = str(event.payload["run_id"])


class ShanduUI:
    def __init__(self, console: Console | None = None) -> None:
        theme = Theme(
            {
                "brand": "bold #10b981",
                "accent": "bold #0ea5e9",
                "muted": "#94a3b8",
                "ok": "bold #22c55e",
                "warn": "bold #f59e0b",
                "danger": "bold #ef4444",
                "label": "bold #14b8a6",
                "panel": "#0f766e",
                "title": "bold #34d399",
            }
        )
        self.theme = theme
        if console is None:
            self.console = Console(theme=theme)
        else:
            console.push_theme(theme)
            self.console = console

    def print_banner(self) -> None:
        top = Text(" SHANDU V3 ", style="bold black on #10b981")
        sub = Text("LeadResearcher · Subagents · Memory · CitationAgent", style="muted")
        self.console.print(
            Panel(
                Group(top, sub),
                border_style="panel",
                box=box.HEAVY,
                padding=(1, 2),
            )
        )

    def new_snapshot(self, request: ResearchRequest, model: str) -> RunSnapshot:
        return RunSnapshot(request=request, model=model)

    def dashboard(self, snapshot: RunSnapshot) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=6),
            Layout(name="body"),
            Layout(name="footer", size=10),
        )
        layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))
        layout["footer"].split_row(Layout(name="footer_left", ratio=2), Layout(name="footer_right", ratio=3))

        header_table = Table.grid(padding=(0, 1))
        header_table.add_column(style="label", no_wrap=True)
        header_table.add_column(style="accent")
        header_table.add_row("Run ID", snapshot.run_id)
        header_table.add_row("Stage", snapshot.current_stage)
        header_table.add_row("Message", snapshot.current_message)
        header_table.add_row("Model", snapshot.model)
        header_table.add_row("Iteration", str(snapshot.iteration + 1))

        layout["header"].update(
            Panel(header_table, title="Control Plane", border_style="panel", box=box.ROUNDED)
        )

        task_table = Table(box=box.SIMPLE_HEAVY, border_style="panel")
        task_table.add_column("#", style="muted", width=4)
        task_table.add_column("Stage", style="title", width=12)
        task_table.add_column("Task", style="label", width=16)
        task_table.add_column("Trace", style="muted", width=16)
        task_table.add_column("Message", style="accent")
        for idx, event in enumerate(snapshot.events[-10:], start=1):
            task_id = str(event.payload.get("task_id", "")).strip()
            trace_type = str(event.metrics.get("trace_type", "")).strip()
            message = event.message
            query = str(event.payload.get("query", "") or event.metrics.get("query", "")).strip()
            if query:
                message = f"{message} | q={shorten(query, width=42, placeholder='...')}"
            url = str(event.payload.get("url", "")).strip()
            if url:
                message = f"{message} | {shorten(url, width=52, placeholder='...')}"
            task_table.add_row(str(idx), event.stage, task_id, trace_type, message)
        if task_table.row_count == 0:
            task_table.add_row("-", "bootstrap", "-", "-", "No events yet")

        layout["left"].update(
            Panel(task_table, title="Execution Timeline", border_style="panel", box=box.ROUNDED)
        )

        metrics_table = Table(box=box.SIMPLE_HEAVY, border_style="panel")
        metrics_table.add_column("Metric", style="label", no_wrap=True)
        metrics_table.add_column("Value", style="accent")
        metrics_table.add_row("Query", snapshot.request.query)
        metrics_table.add_row("Max Iterations", str(snapshot.request.max_iterations))
        metrics_table.add_row("Parallelism", str(snapshot.request.parallelism))
        metrics_table.add_row("Detail", snapshot.request.detail_level)
        for key, value in snapshot.metrics.items():
            metrics_table.add_row(str(key), str(value))

        layout["right"].update(
            Panel(metrics_table, title="Run Metrics", border_style="panel", box=box.ROUNDED)
        )

        footer_notes = Table(box=box.SIMPLE)
        footer_notes.add_column(style="muted")
        footer_notes.add_row("1. User query enters LeadResearcher.")
        footer_notes.add_row("2. LeadResearcher plans and fans out subagent tasks.")
        footer_notes.add_row("3. Subagents emit query/scrape/extract traces.")
        footer_notes.add_row("4. Lead synthesizes loop and decides continue/exit.")
        footer_notes.add_row("5. CitationAgent normalizes references.")

        trace_table = Table(box=box.SIMPLE_HEAVY, border_style="panel")
        trace_table.add_column("Task", style="label", width=16)
        trace_table.add_column("Trace", style="muted", width=16)
        trace_table.add_column("Details", style="accent")
        for event in reversed(snapshot.events):
            trace_type = str(event.metrics.get("trace_type", "")).strip()
            if not trace_type:
                continue
            task_id = str(event.payload.get("task_id", "")).strip()
            query = str(event.payload.get("query", "") or event.metrics.get("query", "")).strip()
            url = str(event.payload.get("url", "")).strip()
            details = query or url or event.message
            trace_table.add_row(task_id, trace_type, shorten(details, width=70, placeholder="..."))
            if trace_table.row_count >= 6:
                break
        if trace_table.row_count == 0:
            trace_table.add_row("-", "-", "No trace events yet")

        layout["footer_left"].update(
            Panel(footer_notes, title="System Topology", border_style="panel", box=box.ROUNDED)
        )
        layout["footer_right"].update(
            Panel(trace_table, title="Subagent Trace Feed", border_style="panel", box=box.ROUNDED)
        )

        return layout

    def result_panels(self, result: ResearchRunResult) -> Columns:
        summary = Table.grid(padding=(0, 1))
        summary.add_column(style="label")
        summary.add_column(style="accent")
        summary.add_row("Run ID", result.run_id)
        summary.add_row("Iterations", str(result.run_stats.get("iterations", 0)))
        summary.add_row("Evidence", str(result.run_stats.get("evidence_count", 0)))
        summary.add_row("Citations", str(result.run_stats.get("citation_count", 0)))
        summary.add_row("Elapsed", f"{result.run_stats.get('elapsed_seconds', 0)}s")
        model_calls = result.run_stats.get("agent_model_calls")
        if isinstance(model_calls, int) and model_calls > 0:
            summary.add_row("Model Calls", str(model_calls))
        metered_calls = result.run_stats.get("metered_calls", result.run_stats.get("llm_calls"))
        coverage = str(result.run_stats.get("cost_coverage", "")).strip()
        if coverage not in {"partial", "full"}:
            if isinstance(metered_calls, int) and metered_calls > 0 and isinstance(model_calls, int) and model_calls > 0:
                coverage = "partial" if metered_calls < model_calls else "full"
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                if coverage == "partial":
                    summary.add_row("Cost Coverage", f"partial ({metered_calls}/{model_calls})")
                else:
                    summary.add_row("Cost Coverage", f"full ({metered_calls}/{model_calls})")
            else:
                summary.add_row("Metered Calls", str(metered_calls))
        llm_tokens = result.run_stats.get("llm_tokens")
        if isinstance(llm_tokens, int) and llm_tokens > 0:
            summary.add_row("LLM Tokens", str(llm_tokens))
        usd_spent = result.run_stats.get("usd_spent")
        if isinstance(usd_spent, (int, float)) and float(usd_spent) > 0:
            if coverage == "partial":
                summary.add_row("Metered Cost", f"${float(usd_spent):.6f}")
            else:
                summary.add_row("USD Spent", f"${float(usd_spent):.6f}")

        citations = Table(box=box.SIMPLE, border_style="panel")
        citations.add_column("#", style="muted", width=4)
        citations.add_column("Publisher", style="label")
        citations.add_column("Title", style="accent")
        for item in result.citations[:8]:
            citations.add_row(str(item.citation_id), item.publisher, item.title)
        if citations.row_count == 0:
            citations.add_row("-", "none", "No citations")

        return Columns(
            [
                Panel(summary, title="Run Summary", border_style="panel", box=box.ROUNDED),
                Panel(citations, title="Citation Ledger", border_style="panel", box=box.ROUNDED),
            ],
            expand=True,
        )

    def markdown_panel(self, title: str, content: str) -> Panel:
        return Panel(
            Markdown(content),
            title=title,
            border_style="panel",
            box=box.ROUNDED,
        )

    def inspect_panel(self, payload: dict[str, object]) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, border_style="panel")
        table.add_column("Field", style="label")
        table.add_column("Value", style="accent")
        for key in ["run_id", "status", "created_at", "updated_at"]:
            table.add_row(key, str(payload.get(key, "")))
        events = payload.get("events", [])
        table.add_row("events", str(len(events) if isinstance(events, list) else 0))
        return Panel(table, title="Run Inspection", border_style="panel", box=box.ROUNDED)

    def ai_sources_panel(self, result: AISearchResult) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, border_style="panel")
        table.add_column("#", style="muted", width=4)
        table.add_column("Title", style="label")
        table.add_column("URL", style="accent")
        for idx, source in enumerate(result.sources[:10], start=1):
            table.add_row(str(idx), source.title, source.url)
        if table.row_count == 0:
            table.add_row("-", "No sources", "-")
        return Panel(table, title="AISearch Sources", border_style="panel", box=box.ROUNDED)

    def event_line(self, event: RunEvent) -> Text:
        parts: list[str] = [f"[label]{event.stage.upper()}[/]"]
        if event.iteration is not None:
            parts.append(f"[muted]iter={event.iteration + 1}[/]")
        task_id = str(event.payload.get("task_id", "")).strip()
        if task_id:
            parts.append(f"[muted]task={task_id}[/]")
        trace_type = str(event.metrics.get("trace_type", "")).strip()
        if trace_type:
            parts.append(f"[muted]trace={trace_type}[/]")
        parts.append(f"[accent]{event.message}[/]")
        query = str(event.payload.get("query", "") or event.metrics.get("query", "")).strip()
        if query:
            parts.append(f"[muted]q={shorten(query, width=48, placeholder='...')}[/]")
        url = str(event.payload.get("url", "")).strip()
        if url:
            parts.append(f"[muted]url={shorten(url, width=68, placeholder='...')}[/]")
        urls = event.payload.get("urls")
        if isinstance(urls, list) and urls:
            parts.append(f"[muted]urls={len(urls)}[/]")
        if event.metrics:
            metrics_text = ", ".join(
                f"{key}={value}"
                for key, value in sorted(event.metrics.items(), key=lambda item: item[0])
            )
            if metrics_text:
                parts.append(f"[muted]{metrics_text}[/]")
        return Text.from_markup(" ".join(parts))

    def success(self, message: str) -> Panel:
        return Panel(message, border_style="ok", box=box.ROUNDED)

    def warning(self, message: str) -> Panel:
        return Panel(message, border_style="warn", box=box.ROUNDED)

    def error(self, message: str) -> Panel:
        return Panel(message, border_style="danger", box=box.ROUNDED)
