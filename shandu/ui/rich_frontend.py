from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from textwrap import shorten
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.constrain import Constrain
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ..contracts import AISearchResult, ResearchRequest, ResearchRunResult, RunEvent
from ..services.run_store import RunSummary
from .formatting import audit_summary

# Long-form markdown is unreadable at full terminal width; cap the measure.
_READING_WIDTH = 88

_STAGE_WIDTH = 11
# Width of the "HH:MM:SS " event timestamp, so the header's question starts in
# the same column as the stage words below it.
_GUTTER_WIDTH = 9
_TASK_WIDTH = 15

# Metrics that describe a single trace event, not the run so far.
_EVENT_ONLY_METRICS = frozenset(
    {
        "trace_type",
        "query",
        "url",
        "hits",
        "scraped",
        "missed",
        "credibility",
        "confidence",
        "task_index",
        "task_total",
        "max_results",
        "url_count",
    }
)


def _inspect_result_summary(output: object) -> list[tuple[str, str]]:
    if not isinstance(output, dict):
        return []
    stats = output.get("run_stats")
    if not isinstance(stats, dict):
        return []
    rows: list[tuple[str, str]] = []
    for key, label in (
        ("iterations", "iterations"),
        ("evidence_count", "evidence"),
        ("citation_count", "citations"),
    ):
        value = stats.get(key)
        if value is not None:
            rows.append((label, str(value)))
    return rows


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
                "brand": "bold #c2683f",
                "markdown.h1": "bold",
                "markdown.h2": "bold",
                "markdown.h3": "bold",
                "markdown.item.bullet": "dim",
                "markdown.item.number": "dim",
                "markdown.link": "#c2683f",
                "markdown.link_url": "dim",
                "markdown.code": "dim",
                "markdown.block_quote": "dim",
                "accent": "default",
                "muted": "dim",
                "label": "dim",
                "title": "bold",
                "ok": "#3f8f5f",
                "warn": "#b58d2a",
                "danger": "#c0564f",
                "panel": "dim",
            }
        )
        self.theme = theme
        if console is None:
            self.console = Console(theme=theme)
        else:
            console.push_theme(theme)
            self.console = console

    def header(self, question: str, details: list[str]) -> RenderableType:
        """The question a command works on, then the settings it runs under."""
        table = Table.grid()
        table.add_column(width=_GUTTER_WIDTH, no_wrap=True)
        table.add_column(overflow="fold")
        table.add_row(Text("shandu", style="brand"), Text(question, style="accent"))
        if details:
            table.add_row("", Text("  ·  ".join(details), style="muted"))
        return Padding(Constrain(table, _READING_WIDTH), (0, 0, 1, 0))

    def new_snapshot(self, request: ResearchRequest, model: str) -> RunSnapshot:
        return RunSnapshot(request=request, model=model)

    def key_values(self, title: str, rows: list[tuple[str, str]]) -> RenderableType:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="label", no_wrap=True)
        table.add_column(style="accent", overflow="fold")
        for label, value in rows:
            table.add_row(escape(label), escape(str(value)))
        return Group(
            Text(title, style="title"),
            Padding(table, (0, 0, 1, 2)),
        )

    def run_list(self, runs: list[RunSummary]) -> RenderableType:
        # Only the trailing question column shrinks: the run id must stay
        # whole so it can be copied into show, report, or resume.
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="muted", no_wrap=True, min_width=32)
        table.add_column(style="label", no_wrap=True, min_width=10)
        table.add_column(style="label", no_wrap=True, min_width=9)
        table.add_column(style="muted", no_wrap=True, justify="right", min_width=7)
        table.add_column(style="accent", overflow="ellipsis", no_wrap=True, ratio=1)
        for run in runs:
            table.add_row(
                escape(run.run_id),
                escape(run.created_at[:10]),
                escape(run.status),
                f"${run.usd_spent:.3f}" if run.usd_spent else "",
                escape(run.query),
            )
        return Group(Text("Runs", style="title"), Padding(table, (0, 0, 1, 2)))

    def event_line(self, event: RunEvent) -> Text:
        # One event, one line: a wrapped stream loses the column alignment.
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(datetime.now().strftime("%H:%M:%S "), style="muted")
        line.append(event.stage.ljust(_STAGE_WIDTH), style="label")
        task_id = str(event.payload.get("task_id", "")).strip()
        line.append(task_id[: _TASK_WIDTH - 1].ljust(_TASK_WIDTH), style="muted")
        message = event.message
        prefix = f"Task {task_id} "
        if task_id and message.startswith(prefix):
            message = message[len(prefix) :]
        line.append(message, style="accent")

        details: list[str] = []
        if event.iteration is not None:
            details.append(f"iter {event.iteration + 1}")
        trace_type = str(event.metrics.get("trace_type", "")).strip()
        if trace_type:
            details.append(trace_type)
        query = str(
            event.payload.get("query", "") or event.metrics.get("query", "")
        ).strip()
        if query:
            details.append(f'"{shorten(query, width=48, placeholder="...")}"')
        url = str(event.payload.get("url", "")).strip()
        if url:
            details.append(shorten(url, width=68, placeholder="..."))
        urls = event.payload.get("urls")
        if isinstance(urls, list) and urls:
            details.append(f"urls={len(urls)}")
        metrics = " ".join(
            f"{key}={value}"
            for key, value in sorted(event.metrics.items())
            if key not in ("trace_type", "query")
        )
        if metrics:
            details.append(metrics)
        if details:
            line.append("  " + " · ".join(details), style="muted")
        return line

    def print_event(self, event: RunEvent, target: Console | None = None) -> None:
        (target or self.console).print(
            self.event_line(event), no_wrap=True, overflow="ellipsis", crop=True
        )

    def recap(self, snapshot: RunSnapshot) -> RenderableType:
        rows = [
            ("run", snapshot.run_id),
            ("model", snapshot.model),
            ("stage", snapshot.current_stage),
            ("iteration", str(snapshot.iteration + 1)),
            ("query", snapshot.request.query),
            ("detail", snapshot.request.detail_level),
            ("parallelism", str(snapshot.request.parallelism)),
        ]
        rows.extend(
            (str(key), str(value))
            for key, value in sorted(snapshot.metrics.items())
            if key not in _EVENT_ONLY_METRICS
        )

        timeline = Table.grid(padding=(0, 2))
        timeline.add_column(style="label", no_wrap=True)
        timeline.add_column(style="muted", no_wrap=True)
        timeline.add_column(style="accent", overflow="ellipsis")
        for event in snapshot.events[-12:]:
            timeline.add_row(
                escape(event.stage),
                escape(str(event.payload.get("task_id", "")).strip()),
                escape(event.message),
            )
        if timeline.row_count == 0:
            timeline.add_row("", "", "No events recorded")

        return Group(
            Rule(style="panel"),
            self.key_values("Run detail", rows),
            Text("Last events", style="title"),
            Padding(timeline, (0, 0, 1, 2)),
        )

    def result_summary(self, result: ResearchRunResult) -> RenderableType:
        stats = result.run_stats
        rows = [
            ("run", result.run_id),
            ("iterations", str(stats.get("iterations", 0))),
            ("evidence", str(stats.get("evidence_count", 0))),
            ("citations", str(stats.get("citation_count", 0))),
            ("elapsed", f"{stats.get('elapsed_seconds', 0)}s"),
        ]
        sub_question_coverage = stats.get("sub_question_coverage")
        if isinstance(sub_question_coverage, dict) and sub_question_coverage:
            rows.append(
                (
                    "sub-questions",
                    " · ".join(
                        f"{count} {level}"
                        for level, count in sub_question_coverage.items()
                        if count
                    ),
                )
            )
        findings = stats.get("findings")
        if isinstance(findings, int) and findings > 0:
            rows.append(("findings", str(findings)))
        checked = audit_summary(stats.get("citation_audit"))
        if checked:
            rows.append(("citations checked", checked))
        model_calls = stats.get("agent_model_calls")
        if isinstance(model_calls, int) and model_calls > 0:
            rows.append(("Model Calls", str(model_calls)))
        metered_calls = stats.get("metered_calls", stats.get("llm_calls"))
        coverage = self._cost_coverage(stats, metered_calls, model_calls)
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                rows.append(
                    ("Cost Coverage", f"{coverage} ({metered_calls}/{model_calls})")
                )
            else:
                rows.append(("Metered Calls", str(metered_calls)))
        llm_tokens = stats.get("llm_tokens")
        if isinstance(llm_tokens, int) and llm_tokens > 0:
            rows.append(("LLM Tokens", f"{llm_tokens:,}"))
        usd_spent = stats.get("usd_spent")
        if isinstance(usd_spent, (int, float)) and float(usd_spent) > 0:
            label = "Metered Cost" if coverage == "partial" else "USD Spent"
            rows.append((label, f"${float(usd_spent):.6f}"))

        sources = Table.grid(padding=(0, 2))
        sources.add_column(style="muted", no_wrap=True, justify="right")
        sources.add_column(style="label", no_wrap=True)
        sources.add_column(style="accent", overflow="ellipsis")
        for item in result.citations[:8]:
            sources.add_row(
                f"[{item.citation_id}]", escape(item.publisher), escape(item.title)
            )
        remainder = len(result.citations) - 8
        if remainder > 0:
            sources.add_row("", "", f"and {remainder} more")
        if sources.row_count == 0:
            sources.add_row("", "", "No citations")

        return Group(
            Rule(style="panel"),
            self.key_values("Run summary", rows),
            Text("Sources", style="title"),
            Padding(sources, (0, 0, 1, 2)),
        )

    def document(self, title: str, content: str) -> RenderableType:
        return Group(
            Rule(Text(title, style="title"), style="panel", align="left"),
            Padding(Constrain(Markdown(content), _READING_WIDTH), (1, 2)),
        )

    def inspect_summary(self, payload: dict[str, object]) -> RenderableType:
        events = payload.get("events", [])
        rows = [
            (key, str(payload.get(key, "")))
            for key in ("run_id", "status", "created_at", "updated_at")
        ]
        rows.append(("events", str(len(events) if isinstance(events, list) else 0)))
        rows.extend(_inspect_result_summary(payload.get("output_json")))
        return self.key_values("Run inspection", rows)

    def sources(self, result: AISearchResult) -> RenderableType:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="muted", no_wrap=True, justify="right")
        table.add_column(style="accent", overflow="ellipsis")
        table.add_column(style="label", overflow="fold")
        for index, source in enumerate(result.sources[:10], start=1):
            table.add_row(f"[{index}]", escape(source.title), escape(source.url))
        if table.row_count == 0:
            table.add_row("", "No sources", "")
        return Group(
            Text("Sources", style="title"),
            Padding(table, (0, 0, 1, 2)),
        )

    def success(self, message: str) -> Text:
        return Text.assemble(("ok     ", "ok"), (message, "accent"))

    def warning(self, message: str) -> Text:
        return Text.assemble(("warn   ", "warn"), (message, "accent"))

    def error(self, message: str) -> Text:
        return Text.assemble(("error  ", "danger"), (message, "accent"))

    def _cost_coverage(
        self, stats: dict[str, Any], metered_calls: object, model_calls: object
    ) -> str:
        coverage = str(stats.get("cost_coverage", "")).strip()
        if coverage in ("partial", "full"):
            return coverage
        if (
            isinstance(metered_calls, int)
            and metered_calls > 0
            and isinstance(model_calls, int)
            and model_calls > 0
        ):
            return "partial" if metered_calls < model_calls else "full"
        return ""
