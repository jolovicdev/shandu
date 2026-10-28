from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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

# The metrics an event line shows, worded as the GUI's event feed words them.
# Every other metric shows only with --verbose.
_METRIC_LABELS = {
    "sub_questions": "sub-questions",
    "tasks": "subagents",
    "hits": "results",
    "candidates": "candidates",
    "url_count": "pages to read",
    "scraped": "pages read",
    "evidence": "evidence",
    "citations": "sources",
    "report_words": "words",
}

# Trace events that announce a query, scrape or page read. The completion line
# carries the same query or URL and the result, so only --verbose shows both.
_START_TRACES = frozenset({"query_started", "scrape_started", "extract_started"})

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


_STATUS_STYLES = {"completed": "ok", "cancelled": "warn", "failed": "danger"}


def _metric_text(metrics: dict[str, Any]) -> str:
    return " · ".join(
        f"{label} {metrics[key]}" for key, label in _METRIC_LABELS.items() if key in metrics
    )


def _local_time(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stamp


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
        ("citation_count", "sources"),
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
    iteration: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    events: list[RunEvent] = field(default_factory=list)
    subagents: dict[str, str] = field(default_factory=dict)

    def apply(self, event: RunEvent) -> None:
        self.events.append(event)
        self.current_stage = event.stage
        self.iteration = event.iteration or self.iteration
        if event.metrics:
            self.metrics.update(event.metrics)
        # The first event carries the id in its metrics; the last, in its payload.
        run_id = event.payload.get("run_id") or event.metrics.get("run_id")
        if run_id:
            self.run_id = str(run_id)
        task_id = str(event.payload.get("task_id", "")).strip()
        if task_id:
            self.subagent(task_id)

    def subagent(self, task_id: str) -> str:
        """The label for a task, numbered by arrival across the whole run."""
        return self.subagents.setdefault(task_id, f"Subagent {len(self.subagents) + 1}")


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
        # Only the trailing question column shrinks, and it wraps so that no
        # question is cut. Commands that take a run id accept its first eight
        # characters.
        table = Table.grid(padding=(0, 2), expand=True)
        table.show_header = True
        table.header_style = "label"
        table.add_column("run", style="muted", no_wrap=True, min_width=8)
        table.add_column("started", style="label", no_wrap=True, min_width=16)
        table.add_column("status", no_wrap=True, min_width=9)
        table.add_column("cost", style="muted", no_wrap=True, justify="right", min_width=7)
        table.add_column("question", style="accent", overflow="fold", ratio=1)
        for run in runs:
            table.add_row(
                escape(run.run_id[:8]),
                escape(_local_time(run.created_at)),
                Text(run.status, style=_STATUS_STYLES.get(run.status, "label")),
                f"${run.usd_spent:.3f}" if run.usd_spent else "",
                escape(run.query),
            )
        return Group(Text("Runs", style="title"), Padding(table, (0, 0, 1, 2)))

    def event_line(self, event: RunEvent, snapshot: RunSnapshot, verbose: bool = False) -> Text:
        message = event.message
        task_id = str(event.payload.get("task_id", "")).strip()
        if task_id:
            label = snapshot.subagent(task_id)
            message = message.replace(f"Task {task_id}", label).replace(task_id, label)
        line = Text()
        line.append(datetime.now().strftime("%H:%M:%S "), style="muted")
        line.append(event.stage.ljust(_STAGE_WIDTH), style="label")
        line.append(message, style="accent")

        query = str(
            event.payload.get("query", "") or event.metrics.get("query", "")
        ).strip()
        if verbose:
            details = [
                f"iteration {event.iteration + 1}" if event.iteration is not None else "",
                str(event.metrics.get("trace_type", "")).strip(),
                " ".join(
                    f"{key}={value}"
                    for key, value in sorted(event.metrics.items())
                    if key not in ("trace_type", "query")
                ),
            ]
        else:
            details = [_metric_text(event.metrics)]
        details += [
            f'"{query}"' if query else "",
            str(event.payload.get("url", "")).strip(),
            str(event.payload.get("error") or event.payload.get("reason") or "").strip(),
        ]
        text = " · ".join(part for part in details if part)
        if text:
            line.append("  " + text, style="muted")
        return line

    def print_event(
        self,
        event: RunEvent,
        snapshot: RunSnapshot,
        target: Console | None = None,
        verbose: bool = False,
    ) -> None:
        if not verbose and event.metrics.get("trace_type") in _START_TRACES:
            return
        target = target or self.console
        line = self.event_line(event, snapshot, verbose)
        if target.is_terminal:
            # One event, one line: a wrapped stream loses the column alignment.
            target.print(line, no_wrap=True, overflow="ellipsis", crop=True)
        else:
            # A log keeps the whole query and URL.
            target.print(line, soft_wrap=True)

    def recap(self, snapshot: RunSnapshot) -> RenderableType:
        rows = [
            ("run", snapshot.run_id),
            ("model", snapshot.model),
            ("stage", snapshot.current_stage),
            ("iteration", str(snapshot.iteration + 1)),
            ("query", snapshot.request.query),
            ("detail", snapshot.request.detail_level),
            ("subagents", str(snapshot.request.parallelism)),
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
            ("run", result.run_id[:8]),
            ("iterations", str(stats.get("iterations", 0))),
            ("evidence", str(stats.get("evidence_count", 0))),
            ("sources", str(stats.get("citation_count", 0))),
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
            rows.append(("citation checks", checked))
        model_calls = stats.get("agent_model_calls")
        if isinstance(model_calls, int) and model_calls > 0:
            rows.append(("model calls", str(model_calls)))
        metered_calls = stats.get("metered_calls", stats.get("llm_calls"))
        coverage = str(stats.get("cost_coverage", "")).strip()
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                if coverage not in ("partial", "full"):
                    coverage = "partial" if metered_calls < model_calls else "full"
            else:
                rows.append(("metered calls", str(metered_calls)))
        llm_tokens = stats.get("llm_tokens")
        if isinstance(llm_tokens, int) and llm_tokens > 0:
            rows.append(("LLM tokens", f"{llm_tokens:,}"))
        usd_spent = stats.get("usd_spent")
        if isinstance(usd_spent, (int, float)) and float(usd_spent) > 0:
            cost = f"${float(usd_spent):.6f}"
            if coverage == "partial":
                cost = f"at least {cost} · {metered_calls} of {model_calls} model calls metered"
            rows.append(("USD spent", cost))

        sources = Table.grid(padding=(0, 2))
        sources.add_column(style="muted", no_wrap=True, justify="right")
        # A long host name as publisher would squeeze every title into a narrow
        # wrapped column.
        sources.add_column(style="label", no_wrap=True, max_width=24, overflow="ellipsis")
        sources.add_column(style="accent", overflow="ellipsis")
        for item in result.citations[:8]:
            sources.add_row(
                f"[{item.citation_id}]", escape(item.publisher), escape(item.title)
            )
        remainder = len(result.citations) - 8
        if remainder > 0:
            sources.add_row("", "", f"and {remainder} more")
        if sources.row_count == 0:
            sources.add_row("", "", "No sources")

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
            ("run", str(payload.get("run_id", ""))),
            ("status", str(payload.get("status", ""))),
            ("started", _local_time(str(payload.get("created_at", "")))),
            ("updated", _local_time(str(payload.get("updated_at", "")))),
            ("events", str(len(events) if isinstance(events, list) else 0)),
        ]
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
