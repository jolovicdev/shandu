from __future__ import annotations

import logging
import sys
import threading
import time
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from rich.console import Console, ConsoleOptions, Group, JustifyMethod, RenderableType, RenderResult
from rich.constrain import Constrain
from rich.live import Live
from rich.markdown import Heading, Markdown, MarkdownElement, TableElement
from rich.markup import escape
from rich.padding import Padding
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ..contracts import AISearchResult, ResearchRequest, ResearchRunResult, RunEvent
from ..services.run_store import RunSummary
from . import intro
from .formatting import audit_summary

# Long-form markdown is unreadable at full terminal width; cap the measure.
_READING_WIDTH = 88
# Cards stop growing here, so a wide terminal does not stretch two short
# columns of labels and values apart.
_CARD_WIDTH = 104

_STAGE_WIDTH = 11

# The flag's crimson and blue over cool grays, each text color at 3:1 or
# more against both a dark and a white background. Crimson marks where to
# look; blue marks what is done or cited; gold appears only in warnings.
_RED = "#d0454c"
_BLUE = "#5a86c8"
_BLUE_SOFT = "#9db8e3"
_GRAY = "#80868f"
_PENDING = "#6e747e"
_LINE = "#4b515c"
_CODE = "#8394ad"
_GOLD = "#b5892b"

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
_STAGE_STYLES = {"complete": "ok", "error": "danger"}

# The steps a run goes through, in order. Plan, research and synthesis repeat
# once per iteration; a report rewrite starts at the citations.
_PHASES = ("brief", "plan", "research", "synthesize", "cite", "write", "audit")
_PHASE_LABELS = {
    "brief": "Brief",
    "plan": "Plan",
    "research": "Research",
    "synthesize": "Synthesize",
    "cite": "Cite",
    "write": "Write",
    "audit": "Audit",
}
# What each step is doing, in the words the progress block shows under the
# step row.
_PHASE_NOTES = {
    "brief": "Turning the question into sub-questions",
    "plan": "Planning tasks for the open sub-questions",
    "research": "Searching and reading pages",
    "synthesize": "Rating how well each sub-question is answered",
    "cite": "Building the list of sources",
    "write": "Writing the report",
    "audit": "Checking each citation against its source",
}
# Fetch errors as an event line words them.
_FETCH_ERRORS = {
    "blocked_403": "blocked (403)",
    "login_required": "login required",
    "paywall": "paywalled",
    "captcha": "behind a captcha",
    "empty_content": "empty page",
    "empty_js_shell": "needs JavaScript",
    "error_page": "error page",
    "timeout": "timed out",
    "scrape_failed": "could not load",
    "pdf_parse_failed": "unreadable PDF",
    "non_text_content": "not text",
}
_REASON_LIMIT = 160
# A dot that breathes in and out marks the step under way.
_PULSE = ("·", "•", "●", "●", "•", "·")

_COVERAGE_STYLES = (
    ("strong", "ok"),
    ("moderate", "soft"),
    ("weak", "label"),
    ("none", "danger"),
)
_AUDIT_STYLES = (
    ("supported", "ok"),
    ("partial", "soft"),
    ("unsupported", "danger"),
    ("unchecked", "panel"),
)
_BAR_WIDTH = 24


def _metric_text(metrics: dict[str, Any]) -> str:
    return " · ".join(
        f"{label} {metrics[key]}" for key, label in _METRIC_LABELS.items() if key in metrics
    )


def _local_time(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stamp


def _count(number: int, noun: str, plural: str) -> str:
    return f"{number} {noun if number == 1 else plural}"


def _duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m {rest:02d}s" if minutes else f"{rest}s"


def _clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes:02d}:{rest:02d}"


def _bar(counts: dict[str, Any], styles: tuple[tuple[str, str], ...]) -> Text:
    """A stacked bar whose segments are each count's share of the total."""
    parts = [(int(counts.get(key) or 0), style) for key, style in styles]
    total = sum(count for count, _ in parts)
    bar = Text()
    if total <= 0:
        return bar
    # Each boundary is rounded from the running total, so the segments always
    # add up to the full width.
    reached = 0
    drawn = 0
    for count, style in parts:
        reached += count
        end = round(_BAR_WIDTH * reached / total)
        bar.append("━" * (end - drawn), style=style)
        drawn = end
    return bar


class _Heading(Heading):
    # Rich centers a top-level heading, which would float it above the
    # left-aligned reading column.
    LEVEL_ALIGN: ClassVar[dict[str, JustifyMethod]] = {**Heading.LEVEL_ALIGN, "h1": "left"}


class _Table(TableElement):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # A narrow terminal would cut long cell values to "…"; folding keeps
        # every figure, broken across lines where it must be.
        for renderable in super().__rich_console__(console, options):
            if isinstance(renderable, Table):
                for column in renderable.columns:
                    column.overflow = "fold"
            yield renderable


class _Report(Markdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "heading_open": _Heading,
        "table_open": _Table,
    }


class _Card(Live):
    """A Live display that also moves logging above the card.

    Live redirects print() and sys.stderr, but a logging handler made at
    import, such as LiteLLM's, holds the original stream and would draw
    through the card.
    """

    _moved: list[logging.StreamHandler[Any]] = []

    def start(self, refresh: bool = False) -> None:
        super().start(refresh)
        loggers = [logging.getLogger()] + [
            logger for logger in logging.Logger.manager.loggerDict.values() if isinstance(logger, logging.Logger)
        ]
        self._moved = [
            handler
            for logger in loggers
            for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.__stderr__
        ]
        for handler in self._moved:
            handler.setStream(sys.stderr)

    def stop(self) -> None:
        for handler in self._moved:
            handler.setStream(sys.__stderr__)
        super().stop()


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
class TaskProgress:
    label: str
    iteration: int
    focus: str = ""
    queries: int = 0
    pages: int = 0
    state: str = "running"


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
    # Keyed by iteration too: a planner may number each iteration's tasks afresh.
    tasks: dict[tuple[int, str], TaskProgress] = field(default_factory=dict)
    # The step under way: one of _PHASES, or "done".
    phase: str = "brief"
    first_phase: str = "brief"
    # The iteration the steps are working toward: planning for the next one
    # starts before any of its events carry its number.
    round: int = 0
    started: float = field(default_factory=time.monotonic)
    phase_started: float = field(default_factory=time.monotonic)
    # Events arrive on the runner's thread while a live display reads the
    # snapshot from its refresh thread.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def apply(self, event: RunEvent) -> None:
        with self.lock:
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
            phase = self.phase
            self._advance(event, task_id)
            if self.phase != phase:
                self.phase_started = time.monotonic()

    def subagent(self, task_id: str) -> str:
        """The label for a task, numbered by arrival across the whole run."""
        return self.subagents.setdefault(task_id, f"Subagent {len(self.subagents) + 1}")

    def _advance(self, event: RunEvent, task_id: str) -> None:
        metrics, payload = event.metrics, event.payload
        if task_id:
            task = self.tasks.setdefault(
                (self.iteration, task_id),
                TaskProgress(label=self.subagent(task_id), iteration=self.iteration),
            )
            task.focus = task.focus or str(payload.get("focus", "")).strip()
            trace = metrics.get("trace_type")
            # A failed search or page is a trace event inside a task that
            # keeps working; only the task's own failure has no trace type.
            if event.stage == "error" and not trace:
                task.state = "failed"
            elif "tasks_completed" in metrics:
                task.state = "done"
            elif trace == "query_completed":
                task.queries += 1
            elif trace == "scrape_completed":
                task.pages += int(metrics.get("scraped") or 0)
        elif event.stage == "plan":
            self.phase = "research" if "tasks" in metrics else "plan"
            if "tasks" in metrics:
                self.round = self.iteration
        elif event.stage == "search" and "evidence" in metrics:
            for task in self.tasks.values():
                if task.iteration == self.iteration and task.state == "running":
                    task.state = "done"
            self.phase = "synthesize"
        elif event.stage == "synthesize":
            last = self.iteration + 1 >= self.request.max_iterations
            self.phase = "cite" if last else "plan"
            if not last:
                self.round = self.iteration + 1
        elif event.stage == "cite":
            self.phase = "write"
        elif event.stage == "report" and "report_words" in metrics:
            self.phase = "audit" if self.request.audit else "done"
        elif (event.stage == "report" and "citation_audit" in metrics) or event.stage == "complete":
            self.phase = "done"


class _Details:
    """Short labels laid out left to right, wrapping between labels only."""

    def __init__(self, labels: list[Text]) -> None:
        self.labels = labels

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        line = Text()
        for label in self.labels:
            if line and line.cell_len + 3 + label.cell_len > options.max_width:
                yield line
                line = Text()
            if line:
                line.append(" · ", style="panel")
            line.append_text(label)
        if line:
            yield line


class ShanduUI:
    def __init__(self, console: Console | None = None) -> None:
        theme = Theme(
            {
                "brand": f"bold {_RED}",
                "markdown.h1": f"bold {_RED}",
                "markdown.h1.border": _LINE,
                "markdown.h2": f"bold {_BLUE}",
                "markdown.h3": "bold",
                "markdown.h4": "italic",
                "markdown.item.bullet": _RED,
                "markdown.item.number": _BLUE,
                "markdown.link": _BLUE,
                # With hyperlinks on, Rich styles a link's text with this one.
                "markdown.link_url": _BLUE,
                "markdown.code": _CODE,
                "markdown.block_quote": f"italic {_GRAY}",
                "markdown.hr": _LINE,
                "markdown.table.border": _LINE,
                "markdown.table.header": "bold",
                "accent": "default",
                "muted": _GRAY,
                "label": _GRAY,
                "title": "bold",
                "number": "bold",
                "cite": _BLUE,
                "soft": _BLUE_SOFT,
                "ok": _BLUE,
                "warn": _GOLD,
                "danger": _RED,
                "panel": _LINE,
                "pending": _PENDING,
            }
        )
        self.theme = theme
        if console is None:
            self.console = Console(theme=theme)
        else:
            console.push_theme(theme)
            self.console = console

    def _section(
        self, title: str, body: RenderableType, aside: str | Text = ""
    ) -> RenderableType:
        """A labelled block: a crimson tick and the title in capitals, an
        optional note on the right, and the body indented under them."""
        head = Table.grid(expand=True)
        head.add_column(ratio=1)
        head.add_column(justify="right", no_wrap=True)
        head.add_row(
            Text.assemble(("▍", "brand"), (title.upper(), "label")),
            aside if isinstance(aside, Text) else Text(aside, style="label"),
        )
        return Group(
            Constrain(head, _CARD_WIDTH),
            Padding(Constrain(body, _CARD_WIDTH - 2), (0, 0, 1, 2)),
        )

    def _lockup(self) -> RenderableType:
        """The flag at rest, each band beside one line of the name."""
        if not intro.can_draw(self.console):
            name, tagline, version = intro.lines()
            return Text.assemble((name, "title"), (f"  {tagline} · {version}", "muted"))
        return intro.still()

    def header(self, question: str, details: list[str]) -> RenderableType:
        """The question a command works on, then the settings it runs under."""
        labels = [
            Text(detail, style="accent" if index == 0 else "muted")
            for index, detail in enumerate(details)
        ]
        return Padding(
            Group(
                self._lockup(),
                Text(""),
                Constrain(Text(question, style="title"), _CARD_WIDTH),
                Constrain(_Details(labels), _CARD_WIDTH),
            ),
            (0, 0, 1, 0),
        )

    def home(
        self,
        sections: list[tuple[str, list[tuple[str, str]]]],
        example: str,
        with_mark: bool = True,
        notice: Text | None = None,
    ) -> RenderableType:
        """Every command by what it is for, and a first command to try.

        ``with_mark`` is off when the intro has just left the flag on screen;
        ``notice`` says what a first run would fail on.
        """
        parts: list[RenderableType] = []
        if with_mark:
            parts += [self._lockup(), Text("")]
        if notice is not None:
            parts += [notice, Text("")]
        for title, commands in sections:
            table = Table.grid(padding=(0, 3))
            table.add_column(no_wrap=True, min_width=20)
            table.add_column(style="accent")
            for usage, summary in commands:
                name, _, argument = usage.partition(" ")
                table.add_row(
                    Text.assemble((name, "title"), (f" {argument}" if argument else "", "muted")),
                    escape(summary),
                )
            parts.append(self._section(title, table))
        hints = Table.grid(padding=(0, 2))
        hints.add_column(style="muted", no_wrap=True)
        hints.add_column(style="title")
        hints.add_row("Start with", escape(example))
        hints.add_row("Every option", "shandu run --help")
        parts.append(Padding(hints, (0, 0, 0, 2)))
        return Group(*parts)

    def new_snapshot(
        self, request: ResearchRequest, model: str, phase: str = "brief"
    ) -> RunSnapshot:
        return RunSnapshot(request=request, model=model, phase=phase, first_phase=phase)

    def key_values(self, title: str, rows: list[tuple[str, str | Text]]) -> RenderableType:
        table = Table.grid(padding=(0, 3))
        table.add_column(style="label", no_wrap=True)
        table.add_column(style="accent", overflow="fold")
        for label, value in rows:
            table.add_row(escape(label), value if isinstance(value, Text) else escape(str(value)))
        return self._section(title, table)

    def run_list(self, runs: list[RunSummary]) -> RenderableType:
        # Only the trailing question column shrinks, and it wraps so that no
        # question is cut. Commands that take a run id accept its first eight
        # characters.
        table = Table.grid(padding=(0, 2), expand=True)
        table.show_header = True
        table.header_style = "title"
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
                f"${run.usd_spent:.4f}" if run.usd_spent else "",
                escape(run.query),
            )
        return self._section("Runs", table, f"{len(runs)} shown, newest first")

    def event_line(self, event: RunEvent, snapshot: RunSnapshot, verbose: bool = False) -> Text:
        message = event.message
        task_id = str(event.payload.get("task_id", "")).strip()
        if task_id:
            label = snapshot.subagent(task_id)
            message = message.replace(f"Task {task_id}", label).replace(task_id, label)
        line = Text()
        line.append(datetime.now().strftime("%H:%M:%S "), style="muted")
        line.append(event.stage.ljust(_STAGE_WIDTH), style=_STAGE_STYLES.get(event.stage, "label"))
        line.append(message, style="danger" if event.stage == "error" else "accent")

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
        # The reason goes first: a terminal line is cropped at its end, and a
        # provider's error can span lines.
        reason = " ".join(
            str(event.payload.get("error") or event.payload.get("reason") or "").split()
        )
        fetch_error = str(event.payload.get("fetch_error") or "")
        reason = reason or _FETCH_ERRORS.get(fetch_error, fetch_error.replace("_", " "))
        if len(reason) > _REASON_LIMIT:
            reason = reason[: _REASON_LIMIT - 1] + "…"
        details += [
            reason,
            f'"{" ".join(query.split())}"' if query else "",
            str(event.payload.get("url", "")).strip(),
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

    def live_progress(self, snapshot: RunSnapshot, console: Console) -> Live:
        """A progress card pinned under the event lines while a run works.

        It clears itself when the run ends, so the scrollback keeps the event
        lines and the summary that follows them.
        """
        return _Card(
            console=console,
            get_renderable=lambda: self.progress(snapshot),
            refresh_per_second=10,
            transient=True,
        )

    def progress(self, snapshot: RunSnapshot) -> RenderableType:
        with snapshot.lock:
            phase = snapshot.phase
            phase_started = snapshot.phase_started
            iteration = snapshot.iteration
            shown_round = snapshot.round
            tasks = [copy(task) for task in snapshot.tasks.values() if task.iteration == iteration]
            pages = sum(task.pages for task in snapshot.tasks.values())
            calls = snapshot.metrics.get("agent_model_calls")
            sources = snapshot.metrics.get("citations")
            elapsed = time.monotonic() - snapshot.started
        spinner = _PULSE[int(time.monotonic() * 5) % len(_PULSE)]
        phases = [
            name
            for name in _PHASES[_PHASES.index(snapshot.first_phase):]
            if name != "audit" or snapshot.request.audit
        ]
        current = phases.index(phase) if phase in phases else len(phases)

        steps = Text()
        for index, name in enumerate(phases):
            if index:
                steps.append("  ")
            label = _PHASE_LABELS[name]
            if index < current:
                steps.append("✓ ", style="ok")
                steps.append(label, style="muted")
            elif index == current:
                steps.append(f"{spinner} ", style="brand")
                steps.append(label, style="title")
            else:
                steps.append(f"  {label}", style="pending")
        body: list[RenderableType] = [steps]
        if phase in _PHASE_NOTES:
            body.append(
                Text.assemble(
                    (_PHASE_NOTES[phase], "muted"), (f"  {_clock(time.monotonic() - phase_started)}", "pending")
                )
            )

        if tasks and phase in ("research", "synthesize"):
            table = Table.grid(padding=(0, 2), expand=True)
            table.add_column(no_wrap=True)
            table.add_column(no_wrap=True, style="title")
            table.add_column(overflow="ellipsis", no_wrap=True, ratio=1, style="muted")
            table.add_column(no_wrap=True, justify="right", style="muted")
            for task in tasks[:8]:
                mark = {"done": Text("✓", style="ok"), "failed": Text("✗", style="danger")}.get(
                    task.state, Text(spinner, style="brand")
                )
                table.add_row(
                    mark,
                    task.label,
                    escape(task.focus),
                    f"{_count(task.queries, 'query', 'queries')} · {_count(task.pages, 'page', 'pages')}",
                )
            body += [Text(""), table]

        totals = Text()
        for label, value in (("pages read", pages), ("model calls", calls), ("sources", sources)):
            if value:
                if totals:
                    totals.append("   ")
                totals.append(f"{label} ", style="muted")
                totals.append(str(value), style="number")
        if totals:
            body += [Text(""), totals]

        if phase in ("cite", "write"):
            title = "Rewriting the report" if snapshot.first_phase == "cite" else "Writing the report"
        elif phase == "audit":
            title = "Checking citations"
        elif phase == "done":
            title = "Finishing"
        else:
            title = f"Iteration {shown_round + 1} of {snapshot.request.max_iterations}"
        aside = Text.assemble(("Ctrl+C stops and keeps progress   ", "pending"), (_clock(elapsed), "label"))
        return Group(Text(""), self._section(title, Group(*body), aside))

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
        timeline.add_column(no_wrap=True)
        timeline.add_column(style="muted", no_wrap=True)
        timeline.add_column(style="accent", overflow="ellipsis")
        for event in snapshot.events[-12:]:
            timeline.add_row(
                Text(event.stage, style=_STAGE_STYLES.get(event.stage, "label")),
                escape(str(event.payload.get("task_id", "")).strip()),
                escape(event.message),
            )
        if timeline.row_count == 0:
            timeline.add_row("", "", "No events recorded")

        return Group(
            self.key_values("Run detail", rows),
            self._section("Last events", timeline),
        )

    def result_summary(self, result: ResearchRunResult) -> RenderableType:
        stats = result.run_stats
        table = Table.grid(padding=(0, 3))
        table.add_column(style="label", no_wrap=True)
        table.add_column(overflow="fold")

        def row(label: str, value: str, style: str = "accent", bar: Text | None = None) -> None:
            # The bar goes under its counts, so they never wrap to make room.
            cell = Text(value, style=style)
            table.add_row(label, Group(cell, bar) if bar else cell)

        row("run", result.run_id[:8], "muted")
        row("iterations", str(stats.get("iterations", 0)), "number")
        row("evidence", str(stats.get("evidence_count", 0)), "number")
        row("sources", str(stats.get("citation_count", 0)), "number")
        row("elapsed", _duration(float(stats.get("elapsed_seconds") or 0)), "number")
        sub_question_coverage = stats.get("sub_question_coverage")
        if isinstance(sub_question_coverage, dict) and sub_question_coverage:
            row(
                "sub-questions",
                " · ".join(
                    f"{count} {level}" for level, count in sub_question_coverage.items() if count
                ),
                bar=_bar(sub_question_coverage, _COVERAGE_STYLES),
            )
        findings = stats.get("findings")
        if isinstance(findings, int) and findings > 0:
            row("findings", str(findings), "number")
        checked = audit_summary(stats.get("citation_audit"))
        if checked:
            row("citation checks", checked, bar=_bar(stats["citation_audit"], _AUDIT_STYLES))
        model_calls = stats.get("agent_model_calls")
        if isinstance(model_calls, int) and model_calls > 0:
            row("model calls", str(model_calls), "number")
        fallbacks = stats.get("agent_fallbacks")
        if isinstance(fallbacks, int) and fallbacks > 0:
            row("fallbacks", f"{fallbacks} model calls fell back to a simpler answer", "warn")
        metered_calls = stats.get("metered_calls", stats.get("llm_calls"))
        coverage = str(stats.get("cost_coverage", "")).strip()
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                if coverage not in ("partial", "full"):
                    coverage = "partial" if metered_calls < model_calls else "full"
            else:
                row("metered calls", str(metered_calls), "number")
        llm_tokens = stats.get("llm_tokens")
        if isinstance(llm_tokens, int) and llm_tokens > 0:
            row("LLM tokens", f"{llm_tokens:,}", "number")
        usd_spent = stats.get("usd_spent")
        if isinstance(usd_spent, (int, float)) and float(usd_spent) > 0:
            cost = f"${float(usd_spent):.4f}"
            if coverage == "partial":
                cost = f"at least {cost} · {metered_calls} of {model_calls} model calls metered"
            row("USD spent", cost, "number")

        sources = Table.grid(padding=(0, 2))
        sources.add_column(style="cite", no_wrap=True, justify="right")
        # A long host name as publisher would squeeze every title into a narrow
        # wrapped column.
        sources.add_column(style="muted", no_wrap=True, max_width=24, overflow="ellipsis")
        sources.add_column(style="accent", overflow="ellipsis")
        for item in result.citations[:8]:
            sources.add_row(
                f"[{item.citation_id}]", escape(item.publisher), escape(item.title)
            )
        remainder = len(result.citations) - 8
        if remainder > 0:
            sources.add_row("", "", Text(f"and {remainder} more", style="muted"))
        if sources.row_count == 0:
            sources.add_row("", "", Text("No sources", style="muted"))

        return Group(
            Text(""),
            self._section("Run summary", table),
            self._section("Sources", sources, str(len(result.citations))),
        )

    def document(self, title: str, content: str) -> RenderableType:
        # The terminal's own ANSI colors for code, on its own background.
        report = _Report(content, code_theme="ansi_dark")
        return self._section(title, Padding(Constrain(report, _READING_WIDTH), (1, 0, 0, 0)))

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
        table.add_column(style="cite", no_wrap=True, justify="right")
        table.add_column()
        for index, source in enumerate(result.sources[:10], start=1):
            # A URL folded across lines stops being one link to the terminal;
            # cut short, it still links to the whole address.
            url = Text(source.url, style="label", no_wrap=True, overflow="ellipsis")
            url.stylize(Style(link=source.url))
            table.add_row(f"[{index}]", Group(Text(source.title, style="accent"), url))
        if table.row_count == 0:
            table.add_row("", "No sources")
        return self._section("Sources", table)

    def success(self, message: str) -> Text:
        return Text.assemble(("✓ ", "ok"), (message, "accent"))

    def warning(self, message: str) -> Text:
        return Text.assemble(("! ", "warn"), (message, "accent"))

    def error(self, message: str) -> Text:
        return Text.assemble(("✗ ", "danger"), (message, "accent"))
