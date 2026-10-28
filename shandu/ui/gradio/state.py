from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from typing import Any

from ...contracts import CitationEntry, ResearchRunResult, RunEvent
from ..formatting import audit_summary

_STAGE_WORDS = {
    "bootstrap": "Starting",
    "plan": "Planning",
    "search": "Searching",
    "synthesize": "Synthesizing",
    "cite": "Collecting citations",
    "report": "Writing",
    "error": "Failed",
}

# A reference line as the report service writes it: "- **[3] host** - ...".
_REFERENCE_LINE = re.compile(r"^- (?=\*\*\[(\d+)\])")
# A citation marker that is not already a link's text.
_MARKER = re.compile(r"\[(\d+)\](?!\()")

# The loop always passes through these five before it completes.
_STEPS = ("plan", "search", "synthesize", "cite", "write")
_STAGE_STEP = {"plan": 0, "search": 1, "synthesize": 2, "cite": 3, "report": 4}

_TASK_STATE_CLASS = {
    "running": "is-running",
    "completed": "is-complete",
    "failed": "is-error",
    "stopped": "is-stopped",
}

# Feed lines use the same words as the task board; queries and URLs have
# their own place in the line, and ids stay in the run data.
_METRIC_LABELS = {
    "sub_questions": "sub-questions",
    "tasks": "subagents",
    "hits": "results",
    "candidates": "candidates",
    "url_count": "pages to read",
    "scraped": "pages read",
    "evidence": "evidence",
    "citations": "citations",
    "report_words": "words",
}


def _display(value: object) -> str:
    if value is None:
        return ""
    return escape(str(value).strip())


def _metric_text(metrics: dict[str, Any]) -> str:
    return " · ".join(
        f"{label} {metrics[key]}" for key, label in _METRIC_LABELS.items() if key in metrics
    )


def format_cost(amount: float) -> str:
    return f"${amount:.2f}" if amount >= 1 else f"${amount:.4f}"


def _field(label: str, value: object, extra_class: str = "") -> str:
    text = _display(value)
    if not text:
        return ""
    classes = f"field {extra_class}".strip()
    return f"<span class='{classes}'>{escape(label)} <b>{text}</b></span>"


@dataclass(frozen=True, slots=True)
class RenderBundle:
    strip_html: str
    title_html: str
    contents_html: str
    feed_html: str
    task_board_html: str
    report_markdown: str
    run_payload: dict[str, Any]

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.strip_html,
            self.title_html,
            self.contents_html,
            self.feed_html,
            self.task_board_html,
            self.report_markdown,
            self.run_payload,
        )


@dataclass(slots=True)
class GuiRunState:
    max_iterations: int = 0
    run_id: str = ""
    stage: str = "idle"
    iteration: int = 0
    event_count: int = 0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics: dict[str, Any] = field(default_factory=dict)
    entries: list[dict[str, Any]] = field(default_factory=list)
    task_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_markdown: str = ""
    citations: list[CitationEntry] = field(default_factory=list)
    run_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    error_detail: str = ""

    def apply_event(self, event: RunEvent) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.event_count += 1
        self.stage = event.stage
        if event.iteration is not None:
            self.iteration = event.iteration + 1
        if event.metrics:
            self.metrics.update(event.metrics)
            model_calls = event.metrics.get("agent_model_calls")
            if isinstance(model_calls, int) and model_calls > 0:
                self.run_stats["agent_model_calls"] = model_calls
        if event.payload.get("run_id"):
            self.run_id = str(event.payload["run_id"])

        task_id = str(event.payload.get("task_id", "")).strip()
        query = str(
            event.payload.get("query", "") or event.metrics.get("query", "")
        ).strip()
        url = str(event.payload.get("url", "")).strip()
        self.entries.append(
            {
                "time": now,
                "stage": event.stage,
                "task": task_id,
                "message": event.message,
                "metrics": _metric_text(event.metrics),
                "query": query,
                "url": url,
            }
        )
        self.entries = self.entries[-300:]

        if task_id:
            task = self._task(
                task_id, now, default_status="queued", focus=event.payload.get("focus")
            )
            task["Last Update"] = now
            if event.message == f"Task {task_id} started":
                task["Status"] = "running"
            elif event.message == f"Task {task_id} completed":
                task["Status"] = "completed"
            if event.stage == "error":
                task["Status"] = "failed"
            if event.payload.get("focus") and not task["Focus"]:
                task["Focus"] = str(event.payload["focus"])
            if "evidence" in event.metrics:
                task["Evidence"] = str(event.metrics["evidence"])
            if str(event.metrics.get("trace_type", "")).strip():
                if query:
                    task["Last Query"] = query
                if "hits" in event.metrics:
                    task["Hits"] = str(event.metrics["hits"])
                if "scraped" in event.metrics:
                    task["Scraped"] = str(event.metrics["scraped"])

    def apply_result(self, result: ResearchRunResult) -> None:
        self.run_id = result.run_id
        self.report_markdown = result.report_markdown
        self.citations = result.citations
        self.run_stats = result.run_stats
        self.metrics.update(result.run_stats)

    def apply_error(self, message: str, detail: str = "") -> None:
        self.errors.append(message)
        self.error_detail = detail
        self.stage = "error"
        self._halt_open_tasks()

    def apply_stopped(self) -> None:
        self.stage = "stopped"
        self._halt_open_tasks()

    def _halt_open_tasks(self) -> None:
        for task in self.task_rows.values():
            if task["Status"] in ("queued", "running"):
                task["Status"] = "stopped"

    def strip_html(self, running: bool) -> str:
        if self.stage == "idle" and not self.errors:
            return ""
        tone = "running" if running else self.stage
        label = _STAGE_WORDS.get(self.stage, self.stage.replace("_", " ").capitalize())
        fields = [
            _field("iteration", self._iteration_label(running)),
            _field("evidence", self._evidence_count() or ""),
            _field("pages", self._scraped_pages() or ""),
            _field("sources", self._citation_count() or ""),
            _field("citation checks", audit_summary(self.run_stats.get("citation_audit"))),
            _field("model calls", self._model_calls()),
            _field("cost", self._cost_summary()),
            _field("run", self.run_id[:8], extra_class="run"),
        ]
        strip = (
            f"<section class='shandu-strip is-{escape(tone)}'>"
            f"<span class='stage'>{escape(label)}</span>"
            f"{''.join(fields)}"
            "</section>"
        )
        if running or self.stage == "complete":
            strip += self._steps_html()
        if self.stage == "stopped" and self.run_id:
            run = escape(self.run_id[:8])
            strip += (
                "<p class='shandu-stop-note'>Progress up to the last finished iteration is "
                f"saved as run {run}. Continue it with <code>shandu resume {run}</code>.</p>"
            )
        if not self.errors:
            return strip
        faults = "".join(
            f"<p class='shandu-fault'>{escape(message)}</p>"
            for message in self.errors[-2:]
        )
        if self.error_detail:
            faults += f"<p class='shandu-fault-detail'>{escape(self.error_detail)}</p>"
        return strip + faults

    def _steps_html(self) -> str:
        active = len(_STEPS) if self.stage == "complete" else _STAGE_STEP.get(self.stage)
        items = []
        for index, label in enumerate(_STEPS):
            state = ""
            if active is not None:
                if index < active:
                    state = " class='is-done'"
                elif index == active:
                    state = " class='is-live'"
            items.append(f"<li{state}>{label}</li>")
        return f"<ol class='shandu-steps'>{''.join(items)}</ol>"

    def title_html(self) -> str:
        for line in self.report_markdown.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                return f"<h1 class='shandu-doc-title'>{escape(title)}</h1>" if title else ""
        return ""

    def body_markdown(self) -> str:
        """The report without its title, which renders on its own so the
        contents list can sit under it. Sections carry anchors the contents
        link to, and each citation marker links to its reference."""
        lines = self.report_markdown.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("# "):
                del lines[index]
                break
        section = 0
        in_references = False
        for index, line in enumerate(lines):
            if line.startswith("## "):
                in_references = line[3:].strip().lower() == "references"
                if not in_references:
                    section += 1
                    lines[index] = f"## <a id='section-{section}'></a>{line[3:]}"
            elif in_references:
                lines[index] = _REFERENCE_LINE.sub(
                    lambda match: f"- <a id='ref-{match[1]}'></a>", line
                )
            else:
                lines[index] = _MARKER.sub(r"[[\1]](#ref-\1)", line)
        return "\n".join(lines).lstrip("\n")

    def contents_html(self) -> str:
        sections = [
            line[3:].strip()
            for line in self.report_markdown.splitlines()
            if line.startswith("## ") and line[3:].strip().lower() != "references"
        ]
        if len(sections) < 3:
            return ""
        items = "".join(
            f"<li><a href='#section-{number}'>{escape(section)}</a></li>"
            for number, section in enumerate(sections, start=1)
        )
        return (
            "<nav class='shandu-contents'>"
            "<h3>Contents</h3>"
            f"<ol>{items}</ol>"
            "</nav>"
        )

    def event_feed_html(self) -> str:
        if not self.entries:
            return ""
        items = []
        for entry in reversed(self.entries[-60:]):
            meta = [
                text
                for text in (
                    entry["metrics"],
                    f"“{entry['query']}”" if entry["query"] else "",
                    entry["url"],
                )
                if text
            ]
            items.append(
                "<li>"
                f"<time>{_display(entry['time'])}</time>"
                f"<p>{_display(self._readable(entry['message'], entry['task']))}</p>"
                + (f"<span class='meta'>{escape(' · '.join(meta))}</span>" if meta else "")
                + "</li>"
            )
        return (
            "<section class='shandu-feed'>"
            "<h3>Events</h3>"
            f"<ol>{''.join(items)}</ol>"
            "</section>"
        )

    def task_board_html(self) -> str:
        ordered = sorted(self.task_rows.values(), key=lambda row: str(row["Task"]))
        if not ordered:
            return ""
        rows: list[str] = []
        for item in ordered:
            status = str(item["Status"]).strip().lower()
            state_class = _TASK_STATE_CLASS.get(status, "")
            facts = "".join(
                f"<div><dt>{escape(label)}</dt><dd>{_display(item[key])}</dd></div>"
                for label, key in (
                    ("results", "Hits"),
                    ("pages", "Scraped"),
                    ("evidence", "Evidence"),
                )
                if _display(item[key])
            )
            query = _display(item["Last Query"])
            rows.append(
                "<li>"
                "<div class='head'>"
                f"<span class='id'>{_display(item['Label'])}</span>"
                f"<span class='state {state_class}'>{_display(item['Status'])}"
                f" <time>{_display(item['Last Update'])}</time></span>"
                "</div>"
                f"<p>{_display(item['Focus'])}</p>"
                + (f"<dl>{facts}</dl>" if facts else "")
                + (f"<span class='meta'>“{query}”</span>" if query else "")
                + "</li>"
            )
        return (
            "<section class='shandu-tasks'>"
            "<h3>Subagents</h3>"
            f"<ol>{''.join(rows)}</ol>"
            "</section>"
        )

    def run_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "iteration": self.iteration,
            "events": self.event_count,
            "metrics": self.metrics,
            "run_stats": self.run_stats,
            "errors": self.errors,
            "started_at": self.started_at,
        }

    def render(self, running: bool) -> RenderBundle:
        return RenderBundle(
            strip_html=self.strip_html(running=running),
            title_html=self.title_html(),
            contents_html=self.contents_html(),
            feed_html=self.event_feed_html(),
            task_board_html=self.task_board_html(),
            report_markdown=self.body_markdown(),
            run_payload=self.run_payload(),
        )

    def _task(
        self,
        task_id: str,
        now: str,
        default_status: str,
        focus: object = "",
    ) -> dict[str, Any]:
        return self.task_rows.setdefault(
            task_id,
            {
                "Task": task_id,
                "Label": f"Subagent {len(self.task_rows) + 1}",
                "Status": default_status,
                "Focus": str(focus or ""),
                "Last Query": "",
                "Hits": "",
                "Scraped": "",
                "Evidence": "",
                "Last Update": now,
            },
        )

    def _readable(self, message: str, task_id: str) -> str:
        """Engine messages carry raw task ids; the board labels them by arrival."""
        task = self.task_rows.get(task_id)
        if not task or task_id not in message:
            return message
        label = str(task["Label"])
        return message.replace(f"Task {task_id}", label).replace(task_id, label)

    def _scraped_pages(self) -> int:
        return sum(
            int(task["Scraped"])
            for task in self.task_rows.values()
            if str(task["Scraped"]).isdigit()
        )

    def _iteration_label(self, running: bool) -> str:
        # The live counter tracks the current loop; the finished run has a total.
        total = self.run_stats.get("iterations")
        if isinstance(total, int) and not running:
            return str(total)
        if not self.iteration:
            return ""
        if running and self.max_iterations:
            return f"{self.iteration} of {self.max_iterations}"
        return str(self.iteration)

    def _evidence_count(self) -> object:
        # Events carry per-task counts; the finished run carries the total.
        if "evidence_count" in self.run_stats:
            return self.run_stats["evidence_count"]
        return sum(
            int(task["Evidence"])
            for task in self.task_rows.values()
            if str(task["Evidence"]).isdigit()
        )

    def _citation_count(self) -> int:
        count = self.run_stats.get("citation_count")
        if isinstance(count, int):
            return count
        return len(self.citations)

    def _model_calls(self) -> int | None:
        value = self.run_stats.get(
            "agent_model_calls", self.metrics.get("agent_model_calls")
        )
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _cost_summary(self) -> str:
        cost = self.run_stats.get("usd_spent")
        if isinstance(cost, (int, float)) and float(cost) > 0:
            return format_cost(float(cost))
        metered_calls = self.run_stats.get(
            "metered_calls", self.run_stats.get("llm_calls")
        )
        model_calls = self._model_calls()
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and metered_calls < model_calls:
                return f"{metered_calls}/{model_calls} calls metered"
            return f"{metered_calls} calls metered"
        return ""
