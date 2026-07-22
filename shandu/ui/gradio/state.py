from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from typing import Any

from ...contracts import CitationEntry, ResearchRunResult, RunEvent


def _display(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return escape(text if text else "-")


def _metric_text(metrics: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(metrics.items()))


@dataclass(frozen=True, slots=True)
class RenderBundle:
    status_html: str
    lanes_html: str
    event_feed_html: str
    timeline_rows: list[list[Any]]
    task_board_html: str
    trace_rows: list[list[Any]]
    metrics_payload: dict[str, Any]
    report_markdown: str
    citation_rows: list[list[Any]]
    run_payload: dict[str, Any]

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.status_html,
            self.lanes_html,
            self.event_feed_html,
            self.timeline_rows,
            self.task_board_html,
            self.trace_rows,
            self.metrics_payload,
            self.report_markdown,
            self.citation_rows,
            self.run_payload,
        )


@dataclass(slots=True)
class GuiRunState:
    query: str
    run_id: str = "pending"
    stage: str = "idle"
    iteration: int = 0
    event_count: int = 0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics: dict[str, Any] = field(default_factory=dict)
    timeline_rows: list[list[Any]] = field(default_factory=list)
    trace_rows: list[list[Any]] = field(default_factory=list)
    task_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_markdown: str = "Run a research mission to generate a report."
    citations: list[CitationEntry] = field(default_factory=list)
    run_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def apply_event(self, event: RunEvent) -> None:
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
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
        self.timeline_rows.append(
            [now, event.stage, task_id, event.message, _metric_text(event.metrics)]
        )
        self.timeline_rows = self.timeline_rows[-300:]

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

        trace_type = str(event.metrics.get("trace_type", "")).strip()
        if trace_type:
            self._apply_trace_event(
                event=event, now=now, task_id=task_id, trace_type=trace_type
            )

    def apply_result(self, result: ResearchRunResult) -> None:
        self.run_id = result.run_id
        self.report_markdown = result.report_markdown
        self.citations = result.citations
        self.run_stats = result.run_stats
        self.metrics.update(result.run_stats)

    def apply_error(self, message: str) -> None:
        self.errors.append(message)
        self.stage = "error"

    def status_markdown(self, running: bool) -> str:
        state_label = "RUNNING" if running else self.stage.upper()
        lines = [
            "## Mission Status",
            f"- State: **{state_label}**",
            f"- Run ID: **{self.run_id}**",
            f"- Iteration: **{self.iteration}**",
            f"- Events: **{self.event_count}**",
            f"- Query: `{self.query}`",
        ]
        model_calls = self._model_calls()
        if isinstance(model_calls, int) and model_calls > 0:
            lines.append(f"- Model Calls: **{model_calls}**")
        metered_calls = self.run_stats.get(
            "metered_calls", self.run_stats.get("llm_calls")
        )
        coverage = self._cost_coverage(
            metered_calls=metered_calls, model_calls=model_calls
        )
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                label = "partial" if coverage == "partial" else "full"
                lines.append(
                    f"- Cost Coverage: **{label} ({metered_calls}/{model_calls})**"
                )
            else:
                lines.append(f"- Metered Calls: **{metered_calls}**")
        cost = self.run_stats.get("usd_spent")
        if isinstance(cost, (int, float)) and float(cost) > 0:
            label = "Metered Cost" if coverage == "partial" else "Cost"
            lines.append(f"- {label}: **${float(cost):.6f}**")
        if self.errors:
            lines.append("")
            lines.append("### Errors")
            lines.extend(f"- {err}" for err in self.errors[-3:])
        return "\n".join(lines)

    def status_html(self, running: bool) -> str:
        state_label = "RUNNING" if running else self.stage.upper()
        status_class = "is-running" if running else f"is-{self.stage.lower()}"
        query = escape(self.query.strip() or "No query queued")
        cost_line = self._cost_summary()
        error_block = ""
        if self.errors:
            errors = "".join(f"<li>{escape(err)}</li>" for err in self.errors[-3:])
            error_block = f"<ul class='shandu-status-errors'>{errors}</ul>"
        return (
            "<section class='shandu-status-panel'>"
            "<div>"
            f"<span class='shandu-status-pill {status_class}'>{escape(state_label)}</span>"
            "<h2>Mission Status</h2>"
            f"<p>{query}</p>"
            "</div>"
            "<dl>"
            f"<div><dt>Run</dt><dd>{_display(self.run_id)}</dd></div>"
            f"<div><dt>Iteration</dt><dd>{self.iteration}</dd></div>"
            f"<div><dt>Events</dt><dd>{self.event_count}</dd></div>"
            f"<div><dt>Cost</dt><dd>{escape(cost_line)}</dd></div>"
            "</dl>"
            f"{error_block}"
            "</section>"
        )

    def lane_html(self) -> str:
        active_tasks = sum(
            1 for task in self.task_rows.values() if task.get("Status") == "running"
        )
        completed_tasks = sum(
            1 for task in self.task_rows.values() if task.get("Status") == "completed"
        )
        scraped = sum(
            int(task.get("Scraped") or 0)
            for task in self.task_rows.values()
            if str(task.get("Scraped") or "").isdigit()
        )
        citations = self.run_stats.get("citation_count", len(self.citations))
        model_calls = self._model_calls()
        metered_calls = self.run_stats.get(
            "metered_calls", self.run_stats.get("llm_calls")
        )
        coverage = self._cost_coverage(
            metered_calls=metered_calls, model_calls=model_calls
        )
        metered = "-"
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                metered = f"{coverage} {metered_calls}/{model_calls}"
            else:
                metered = str(metered_calls)

        cards: tuple[tuple[str, str, tuple[tuple[str, object], ...]], ...] = (
            (
                "Lead Orchestrator",
                "lead",
                (
                    ("Stage", self.stage),
                    ("Iteration", self.iteration),
                    (
                        "Model calls",
                        model_calls
                        if isinstance(model_calls, int) and model_calls > 0
                        else "-",
                    ),
                ),
            ),
            (
                "Search Subagents",
                "search",
                (
                    ("Active", active_tasks),
                    ("Completed", completed_tasks),
                    ("Evidence", self.metrics.get("evidence", "-")),
                ),
            ),
            (
                "Scrape Pipeline",
                "scrape",
                (
                    ("Pages scraped", scraped),
                    ("Events", self.event_count),
                    ("Metered", metered),
                ),
            ),
            (
                "Citation Agent",
                "cite",
                (
                    ("Citations", citations),
                    ("Cost", self._cost_summary()),
                    ("Run", self.run_id),
                ),
            ),
        )

        html = ["<section class='shandu-lane-grid'>"]
        for title, tone, metrics in cards:
            html.append(f"<article class='shandu-lane-card lane-{tone}'>")
            html.append(f"<header><span></span><h3>{escape(title)}</h3></header>")
            html.append("<dl>")
            for label, value in metrics:
                html.append(
                    f"<div><dt>{escape(label)}</dt><dd>{_display(value)}</dd></div>"
                )
            html.append("</dl></article>")
        html.append("</section>")
        return "".join(html)

    def event_feed_html(self) -> str:
        if not self.timeline_rows:
            return (
                "<section class='shandu-feed empty'>"
                "<header><h3>Live Feed</h3><span>Idle</span></header>"
                "<p>Events will appear here once a mission starts.</p>"
                "</section>"
            )
        rows = list(reversed(self.timeline_rows[-40:]))
        items = []
        for time_value, stage, task, message, metrics in rows:
            task_markup = f"<span>{escape(str(task))}</span>" if task else ""
            metric_markup = f"<small>{escape(str(metrics))}</small>" if metrics else ""
            items.append(
                "<li>"
                f"<time>{escape(str(time_value))}</time>"
                f"<b>{escape(str(stage).upper())}</b>"
                f"{task_markup}"
                f"<p>{escape(str(message))}</p>"
                f"{metric_markup}"
                "</li>"
            )
        return (
            "<section class='shandu-feed'>"
            "<header><h3>Live Feed</h3><span>Latest first</span></header>"
            f"<ol>{''.join(items)}</ol>"
            "</section>"
        )

    def task_table(self) -> list[list[Any]]:
        ordered = sorted(self.task_rows.values(), key=lambda row: str(row["Task"]))
        return [
            [
                item["Task"],
                item["Status"],
                item["Focus"],
                item["Last Query"],
                item["Hits"],
                item["Scraped"],
                item["Evidence"],
                item["Last Update"],
            ]
            for item in ordered
        ]

    def task_board_html(self) -> str:
        ordered = sorted(self.task_rows.values(), key=lambda row: str(row["Task"]))
        if not ordered:
            return (
                "<section class='shandu-task-board empty'>"
                "<p>Tasks will appear once a mission starts.</p>"
                "</section>"
            )

        rows: list[str] = []
        for item in ordered:
            status = str(item["Status"]).strip().lower()
            status_class = {
                "completed": "is-complete",
                "running": "is-running",
                "failed": "is-error",
            }.get(status, "is-idle")
            rows.append(
                "<li>"
                "<div class='task-row-head'>"
                f"<span class='task-id'>{_display(item['Task'])}</span>"
                f"<span class='task-status {status_class}'>{_display(item['Status'])}</span>"
                f"<time>{_display(item['Last Update'])}</time>"
                "</div>"
                f"<p>{_display(item['Focus'])}</p>"
                "<dl>"
                f"<div class='task-query'><dt>Last query</dt><dd>{_display(item['Last Query'])}</dd></div>"
                f"<div><dt>Hits</dt><dd>{_display(item['Hits'])}</dd></div>"
                f"<div><dt>Scraped</dt><dd>{_display(item['Scraped'])}</dd></div>"
                f"<div><dt>Evidence</dt><dd>{_display(item['Evidence'])}</dd></div>"
                "</dl>"
                "</li>"
            )
        return f"<section class='shandu-task-board'><ol>{''.join(rows)}</ol></section>"

    def citation_table(self) -> list[list[Any]]:
        return [
            [
                citation.citation_id,
                citation.publisher,
                citation.title,
                citation.url,
                citation.accessed_at,
            ]
            for citation in self.citations
        ]

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "iteration": self.iteration,
            "events": self.event_count,
            "metrics": self.metrics,
            "run_stats": self.run_stats,
        }

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
            status_html=self.status_html(running=running),
            lanes_html=self.lane_html(),
            event_feed_html=self.event_feed_html(),
            timeline_rows=self.timeline_rows[-120:],
            task_board_html=self.task_board_html(),
            trace_rows=self.trace_rows[-160:],
            metrics_payload=self.metrics_payload(),
            report_markdown=self.report_markdown,
            citation_rows=self.citation_table(),
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
                "Status": default_status,
                "Focus": str(focus or ""),
                "Last Query": "",
                "Hits": "",
                "Scraped": "",
                "Evidence": "",
                "Last Update": now,
            },
        )

    def _apply_trace_event(
        self, event: RunEvent, now: str, task_id: str, trace_type: str
    ) -> None:
        query = str(event.payload.get("query", "")).strip()
        url = str(event.payload.get("url", "")).strip()
        details = []
        for key in (
            "hits",
            "max_results",
            "url_count",
            "scraped",
            "missed",
            "confidence",
            "credibility",
        ):
            if key in event.metrics:
                details.append(f"{key}={event.metrics[key]}")
            elif key in event.payload:
                details.append(f"{key}={event.payload[key]}")
        if task_id:
            task = self._task(
                task_id, now, default_status="running", focus=event.payload.get("focus")
            )
            if query:
                task["Last Query"] = query
            if "hits" in event.metrics:
                task["Hits"] = str(event.metrics["hits"])
            if "scraped" in event.metrics:
                task["Scraped"] = str(event.metrics["scraped"])
        self.trace_rows.append(
            [now, task_id, trace_type, query, url, ", ".join(details)]
        )
        self.trace_rows = self.trace_rows[-300:]

    def _cost_coverage(self, metered_calls: object, model_calls: object) -> str:
        coverage = str(self.run_stats.get("cost_coverage", "")).strip()
        if coverage in {"partial", "full"}:
            return coverage
        if (
            isinstance(metered_calls, int)
            and metered_calls > 0
            and isinstance(model_calls, int)
            and model_calls > 0
        ):
            return "partial" if metered_calls < model_calls else "full"
        return ""

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
            return f"${float(cost):.6f}"
        metered_calls = self.run_stats.get(
            "metered_calls", self.run_stats.get("llm_calls")
        )
        if isinstance(metered_calls, int) and metered_calls > 0:
            return f"{metered_calls} metered"
        return "-"
