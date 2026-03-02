from __future__ import annotations

import gradio as gr
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import config, infer_api_key_env_name
from ..contracts import CitationEntry, ResearchRequest, ResearchRunResult, RunEvent
from ..engine import ShanduEngine
from ..interfaces import DepthPolicy, DetailLevel
from ..runtime import reset_bootstrap

_DETAIL_LEVELS: tuple[DetailLevel, ...] = ("concise", "standard", "high")
_DEPTH_POLICIES: tuple[DepthPolicy, ...] = ("adaptive", "fixed")

_TIMELINE_HEADERS = ["Time", "Stage", "Task", "Message", "Metrics"]
_TASK_HEADERS = ["Task", "Status", "Focus", "Last Query", "Hits", "Scraped", "Evidence", "Last Update"]
_TRACE_HEADERS = ["Time", "Task", "Trace", "Query", "URL", "Details"]
_CITATION_HEADERS = ["#", "Publisher", "Title", "URL", "Accessed"]


@dataclass(slots=True)
class GuiRunState:
    query: str
    run_id: str = "pending"
    stage: str = "idle"
    iteration: int = 0
    event_count: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    timeline_rows: list[list[Any]] = field(default_factory=list)
    trace_rows: list[list[Any]] = field(default_factory=list)
    task_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_markdown: str = "Run a query to generate a report."
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
        if event.payload.get("run_id"):
            self.run_id = str(event.payload["run_id"])

        task_id = str(event.payload.get("task_id", "")).strip()
        metric_text = ", ".join(f"{key}={value}" for key, value in sorted(event.metrics.items()))
        self.timeline_rows.append([now, event.stage, task_id, event.message, metric_text])
        self.timeline_rows = self.timeline_rows[-300:]

        if task_id:
            task = self.task_rows.setdefault(
                task_id,
                {
                    "Task": task_id,
                    "Status": "queued",
                    "Focus": str(event.payload.get("focus", "")),
                    "Last Query": "",
                    "Hits": "",
                    "Scraped": "",
                    "Evidence": "",
                    "Last Update": now,
                },
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
            query = str(event.payload.get("query", "")).strip()
            url = str(event.payload.get("url", "")).strip()
            details = []
            for key in ("hits", "max_results", "url_count", "scraped", "missed", "confidence"):
                if key in event.metrics:
                    details.append(f"{key}={event.metrics[key]}")
                elif key in event.payload:
                    details.append(f"{key}={event.payload[key]}")
            if task_id:
                task = self.task_rows.setdefault(
                    task_id,
                    {
                        "Task": task_id,
                        "Status": "running",
                        "Focus": str(event.payload.get("focus", "")),
                        "Last Query": "",
                        "Hits": "",
                        "Scraped": "",
                        "Evidence": "",
                        "Last Update": now,
                    },
                )
                if query:
                    task["Last Query"] = query
                if "hits" in event.metrics:
                    task["Hits"] = str(event.metrics["hits"])
                if "scraped" in event.metrics:
                    task["Scraped"] = str(event.metrics["scraped"])
            self.trace_rows.append([now, task_id, trace_type, query, url, ", ".join(details)])
            self.trace_rows = self.trace_rows[-300:]

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
        model_calls = self.run_stats.get("agent_model_calls")
        if isinstance(model_calls, int) and model_calls > 0:
            lines.append(f"- Model Calls: **{model_calls}**")
        metered_calls = self.run_stats.get("metered_calls", self.run_stats.get("llm_calls"))
        coverage = str(self.run_stats.get("cost_coverage", "")).strip()
        if coverage not in {"partial", "full"}:
            if isinstance(metered_calls, int) and metered_calls > 0 and isinstance(model_calls, int) and model_calls > 0:
                coverage = "partial" if metered_calls < model_calls else "full"
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                if coverage == "partial":
                    lines.append(f"- Cost Coverage: **partial ({metered_calls}/{model_calls})**")
                else:
                    lines.append(f"- Cost Coverage: **full ({metered_calls}/{model_calls})**")
            else:
                lines.append(f"- Metered Calls: **{metered_calls}**")
        usd_spent = self.run_stats.get("usd_spent")
        if isinstance(usd_spent, (int, float)) and float(usd_spent) > 0:
            if coverage == "partial":
                lines.append(f"- Metered Cost: **${float(usd_spent):.6f}**")
            else:
                lines.append(f"- Cost: **${float(usd_spent):.6f}**")
        if self.errors:
            lines.append("")
            lines.append("### Errors")
            for err in self.errors[-3:]:
                lines.append(f"- {err}")
        return "\n".join(lines)

    def task_table(self) -> list[list[Any]]:
        ordered = sorted(self.task_rows.values(), key=lambda row: str(row["Task"]))
        rows: list[list[Any]] = []
        for item in ordered:
            rows.append(
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
            )
        return rows

    def citation_table(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for citation in self.citations:
            rows.append(
                [
                    citation.citation_id,
                    citation.publisher,
                    citation.title,
                    citation.url,
                    citation.accessed_at,
                ]
            )
        return rows

    def lane_html(self) -> str:
        active_tasks = sum(1 for task in self.task_rows.values() if task.get("Status") == "running")
        completed_tasks = sum(
            1 for task in self.task_rows.values() if task.get("Status") == "completed"
        )
        scraped = sum(
            int(task.get("Scraped") or 0)
            for task in self.task_rows.values()
            if str(task.get("Scraped") or "").isdigit()
        )
        citations = self.run_stats.get("citation_count", len(self.citations))
        model_calls = self.run_stats.get("agent_model_calls")
        model_line = ""
        if isinstance(model_calls, int) and model_calls > 0:
            model_line = f"<p>model calls: <b>{model_calls}</b></p>"
        metered_calls = self.run_stats.get("metered_calls", self.run_stats.get("llm_calls"))
        coverage = str(self.run_stats.get("cost_coverage", "")).strip()
        if coverage not in {"partial", "full"}:
            if isinstance(metered_calls, int) and metered_calls > 0 and isinstance(model_calls, int) and model_calls > 0:
                coverage = "partial" if metered_calls < model_calls else "full"
        metered_line = ""
        if isinstance(metered_calls, int) and metered_calls > 0:
            if isinstance(model_calls, int) and model_calls > 0:
                if coverage == "partial":
                    metered_line = (
                        f"<p>cost coverage: <b>partial ({metered_calls}/{model_calls})</b></p>"
                    )
                else:
                    metered_line = f"<p>cost coverage: <b>full ({metered_calls}/{model_calls})</b></p>"
            else:
                metered_line = f"<p>metered calls: <b>{metered_calls}</b></p>"
        cost_value = self.run_stats.get("usd_spent")
        cost_line = ""
        if isinstance(cost_value, (int, float)) and float(cost_value) > 0:
            if coverage == "partial":
                cost_line = f"<p>metered cost: <b>${float(cost_value):.6f}</b></p>"
            else:
                cost_line = f"<p>cost: <b>${float(cost_value):.6f}</b></p>"
        return (
            "<div class='lane-grid'>"
            "<div class='lane-card lane-lead'><h3>Lead Orchestrator</h3>"
            f"<p>stage: <b>{self.stage}</b></p><p>iteration: <b>{self.iteration}</b></p>{model_line}</div>"
            "<div class='lane-card lane-search'><h3>Search Subagents</h3>"
            f"<p>active: <b>{active_tasks}</b></p><p>completed: <b>{completed_tasks}</b></p></div>"
            "<div class='lane-card lane-scrape'><h3>Scrape Pipeline</h3>"
            f"<p>pages scraped: <b>{scraped}</b></p><p>events: <b>{self.event_count}</b></p>{metered_line}</div>"
            "<div class='lane-card lane-cite'><h3>Citation Agent</h3>"
            f"<p>citations: <b>{citations}</b></p>{cost_line}<p>run: <b>{self.run_id}</b></p></div>"
            "</div>"
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


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary: #0a0e14;
  --bg-secondary: #111820;
  --bg-tertiary: #1a222d;
  --bg-card: #141b24;
  --text-primary: #e6edf3;
  --text-secondary: #7d8590;
  --text-muted: #484f58;
  --accent-cyan: #39c5cf;
  --accent-amber: #f0b429;
  --accent-magenta: #db61a2;
  --accent-green: #3fb950;
  --border-color: #30363d;
  --glow-cyan: 0 0 20px rgba(57, 197, 207, 0.4);
  --glow-amber: 0 0 20px rgba(240, 180, 41, 0.4);
}

@media (prefers-color-scheme: light) {
  :root {
    --bg-primary: #f8f9fa;
    --bg-secondary: #ffffff;
    --bg-tertiary: #e9ecef;
    --bg-card: #ffffff;
    --text-primary: #212529;
    --text-secondary: #495057;
    --text-muted: #6c757d;
    --accent-cyan: #0d9488;
    --accent-amber: #d97706;
    --accent-magenta: #be185d;
    --accent-green: #16a34a;
    --border-color: #dee2e6;
    --glow-cyan: 0 0 10px rgba(13, 148, 136, 0.3);
    --glow-amber: 0 0 10px rgba(217, 119, 6, 0.3);
  }
}

* { box-sizing: border-box; }

.gradio-container {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'JetBrains Mono', monospace !important;
}

.gradio-container::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(57, 197, 207, 0.015) 2px, rgba(57, 197, 207, 0.015) 4px),
    radial-gradient(ellipse at 20% 20%, rgba(57, 197, 207, 0.08) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(219, 97, 162, 0.06) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
}

@media (prefers-color-scheme: light) {
  .gradio-container::before {
    background: radial-gradient(ellipse at 20% 20%, rgba(13, 148, 136, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(190, 24, 93, 0.04) 0%, transparent 50%);
  }
}

.gradio-container > * { position: relative; z-index: 1; }

.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container label {
  font-family: 'Oxanium', sans-serif !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.shandu-title {
  border: 1px solid var(--border-color);
  border-left: 4px solid var(--accent-cyan);
  padding: 1.5rem;
  background: linear-gradient(135deg, rgba(57, 197, 207, 0.05) 0%, transparent 100%);
  margin-bottom: 1.5rem;
}

.shandu-title h1 {
  color: var(--accent-cyan) !important;
  font-size: 2.8rem !important;
  font-weight: 700 !important;
  text-shadow: var(--glow-cyan);
  letter-spacing: 0.15em;
  margin: 0 !important;
  line-height: 1.1 !important;
}

.shandu-title p {
  color: var(--text-secondary) !important;
  font-size: 0.9rem !important;
  margin-top: 0.75rem !important;
  letter-spacing: 0.15em;
  font-weight: 500;
}

.shandu-subtitle {
  color: var(--text-primary) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  border-left: 3px solid var(--accent-amber);
  padding-left: 1rem;
  margin-top: 1rem;
  line-height: 1.5;
}

.query-shell {
  border: 1px solid var(--border-color) !important;
  border-radius: 4px !important;
  background: var(--bg-secondary) !important;
  padding: 1rem !important;
  margin-bottom: 1rem !important;
}

.query-shell .gr-textbox textarea {
  background: var(--bg-tertiary) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 4px !important;
  font-family: 'JetBrains Mono', monospace !important;
  min-height: 100px !important;
}

.query-shell .gr-textbox textarea::placeholder { color: var(--text-muted) !important; }

.run-btn button {
  background: linear-gradient(135deg, var(--accent-cyan), #2563eb) !important;
  color: var(--bg-primary) !important;
  font-family: 'Oxanium', sans-serif !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.15em !important;
  border: none !important;
  border-radius: 4px !important;
  min-height: 54px !important;
  box-shadow: var(--glow-cyan) !important;
  transition: all 0.3s ease !important;
}

.run-btn button:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 30px rgba(57, 197, 207, 0.6) !important;
}

.lane-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
}

.lane-card {
  border: 1px solid var(--border-color);
  background: var(--bg-card) !important;
  border-radius: 4px;
  padding: 1rem;
  position: relative;
  overflow: hidden;
}

.lane-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; width: 3px; height: 100%;
}

.lane-lead::before { background: var(--accent-cyan); }
.lane-search::before { background: var(--accent-green); }
.lane-scrape::before { background: var(--accent-amber); }
.lane-cite::before { background: var(--accent-magenta); }

.lane-card h3 {
  font-size: 0.75rem !important;
  color: var(--text-secondary) !important;
  margin: 0 0 0.5rem 0.5rem !important;
  font-weight: 500 !important;
}

.lane-card p { margin: 0.2rem 0; color: var(--text-muted) !important; font-size: 0.8rem !important; }
.lane-card b { color: var(--text-primary) !important; font-family: 'JetBrains Mono', monospace !important; }

.lane-lead h3 { color: var(--accent-cyan) !important; }
.lane-search h3 { color: var(--accent-green) !important; }
.lane-scrape h3 { color: var(--accent-amber) !important; }
.lane-cite h3 { color: var(--accent-magenta) !important; }

.panel-border { border: 1px solid var(--border-color) !important; border-radius: 4px !important; background: var(--bg-secondary) !important; }

.gradio-container .tab-nav button {
  background: transparent !important;
  color: var(--text-muted) !important;
  border: 1px solid transparent !important;
  border-bottom: 2px solid transparent !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.8rem !important;
}

.gradio-container .tab-nav button.selected {
  color: var(--accent-cyan) !important;
  border-bottom-color: var(--accent-cyan) !important;
  background: transparent !important;
}

.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: var(--bg-tertiary) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 4px !important;
}

.gradio-container input::placeholder, .gradio-container textarea::placeholder { color: var(--text-muted) !important; }

.gradio-json { background: var(--bg-secondary) !important; border: 1px solid var(--border-color) !important; border-radius: 4px !important; }
.gradio-dataframe, .gradio-table { background: var(--bg-secondary) !important; }
.gradio-dataframe table, .gradio-table table { background: var(--bg-secondary) !important; color: var(--text-primary) !important; }
.gradio-dataframe th, .gradio-table th { background: var(--bg-tertiary) !important; color: var(--text-secondary) !important; border-color: var(--border-color) !important; }
.gradio-dataframe td, .gradio-table td { background: var(--bg-secondary) !important; color: var(--text-primary) !important; border-color: var(--border-color) !important; }

.section-note { color: var(--text-muted) !important; font-size: 0.75rem !important; }

.gradio-accordion { border: 1px solid var(--border-color) !important; border-radius: 4px !important; background: var(--bg-secondary) !important; }
.gradio-accordion .summary { background: var(--bg-tertiary) !important; color: var(--text-secondary) !important; }

@media (max-width: 980px) {
  .lane-grid { grid-template-columns: repeat(2, 1fr); }
  .shandu-title h1 { font-size: 2rem !important; }
}

@media (max-width: 640px) {
  .lane-grid { grid-template-columns: 1fr; }
}
"""


def _resolved_detail_level(value: str) -> DetailLevel:
    if value in _DETAIL_LEVELS:
        return value
    return "high"


def _resolved_depth_policy(value: str) -> DepthPolicy:
    if value in _DEPTH_POLICIES:
        return value
    return "adaptive"


def _save_configuration(
    model: object,
    api_key_env: object,
    api_key_value: object,
    temperature: object,
    max_tokens: object,
    max_iterations: object,
    parallelism: object,
    detail_level: object,
    depth_policy: object,
    max_results_per_query: object,
    max_pages_per_task: object,
) -> str:
    model_text = str(model or "").strip() or "deepseek/deepseek-chat"
    env_text = str(api_key_env or "").strip()
    key_text = str(api_key_value or "").strip()
    temperature_value = float(temperature) if temperature is not None else 0.2
    max_tokens_value = int(max_tokens) if max_tokens is not None else 8192
    max_iterations_value = int(max_iterations) if max_iterations is not None else 2
    parallelism_value = int(parallelism) if parallelism is not None else 3
    max_results_value = int(max_results_per_query) if max_results_per_query is not None else 5
    max_pages_value = int(max_pages_per_task) if max_pages_per_task is not None else 3
    detail_text = str(detail_level or "high")
    depth_text = str(depth_policy or "adaptive")

    resolved_env = env_text or infer_api_key_env_name(model_text)
    config.set("api", "model", model_text)
    config.set("api", "api_key_env", resolved_env)
    if key_text:
        config.set("api", "api_key", key_text)
    config.set("api", "temperature", temperature_value)
    config.set("api", "max_tokens", max_tokens_value)
    config.set("orchestration", "max_iterations", max_iterations_value)
    config.set("orchestration", "parallelism", parallelism_value)
    config.set("orchestration", "detail_level", _resolved_detail_level(detail_text))
    config.set("orchestration", "depth_policy", _resolved_depth_policy(depth_text))
    config.set("orchestration", "max_results_per_query", max_results_value)
    config.set("orchestration", "max_pages_per_task", max_pages_value)
    config.save()
    reset_bootstrap()
    config.apply_provider_api_key()
    return f"Saved configuration for `{model_text}` using env key `{resolved_env}`."


def _persist_report_markdown(run_id: str, markdown: str) -> str | None:
    text = markdown.strip()
    if not text:
        return None
    try:
        storage = Path(str(config.get("runtime", "storage_dir", ".blackgeorge")))
        export_dir = storage / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        safe_run = "".join(char if char.isalnum() else "_" for char in run_id).strip("_") or "report"
        file_path = export_dir / f"{safe_run}.md"
        file_path.write_text(text, encoding="utf-8")
        return str(file_path)
    except Exception:
        return None


def _render_bundle(state: GuiRunState, running: bool) -> tuple[Any, ...]:
    return (
        state.status_markdown(running=running),
        state.lane_html(),
        state.timeline_rows[-120:],
        state.task_table(),
        state.trace_rows[-160:],
        state.run_payload(),
        state.report_markdown,
        state.citation_table(),
        state.run_payload(),
    )


def build_gui() -> Any:
    default_model = str(config.get("api", "model", "deepseek/deepseek-chat"))
    default_api_env = config.get_api_key_env_name(default_model)
    default_temperature = float(config.get("api", "temperature", 0.2))
    default_max_tokens = int(config.get("api", "max_tokens", 8192))
    default_iterations = int(config.get("orchestration", "max_iterations", 2))
    default_parallelism = int(config.get("orchestration", "parallelism", 3))
    default_detail = str(config.get("orchestration", "detail_level", "high"))
    default_depth = str(config.get("orchestration", "depth_policy", "adaptive"))
    default_results = int(config.get("orchestration", "max_results_per_query", 5))
    default_pages = int(config.get("orchestration", "max_pages_per_task", 3))

    with gr.Blocks(
        title="Shandu GUI",
    ) as demo:
        gr.HTML(
            """
            <section class="shandu-title">
              <h1>SHANDU CONTROL ROOM</h1>
              <p>LeadResearcher · Subagents · Memory · CitationAgent · Live Telemetry</p>
              <div class="shandu-subtitle">
                Configure model/runtime and watch subagent search/scrape/extract flow in real time while the report is synthesized.
              </div>
            </section>
            """
        )

        with gr.Row(elem_classes=["query-shell"]):
            query = gr.Textbox(
                label="Research Query",
                lines=3,
                info="Write a concrete objective. Example: compare 3 cloud GPU providers for startups in 2026.",
                elem_classes=["panel-border"],
            )
            run_button = gr.Button("Run Mission", variant="primary", elem_classes=["run-btn"])
        gr.Examples(
            examples=[
                "Map likely labor-market shifts in Southeast Europe by 2035 and justify assumptions.",
                "Compare open-source browser automation frameworks in 2026 for reliability and speed.",
                "Which AI agent frameworks are strongest for enterprise workflow automation in 2026?",
            ],
            inputs=query,
            label="Quick start prompts",
        )

        with gr.Accordion("Runtime Configuration", open=False):
            gr.Markdown(
                "<div class='section-note'>API key value is optional only if key is already set in shell/config, or your model/provider does not require a key (for example local models).</div>"
            )
            with gr.Row():
                model = gr.Textbox(label="Model", value=default_model)
                api_key_env = gr.Textbox(label="API key env var", value=default_api_env)
                api_key_value = gr.Textbox(
                    label="API key value",
                    info="Leave empty only when key already exists in env/config or model is local/no-key.",
                    type="password",
                )
            with gr.Row():
                temperature = gr.Slider(0.0, 1.0, value=default_temperature, step=0.05, label="Temperature")
                max_tokens = gr.Number(value=default_max_tokens, label="Max tokens", precision=0)
                max_iterations = gr.Slider(1, 8, value=default_iterations, step=1, label="Max iterations")
                parallelism = gr.Slider(1, 8, value=default_parallelism, step=1, label="Parallelism")
            with gr.Row():
                detail_level = gr.Dropdown(
                    choices=list(_DETAIL_LEVELS),
                    value=default_detail if default_detail in _DETAIL_LEVELS else "high",
                    label="Detail level",
                )
                depth_policy = gr.Dropdown(
                    choices=list(_DEPTH_POLICIES),
                    value=default_depth if default_depth in _DEPTH_POLICIES else "adaptive",
                    label="Depth policy",
                )
                max_results_per_query = gr.Slider(
                    1,
                    20,
                    value=default_results,
                    step=1,
                    label="Max results per query",
                )
                max_pages_per_task = gr.Slider(
                    1,
                    10,
                    value=default_pages,
                    step=1,
                    label="Max pages per task",
                )
            save_button = gr.Button("Save Configuration", variant="secondary")
            save_message = gr.Markdown("Configuration state is loaded from your local Shandu config.")

        with gr.Tabs():
            with gr.Tab("Live Ops"):
                status = gr.Markdown("## Mission Status\n- State: **IDLE**")
                lane_view = gr.HTML(
                    "<div class='lane-grid'><div class='lane-card'><h3>Waiting</h3><p>Run a mission to stream telemetry.</p></div></div>"
                )
                timeline = gr.Dataframe(
                    headers=_TIMELINE_HEADERS,
                    datatype=["str", "str", "str", "str", "str"],
                    wrap=True,
                    interactive=False,
                )
                tasks = gr.Dataframe(
                    headers=_TASK_HEADERS,
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                    wrap=True,
                    interactive=False,
                )
            with gr.Tab("Search + Scrape"):
                traces = gr.Dataframe(
                    headers=_TRACE_HEADERS,
                    datatype=["str", "str", "str", "str", "str", "str"],
                    wrap=True,
                    interactive=False,
                )
                metrics = gr.JSON(label="Runtime Metrics", value={})
            with gr.Tab("Final Report"):
                report = gr.Markdown("Run a mission to generate markdown output.")
                download_report = gr.DownloadButton("Download report markdown", visible=False)
                citations = gr.Dataframe(
                    headers=_CITATION_HEADERS,
                    datatype=["number", "str", "str", "str", "str"],
                    wrap=True,
                    interactive=False,
                )
            with gr.Tab("Run Payload"):
                payload = gr.JSON(label="Run Payload", value={})

        def save_action(
            model_value: str,
            api_env_value: str,
            api_value: str,
            temp_value: float,
            token_value: float,
            iter_value: float,
            par_value: float,
            detail_value: str,
            depth_value: str,
            results_value: float,
            pages_value: float,
        ) -> str:
            return _save_configuration(
                model=model_value,
                api_key_env=api_env_value,
                api_key_value=api_value,
                temperature=float(temp_value),
                max_tokens=int(token_value),
                max_iterations=int(iter_value),
                parallelism=int(par_value),
                detail_level=detail_value,
                depth_policy=depth_value,
                max_results_per_query=int(results_value),
                max_pages_per_task=int(pages_value),
            )

        def run_action(
            query_value: str,
            model_value: str,
            api_env_value: str,
            api_value: str,
            temp_value: float,
            token_value: float,
            iter_value: float,
            par_value: float,
            detail_value: str,
            depth_value: str,
            results_value: float,
            pages_value: float,
        ):
            def download_update(path: str | None):
                if path:
                    name = Path(path).name
                    return gr.DownloadButton(
                        label=f"Download {name}",
                        value=path,
                        visible=True,
                    )
                return gr.DownloadButton(visible=False, value=None)

            text = query_value.strip()
            if not text:
                state = GuiRunState(query="")
                state.apply_error("Query is required.")
                yield (*_render_bundle(state, running=False), download_update(None))
                return

            _save_configuration(
                model=model_value,
                api_key_env=api_env_value,
                api_key_value=api_value,
                temperature=float(temp_value),
                max_tokens=int(token_value),
                max_iterations=int(iter_value),
                parallelism=int(par_value),
                detail_level=detail_value,
                depth_policy=depth_value,
                max_results_per_query=int(results_value),
                max_pages_per_task=int(pages_value),
            )

            request = ResearchRequest(
                query=text,
                max_iterations=int(iter_value),
                parallelism=int(par_value),
                detail_level=_resolved_detail_level(detail_value),
                depth_policy=_resolved_depth_policy(depth_value),
                max_results_per_query=int(results_value),
                max_pages_per_task=int(pages_value),
            )
            state = GuiRunState(query=text)
            state.stage = "bootstrap"
            yield (*_render_bundle(state, running=True), download_update(None))

            event_queue: queue.Queue[RunEvent | None] = queue.Queue()
            result_box: dict[str, Any] = {}
            error_box: dict[str, str] = {}

            def on_event(event: RunEvent) -> None:
                event_queue.put(event)

            def run_worker() -> None:
                try:
                    engine = ShanduEngine.from_config()
                    result_box["result"] = engine.run_sync(request, progress_callback=on_event)
                except Exception as exc:
                    error_box["error"] = str(exc)
                finally:
                    event_queue.put(None)

            thread = threading.Thread(target=run_worker, daemon=True)
            thread.start()

            while True:
                event = event_queue.get()
                if event is None:
                    break
                state.apply_event(event)
                yield (*_render_bundle(state, running=True), download_update(None))

            if "error" in error_box:
                state.apply_error(error_box["error"])
                yield (*_render_bundle(state, running=False), download_update(None))
                return

            result = result_box.get("result")
            if isinstance(result, ResearchRunResult):
                state.apply_result(result)
                state.stage = "complete"
                download_path = _persist_report_markdown(result.run_id, result.report_markdown)
                yield (*_render_bundle(state, running=False), download_update(download_path))
                return
            else:
                state.apply_error("Run did not return a valid result.")
            yield (*_render_bundle(state, running=False), download_update(None))

        config_inputs = [
            model,
            api_key_env,
            api_key_value,
            temperature,
            max_tokens,
            max_iterations,
            parallelism,
            detail_level,
            depth_policy,
            max_results_per_query,
            max_pages_per_task,
        ]

        save_button.click(
            fn=save_action,
            inputs=config_inputs,
            outputs=[save_message],
        )

        run_button.click(
            fn=run_action,
            inputs=[query] + config_inputs,
            outputs=[
                status,
                lane_view,
                timeline,
                tasks,
                traces,
                metrics,
                report,
                citations,
                payload,
                download_report,
            ],
        )

    demo.queue(default_concurrency_limit=1, max_size=12)
    return demo


def launch_gui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
) -> None:
    demo = build_gui()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
        css=_CSS,
        theme=gr.themes.Default(primary_hue="teal", neutral_hue="slate"),
    )
