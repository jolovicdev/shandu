from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console

from shandu.contracts import CitationEntry, ResearchRequest, ResearchRunResult
from shandu.ui.rich_frontend import ShanduUI


def test_result_panels_render() -> None:
    console = Console(record=True, width=140)
    ui = ShanduUI(console=console)
    request = ResearchRequest(query="q")
    snapshot = ui.new_snapshot(request, model="deepseek/deepseek-chat")
    console.print(ui.dashboard(snapshot))
    output = console.export_text()
    assert "Control Plane" in output


def test_result_panels_render_cost_when_available() -> None:
    console = Console(record=True, width=160)
    ui = ShanduUI(console=console)
    request = ResearchRequest(query="q")
    result = ResearchRunResult(
        run_id="run-1",
        request=request,
        report_markdown="# R",
        citations=[
            CitationEntry(
                citation_id=1,
                evidence_ids=["e1"],
                url="https://example.com",
                title="Example",
                publisher="example.com",
                accessed_at=datetime.now(timezone.utc).date().isoformat(),
            )
        ],
        evidence=[],
        iteration_summaries=[],
        run_stats={
            "iterations": 1,
            "evidence_count": 0,
            "citation_count": 1,
            "elapsed_seconds": 1.2,
            "agent_model_calls": 9,
            "usd_spent": 0.012345,
            "llm_calls": 6,
            "llm_tokens": 1234,
        },
    )
    console.print(ui.result_panels(result))
    output = console.export_text()

    assert "Cost Coverage" in output
    assert "Metered Cost" in output
    assert "Model Calls" in output
    assert "LLM Tokens" in output


def test_result_panels_hide_cost_when_unavailable() -> None:
    console = Console(record=True, width=160)
    ui = ShanduUI(console=console)
    request = ResearchRequest(query="q")
    result = ResearchRunResult(
        run_id="run-1",
        request=request,
        report_markdown="# R",
        citations=[],
        evidence=[],
        iteration_summaries=[],
        run_stats={
            "iterations": 1,
            "evidence_count": 0,
            "citation_count": 0,
            "elapsed_seconds": 1.1,
        },
    )
    console.print(ui.result_panels(result))
    output = console.export_text()

    assert "USD Spent" not in output
    assert "Metered Cost" not in output
