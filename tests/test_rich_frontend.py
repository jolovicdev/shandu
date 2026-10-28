from __future__ import annotations

from datetime import datetime, timezone

import pytest
from rich.console import Console

from shandu.contracts import CitationEntry, ResearchRequest, ResearchRunResult
from shandu.ui.rich_frontend import ShanduUI


def test_recap_renders_without_events() -> None:
    console = Console(record=True, width=140)
    ui = ShanduUI(console=console)
    request = ResearchRequest(query="q")
    snapshot = ui.new_snapshot(request, model="deepseek/deepseek-v4-flash")
    console.print(ui.recap(snapshot))
    output = console.export_text()
    assert "deepseek/deepseek-v4-flash" in output
    assert "No events recorded" in output


@pytest.mark.parametrize(
    "extra_stats,present,absent",
    [
        (
            {
                "agent_model_calls": 9,
                "usd_spent": 0.012345,
                "llm_calls": 6,
                "llm_tokens": 1234,
            },
            ["cost coverage", "metered cost", "model calls", "LLM tokens"],
            [],
        ),
        (
            {},
            [],
            ["USD spent", "metered cost"],
        ),
    ],
)
def test_result_summary_shows_cost_only_when_available(
    extra_stats, present, absent
) -> None:
    console = Console(record=True, width=160)
    ui = ShanduUI(console=console)
    request = ResearchRequest(query="q")
    run_stats = {
        "iterations": 1,
        "evidence_count": 0,
        "citation_count": 1,
        "elapsed_seconds": 1.2,
    }
    run_stats.update(extra_stats)
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
        run_stats=run_stats,
    )
    console.print(ui.result_summary(result))
    output = console.export_text()

    for text in present:
        assert text in output
    for text in absent:
        assert text not in output


def test_inspect_summary_shows_run_id_and_stats() -> None:
    console = Console(record=True, width=200)
    ui = ShanduUI(console=console)
    payload = {
        "run_id": "run-abc-123",
        "status": "completed",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-02",
        "output_json": {
            "run_stats": {"iterations": 2, "evidence_count": 5, "citation_count": 3}
        },
        "events": [{"type": "llm.completed", "payload": {"x": "y" * 200}}] * 50,
    }

    console.print(ui.inspect_summary(payload))
    text = console.export_text()

    assert "run-abc-123" in text
    assert "iterations" in text


def test_a_long_publisher_name_leaves_titles_on_one_line() -> None:
    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    result = ResearchRunResult(
        run_id="run-1",
        request=ResearchRequest(query="q"),
        report_markdown="# R",
        citations=[
            CitationEntry(
                citation_id=1,
                evidence_ids=["e1"],
                url="https://scholarlypublications.universiteitleiden.nl/download/1",
                title="Challenges in studying microplastics in human brain",
                publisher="scholarlypublications.universiteitleiden.nl",
                accessed_at="2026-10-28",
            )
        ],
    )

    console.print(ui.result_summary(result))

    assert "Challenges in studying microplastics in human brain" in console.export_text()
