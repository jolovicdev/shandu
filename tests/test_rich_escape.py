from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console
from rich.text import Text

from shandu.cli import _run_query_line
from shandu.contracts import (
    AISearchResult,
    AISearchSource,
    CitationEntry,
    ResearchRequest,
    ResearchRunResult,
    RunEvent,
)
from shandu.ui.rich_frontend import ShanduUI

POISON = "Explain [/notopen] and [bold]this[/]"


def _ui(width: int = 160) -> tuple[ShanduUI, Console]:
    console = Console(record=True, width=width)
    return ShanduUI(console=console), console


def test_event_line_escapes_markup_in_external_fields() -> None:
    ui, _ = _ui()
    event = RunEvent(
        stage="search",
        message=POISON,
        iteration=0,
        metrics={"trace_type": POISON, "query": POISON, "hits": 3},
        payload={"task_id": POISON, "query": POISON, "url": POISON},
    )

    line = ui.event_line(event)

    assert isinstance(line, Text)
    assert "[/notopen]" in line.plain


def test_dashboard_renders_markup_in_events() -> None:
    ui, console = _ui()
    snapshot = ui.new_snapshot(ResearchRequest(query=POISON), model="m")
    snapshot.apply(
        RunEvent(
            stage="search",
            message=POISON,
            iteration=0,
            metrics={"trace_type": POISON, "query": POISON},
            payload={"task_id": POISON, "query": POISON, "url": POISON},
        )
    )

    console.print(ui.dashboard(snapshot))

    assert "[/notopen]" in console.export_text()


def test_result_and_source_panels_escape_web_text() -> None:
    ui, console = _ui()
    result = ResearchRunResult(
        run_id="run-1",
        request=ResearchRequest(query="q"),
        report_markdown="# R",
        citations=[
            CitationEntry(
                citation_id=1,
                evidence_ids=["e1"],
                url="https://example.com",
                title=POISON,
                publisher=POISON,
                accessed_at=datetime.now(timezone.utc).date().isoformat(),
            )
        ],
        evidence=[],
        iteration_summaries=[],
        run_stats={"iterations": 1},
    )
    ai_result = AISearchResult(
        query="q",
        answer_markdown="# A",
        sources=[
            AISearchSource(title=POISON, url=POISON, snippet="s", text_excerpt="t")
        ],
    )

    console.print(ui.result_panels(result))
    console.print(ui.ai_sources_panel(ai_result))
    console.print(ui.inspect_panel({"run_id": POISON, "status": POISON}))
    console.print(ui.warning(POISON))

    assert "[/notopen]" in console.export_text()


def test_cli_query_echo_escapes_markup() -> None:
    _, console = _ui()

    console.print(_run_query_line(POISON))

    assert "[/notopen]" in console.export_text()
