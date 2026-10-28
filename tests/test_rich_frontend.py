from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
from rich.console import Console

from shandu.contracts import CitationEntry, ResearchRequest, ResearchRunResult, RunEvent
from shandu.services.run_store import RunSummary
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
            [
                "USD spent",
                "at least $0.012345 · 6 of 9 model calls metered",
                "model calls",
                "LLM tokens",
            ],
            ["cost coverage", "metered cost"],
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


@pytest.fixture
def utc_plus_5_30(monkeypatch):
    monkeypatch.setenv("TZ", "IST-5:30")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_inspect_summary_shows_local_times_under_lowercase_labels(utc_plus_5_30) -> None:
    console = Console(record=True, width=120)
    ui = ShanduUI(console=console)
    payload = {
        "run_id": "r1",
        "status": "completed",
        "created_at": "2026-10-28T17:34:48.761300+00:00",
        "updated_at": "2026-10-28T18:40:01+00:00",
        "output_json": {"run_stats": {"citation_count": 3}},
    }

    console.print(ui.inspect_summary(payload))
    lines = console.export_text().splitlines()

    assert any(line.split() == ["started", "2026-10-28", "23:04"] for line in lines)
    assert any(line.split() == ["updated", "2026-10-29", "00:10"] for line in lines)
    assert any(line.split() == ["sources", "3"] for line in lines)
    assert "+00:00" not in "".join(lines)


def _summary_text(**run_stats) -> str:
    console = Console(record=True, width=120)
    result = ResearchRunResult(
        run_id="0123456789abcdef0123456789abcdef",
        request=ResearchRequest(query="q"),
        report_markdown="# R",
        run_stats=run_stats,
    )
    console.print(ShanduUI(console=console).result_summary(result))
    return console.export_text()


def test_result_summary_shows_the_short_run_id_and_calls_references_sources() -> None:
    text = _summary_text(citation_count=32)
    lines = [line.split() for line in text.splitlines()]

    assert ["run", "01234567"] in lines
    assert ["sources", "32"] in lines
    assert "0123456789abcdef" not in text
    assert "No sources" in text


def test_result_summary_words_the_citation_checks() -> None:
    text = _summary_text(citation_audit={"supported": 144, "partial": 23, "unsupported": 1})

    assert "citation checks 144 supported · 23 partial · 1 removed" in " ".join(text.split())


def test_result_summary_leaves_cost_coverage_out_when_every_call_is_metered() -> None:
    text = _summary_text(
        agent_model_calls=59,
        metered_calls=69,
        cost_coverage="full",
        usd_spent=0.05,
    )

    assert "USD spent $0.050000" in " ".join(text.split())
    assert "coverage" not in text
    assert "69" not in text


def test_result_summary_marks_a_partly_metered_cost_as_a_lower_bound() -> None:
    text = _summary_text(
        agent_model_calls=16,
        metered_calls=12,
        cost_coverage="partial",
        usd_spent=0.05,
    )

    assert "USD spent at least $0.050000 · 12 of 16 model calls metered" in " ".join(text.split())


def _list_text(runs: list[RunSummary], width: int = 100) -> str:
    console = Console(record=True, width=width)
    console.print(ShanduUI(console=console).run_list(runs))
    return console.export_text()


def _stored(run_id: str, query: str, status: str = "completed", created: str = "") -> RunSummary:
    return RunSummary(
        run_id=run_id,
        query=query,
        status=status,
        created_at=created or "2026-10-28T17:34:48.761300+00:00",
        citation_count=None,
        usd_spent=None,
        report_path=None,
    )


def test_run_list_has_headers_and_local_start_times(utc_plus_5_30) -> None:
    text = _list_text([_stored("473677ec1d73484f919f35932bbdeddd", "Why is the sky blue?")])
    header, row = [line.split() for line in text.splitlines()[1:3]]

    assert header == ["run", "started", "status", "cost", "question"]
    assert row == ["473677ec", "2026-10-28", "23:04", "completed", "Why", "is", "the", "sky", "blue?"]


def test_run_list_wraps_a_long_question_instead_of_cutting_it() -> None:
    query = "Beyond weight loss, what do the SELECT, FLOW and SOUL trials show about kidney outcomes?"

    text = _list_text([_stored("473677ec1d73484f919f35932bbdeddd", query)], width=60)

    assert "…" not in text
    assert " ".join(text.split()).endswith(query)


def test_run_list_styles_each_status() -> None:
    console = Console(record=True, width=100, force_terminal=True, color_system="truecolor")
    ui = ShanduUI(console=console)
    runs = [
        _stored("a" * 32, "q", status=status)
        for status in ("completed", "failed", "cancelled", "running")
    ]

    styles = {
        segment.text.strip(): segment.style
        for segment in console.render(ui.run_list(runs))
        if segment.text.strip() in {run.status for run in runs} and segment.style
    }

    assert styles["completed"].color == console.get_style("ok").color
    assert styles["failed"].color == console.get_style("danger").color
    assert styles["cancelled"].color == console.get_style("warn").color
    assert styles["running"].color == console.get_style("label").color


def _event(message: str, task: str = "", **fields) -> RunEvent:
    payload = {"task_id": task, **fields.pop("payload", {})} if task else fields.pop("payload", {})
    return RunEvent(stage="search", message=message, payload=payload, **fields)


def test_subagents_are_numbered_by_arrival_across_iterations() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    events = [
        _event("Task iter_1_alpha started", "iter_1_alpha"),
        _event("Task iter_1_beta started", "iter_1_beta"),
        _event("Task iter_1_alpha completed", "iter_1_alpha"),
        _event("Task iter_2_gamma started", "iter_2_gamma"),
    ]
    lines = []
    for event in events:
        snapshot.apply(event)
        lines.append(ui.event_line(event, snapshot).plain)

    assert [line.split(maxsplit=2)[2] for line in lines] == [
        "Subagent 1 started",
        "Subagent 2 started",
        "Subagent 1 completed",
        "Subagent 3 started",
    ]
    assert not any("iter_" in line for line in lines)


def test_event_line_words_metrics_as_the_gui_feed_does() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    event = _event(
        "Iteration 1 subagents completed",
        metrics={
            "tasks": 3,
            "evidence": 12,
            "scraped": 5,
            "url_count": 6,
            "hits": 8,
            "citations": 4,
            "agent_model_calls": 9,
            "llm_usage": {"total_tokens": 10},
            "parallelism": 3,
        },
    )

    text = ui.event_line(event, snapshot).plain

    assert text.endswith(
        "subagents 3 · results 8 · pages to read 6 · pages read 5 · evidence 12 · sources 4"
    )
    assert "agent_model_calls" not in text
    assert "llm_usage" not in text


def test_event_line_shows_raw_metrics_and_trace_type_when_verbose() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    event = _event(
        "Task t1 searching query",
        "t1",
        iteration=1,
        metrics={"trace_type": "query_started", "query": "tides", "agent_model_calls": 2},
        payload={"query": "tides"},
    )

    text = ui.event_line(event, snapshot, verbose=True).plain

    assert text.endswith('Subagent 1 searching query  iteration 2 · query_started · agent_model_calls=2 · "tides"')


def _printed(events: list[RunEvent], verbose: bool = False, **console_options) -> str:
    console = Console(record=True, width=60, **console_options)
    ui = ShanduUI(console=console)
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    for event in events:
        snapshot.apply(event)
        ui.print_event(event, snapshot, console, verbose)
    return console.export_text()


def test_print_event_skips_announcements_that_a_completion_line_repeats() -> None:
    events = [
        _event("Task t1 searching query", "t1", metrics={"trace_type": "query_started"}),
        _event("Task t1 query completed", "t1", metrics={"trace_type": "query_completed", "hits": 6}),
        _event("Task t1 scraping pages", "t1", metrics={"trace_type": "scrape_started", "url_count": 3}),
        _event("Task t1 extracting page", "t1", metrics={"trace_type": "extract_started"}),
        _event("Task t1 extracted page", "t1", metrics={"trace_type": "extract_completed"}),
    ]

    quiet = _printed(events)
    verbose = _printed(events, verbose=True)

    assert [line.split(maxsplit=2)[2].split("  ")[0] for line in quiet.splitlines()] == [
        "Subagent 1 query completed",
        "Subagent 1 extracted page",
    ]
    assert len(verbose.splitlines()) == 5


def test_print_event_keeps_the_whole_line_when_the_output_is_not_a_terminal() -> None:
    url = "https://example.com/" + "a-long-path/" * 8
    event = _event("Task t1 extracted page", "t1", payload={"url": url, "query": "q" * 40})

    text = _printed([event], force_terminal=False)

    assert len(text.splitlines()) == 1
    assert url in text and "q" * 40 in text


def test_print_event_crops_to_one_line_on_a_terminal() -> None:
    url = "https://example.com/" + "a-long-path/" * 8
    event = _event("Task t1 extracted page", "t1", payload={"url": url})

    text = _printed([event], force_terminal=True)

    assert len(text.splitlines()) == 1
    assert len(text.splitlines()[0]) == 60 and text.splitlines()[0].endswith("…")


def test_snapshot_takes_the_run_id_from_the_first_event_and_the_last() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")

    assert snapshot.run_id == "pending"
    snapshot.apply(RunEvent(stage="bootstrap", message="Initializing run", metrics={"run_id": "abc123"}))
    assert snapshot.run_id == "abc123"
    snapshot.apply(RunEvent(stage="complete", message="Run completed", payload={"run_id": "abc124"}))
    assert snapshot.run_id == "abc124"


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


def test_event_line_says_why_a_subagent_failed() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    event = _event(
        "Task t1 failed",
        "t1",
        payload={"task_id": "t1", "error": "researcher loop timed out"},
    )

    text = ui.event_line(event, snapshot).plain

    assert text.endswith("Subagent 1 failed  researcher loop timed out")
