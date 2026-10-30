from __future__ import annotations

import io
import logging
import sys

from rich.console import Console

from shandu.contracts import ResearchRequest, RunEvent
from shandu.ui import intro
from shandu.ui.rich_frontend import ShanduUI


def test_recap_renders_run_detail_and_events() -> None:
    console = Console(record=True, width=160)
    ui = ShanduUI(console=console)

    request = ResearchRequest(query="Distributed retrieval", max_iterations=2, parallelism=2)
    snapshot = ui.new_snapshot(request, model="deepseek/deepseek-v4-flash")
    snapshot.apply(RunEvent(stage="plan", message="Plan ready", iteration=0, metrics={"tasks": 2}))
    snapshot.apply(RunEvent(stage="search", message="Search complete", iteration=0, metrics={"evidence": 4}))

    console.print(ui.recap(snapshot))
    output = console.export_text()

    assert "deepseek/deepseek-v4-flash" in output
    assert "Distributed retrieval" in output
    assert "Search complete" in output
    assert "evidence" in output


def test_event_line_renders_stage_and_subagent() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    event = RunEvent(
        stage="search",
        message="Task iter_1_task_1 started",
        iteration=0,
        metrics={"task_index": 1, "task_total": 4},
        payload={"task_id": "iter_1_task_1"},
    )
    line = ui.event_line(event, snapshot)
    assert "search" in line.plain
    assert "Subagent 1 started" in line.plain
    assert "iter_1_task_1" not in line.plain


def test_event_line_renders_query_and_url_without_the_raw_trace_name() -> None:
    ui = ShanduUI(console=Console(record=True, width=160))
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    event = RunEvent(
        stage="search",
        message="Task iter_1_task_1 query completed",
        iteration=0,
        metrics={"trace_type": "query_completed", "hits": 3},
        payload={
            "task_id": "iter_1_task_1",
            "query": "future of multimodal agents",
            "url": "https://example.com/article",
            "urls": ["https://example.com/article"],
        },
    )
    line = ui.event_line(event, snapshot)
    assert "query_completed" not in line.plain
    assert "future of multimodal agents" in line.plain
    assert "https://example.com/article" in line.plain


def test_header_wraps_settings_between_labels_and_never_inside_one() -> None:
    console = Console(record=True, width=48)
    ui = ShanduUI(console=console)
    details = ["deepseek/deepseek-flash", "2 iterations", "pipeline researcher", "length fits the evidence"]

    console.print(ui.header("Why is the sky blue?", details))
    lines = console.export_text().splitlines()

    assert any("S H A N D U" in line for line in lines)
    assert any("Why is the sky blue?" in line for line in lines)
    for detail in details:
        assert any(detail in line for line in lines), detail


def test_result_summary_shows_sub_question_coverage() -> None:
    from shandu.contracts import ResearchRunResult

    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    result = ResearchRunResult(
        run_id="r1",
        request=ResearchRequest(query="q"),
        report_markdown="# R",
        run_stats={
            "sub_question_coverage": {"strong": 3, "moderate": 1, "weak": 0, "none": 2},
            "findings": 41,
        },
    )

    console.print(ui.result_summary(result))
    text = console.export_text()

    assert "3 strong · 1 moderate · 2 none" in text
    assert "41" in text


def test_result_summary_shows_citation_audit_counts() -> None:
    from shandu.contracts import ResearchRunResult

    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    result = ResearchRunResult(
        run_id="r1",
        request=ResearchRequest(query="q"),
        report_markdown="# R",
        run_stats={
            "citation_audit": {"supported": 36, "partial": 3, "unsupported": 2, "unchecked": 4}
        },
    )

    console.print(ui.result_summary(result))

    assert "36 supported · 3 partial · 2 removed · 4 unchecked" in console.export_text()


def test_progress_card_marks_each_subagent_as_it_finishes_or_fails() -> None:
    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    snapshot = ui.new_snapshot(ResearchRequest(query="q", max_iterations=2), model="m")
    for event in (
        RunEvent(stage="plan", message="Research brief ready", metrics={"sub_questions": 4}),
        RunEvent(stage="plan", message="Iteration 1 plan ready", iteration=0, metrics={"tasks": 3}),
        RunEvent(stage="search", message="Task a started", iteration=0, payload={"task_id": "a", "focus": "Trial results"}),
        RunEvent(
            stage="search",
            message="Task a query completed",
            iteration=0,
            metrics={"trace_type": "query_completed", "hits": 8},
            payload={"task_id": "a"},
        ),
        RunEvent(stage="search", message="Task b started", iteration=0, payload={"task_id": "b", "focus": "Adherence"}),
        RunEvent(stage="error", message="Task b failed", iteration=0, payload={"task_id": "b", "error": "timed out"}),
        RunEvent(stage="search", message="Task c started", iteration=0, payload={"task_id": "c", "focus": "Side effects"}),
        # One failed search inside a task that keeps working.
        RunEvent(
            stage="error",
            message="Search failed",
            iteration=0,
            metrics={"trace_type": "search_failed"},
            payload={"task_id": "c", "reason": "rate limited"},
        ),
        RunEvent(
            stage="search",
            message="Task a completed",
            iteration=0,
            metrics={"tasks_completed": 1, "evidence": 5},
            payload={"task_id": "a"},
        ),
    ):
        snapshot.apply(event)

    console.print(ui.progress(snapshot))
    rows = {line.split("Subagent ")[1][0]: line for line in console.export_text().splitlines() if "Subagent " in line}

    assert "✓" in rows["1"] and "Trial results" in rows["1"] and "1 query · 0 pages" in rows["1"]
    assert "✗" in rows["2"]
    assert "✓" not in rows["3"] and "✗" not in rows["3"]


def test_progress_card_follows_a_report_rewrite_from_its_citations() -> None:
    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m", phase="cite")
    snapshot.apply(RunEvent(stage="cite", message="Citation subagent completed", metrics={"citations": 12}))

    console.print(ui.progress(snapshot))
    text = console.export_text()

    assert "REWRITING THE REPORT" in text
    assert "✓ Cite" in text and "Write" in text and "Audit" in text
    assert "Plan" not in text and "Brief" not in text
    assert "sources 12" in text


def test_progress_title_names_the_next_iteration_while_it_is_planned() -> None:
    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    snapshot = ui.new_snapshot(ResearchRequest(query="q", max_iterations=2), model="m")
    snapshot.apply(RunEvent(stage="plan", message="Iteration 1 plan ready", iteration=0, metrics={"tasks": 2}))
    snapshot.apply(RunEvent(stage="synthesize", message="Iteration 1 synthesized", iteration=0))

    console.print(ui.progress(snapshot))

    assert "ITERATION 2 OF 2" in console.export_text()


def test_progress_card_lists_a_later_iteration_whose_tasks_reuse_earlier_ids() -> None:
    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    snapshot = ui.new_snapshot(ResearchRequest(query="q", max_iterations=2), model="m")
    for iteration, focus in ((0, "Trial results"), (1, "Long-term follow-up")):
        snapshot.apply(RunEvent(stage="plan", message="plan ready", iteration=iteration, metrics={"tasks": 1}))
        snapshot.apply(
            RunEvent(stage="search", message="Task t1 started", iteration=iteration, payload={"task_id": "t1", "focus": focus})
        )

    console.print(ui.progress(snapshot))
    text = console.export_text()

    assert "Long-term follow-up" in text
    assert "Trial results" not in text


def test_the_mark_is_text_where_the_flag_cannot_be_drawn() -> None:
    class Legacy(io.StringIO):
        encoding = "cp1252"

    drawn = Console(record=True, width=80, force_terminal=True, color_system="truecolor")
    for console in (
        # A pipe or a log: no color to draw with.
        Console(record=True, width=80),
        Console(record=True, width=80, force_terminal=True, color_system="truecolor", file=Legacy()),
    ):
        ShanduUI(console=console)
        console.print(ShanduUI(console=console).header("q", ["m"]))
        text = console.export_text()
        assert "S H A N D U" in text
        assert "▀" not in text

    ui = ShanduUI(console=drawn)
    drawn.print(ui.header("q", ["m"]))
    header = [line.rstrip() for line in drawn.export_text().splitlines()]
    mark = [line.rstrip() for line in intro.still().plain.splitlines()]
    assert header[: len(mark)] == mark


def test_a_report_keeps_every_table_value_and_starts_its_title_at_the_left_edge() -> None:
    console = Console(record=True, width=60)
    ui = ShanduUI(console=console)
    report = (
        "# Time-restricted eating versus calorie restriction\n\n"
        "| Trial | Participants | Duration | TRE arm | Comparator | Deficit |\n"
        "|---|---|---|---|---|---|\n"
        "| Liu 2022 | 139 adults | 12 months | 08:00–16:00 | Self-selected | restriction |\n"
    )

    console.print(ui.document("Final Report", report))
    lines = console.export_text().splitlines()

    assert not any("…" in line for line in lines)
    assert next(line for line in lines if "Time-restricted" in line).startswith("  Time-restricted")


def test_aisearch_sources_keep_each_url_on_one_line_and_link_it_in_full() -> None:
    from shandu.contracts import AISearchResult, AISearchSource

    url = "https://www.example.org/" + "section/" * 12 + "page.html"
    console = Console(record=True, width=60, force_terminal=True, color_system="truecolor")
    ui = ShanduUI(console=console)

    console.print(ui.sources(AISearchResult(query="q", answer_markdown="", sources=[AISearchSource(title="A page", url=url, snippet="")])))
    styled = console.export_text(styles=True, clear=False)
    lines = console.export_text().splitlines()

    assert [line.rstrip()[-1] for line in lines if "example.org" in line] == ["…"]
    assert url in styled


def test_the_progress_card_moves_logging_above_itself() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    ui = ShanduUI(console=console)
    handler = logging.StreamHandler(sys.__stderr__)
    logger = logging.getLogger("shandu.tests.card")
    logger.addHandler(handler)
    try:
        with ui.live_progress(ui.new_snapshot(ResearchRequest(query="q"), "m"), console):
            assert handler.stream is sys.stderr
            assert handler.stream is not sys.__stderr__
        assert handler.stream is sys.__stderr__
    finally:
        logger.removeHandler(handler)
