from __future__ import annotations

from rich.console import Console

from shandu.contracts import ResearchRequest, RunEvent
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


def test_header_question_aligns_with_event_stage_column() -> None:
    console = Console(record=True, width=100)
    ui = ShanduUI(console=console)
    snapshot = ui.new_snapshot(ResearchRequest(query="q"), model="m")
    console.print(ui.header("Why is the sky blue?", ["deepseek/deepseek-flash"]))
    console.print(ui.event_line(RunEvent(stage="plan", message="Iteration 1 plan ready"), snapshot))

    lines = console.export_text().splitlines()
    question = next(line for line in lines if "Why is the sky blue?" in line)
    details = next(line for line in lines if "deepseek/deepseek-flash" in line)
    event = next(line for line in lines if "plan ready" in line)
    assert question.index("Why") == details.index("deepseek") == event.index("plan")


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
