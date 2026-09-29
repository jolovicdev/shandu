from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from shandu.contracts import (
    ResearchBrief,
    EvidenceRecord,
    FinalReportDraft,
    IterationPlan,
    IterationSynthesis,
    ResearchRequest,
    SubagentTask,
)
from shandu.engine import ShanduEngine
from shandu.orchestration.lead_orchestrator import LeadOrchestrator
from shandu.services.report import ReportService
from shandu.services.run_store import RunNotFoundError, RunStore


class _Lead:
    fallback_count = 0

    def __init__(self, fail_synthesis_at: int | None = None) -> None:
        self.fail_synthesis_at = fail_synthesis_at
        self.briefs_written = 0
        self.briefs_seen: list[ResearchBrief | None] = []
        self.planned: list[int] = []
        self.report_request: ResearchRequest | None = None
        self.report_summaries: list[str] = []
        self.report_evidence_ids: list[str] = []

    async def create_brief(self, request):
        self.briefs_written += 1
        return ResearchBrief(scope=request.query, sub_questions=["sq one", "sq two"])

    async def create_iteration_plan(
        self, request, iteration, prior_summaries, memory_context, brief=None
    ):
        del request, prior_summaries, memory_context
        self.briefs_seen.append(brief)
        self.planned.append(iteration)
        return IterationPlan(
            iteration_index=iteration,
            subagent_tasks=[
                SubagentTask(
                    task_id=f"t{iteration}",
                    focus=f"focus {iteration}",
                    search_queries=[f"query {iteration}"],
                )
            ],
        )

    async def synthesize_iteration(
        self, request, iteration, iteration_evidence, prior_summaries, brief=None, task_memos=None
    ):
        del request, iteration_evidence, prior_summaries
        self.briefs_seen.append(brief)
        if iteration == self.fail_synthesis_at:
            raise RuntimeError("synthesis boom")
        return IterationSynthesis(summary=f"summary {iteration}")

    async def build_final_report(
        self, request, iteration_summaries, evidence_payload, citations_payload, brief=None
    ):
        del citations_payload
        self.briefs_seen.append(brief)
        self.report_request = request
        self.report_summaries = [item.summary for item in iteration_summaries]
        self.report_evidence_ids = [item["evidence_id"] for item in evidence_payload]
        return FinalReportDraft(
            title="T", executive_summary="E", markdown="# T\n\nBody [1]."
        )


class _Subagent:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    async def execute_task(
        self, run_scope, task, request, progress_callback=None, extracted_urls=None
    ):
        del run_scope, request, progress_callback, extracted_urls
        return [
            EvidenceRecord(
                evidence_id=f"{self.prefix}-{task.task_id}",
                task_id=task.task_id,
                query=task.focus,
                requested_url=f"https://example.com/{self.prefix}/{task.task_id}",
                title="Title",
                snippet="s",
                extracted_text="x",
            )
        ]


class _NoSearch:
    async def execute_task(self, *args, **kwargs):
        raise AssertionError("report rewrite must not search")


class _Citations:
    async def build_citations(self, query, evidence):
        del query, evidence
        return []


def _orchestrator(store: RunStore, lead, subagent) -> LeadOrchestrator:
    return LeadOrchestrator(
        lead_agent=lead,
        search_subagent=subagent,
        citation_agent=_Citations(),
        run_store=store,
        report_service=ReportService(),
    )


def _crashed_run(store: RunStore) -> str:
    """A two-iteration run that dies in the second synthesis."""
    seen: dict[str, str] = {}

    def on_event(event) -> None:
        if event.stage == "bootstrap":
            seen["run_id"] = event.metrics["run_id"]

    request = ResearchRequest(
        query="q", max_iterations=2, parallelism=1, depth_policy="fixed"
    )
    orchestrator = _orchestrator(store, _Lead(fail_synthesis_at=1), _Subagent("old"))
    with pytest.raises(RuntimeError, match="synthesis boom"):
        asyncio.run(orchestrator.run(request, progress_callback=on_event))
    return seen["run_id"]


def test_resume_continues_after_last_completed_iteration(tmp_path) -> None:
    store = RunStore(tmp_path)
    run_id = _crashed_run(store)
    lead = _Lead()

    result = asyncio.run(_orchestrator(store, lead, _Subagent("new")).resume(run_id))

    assert lead.planned == [1]
    assert lead.report_summaries == ["summary 0", "summary 1"]
    assert lead.report_evidence_ids == ["old-t0", "new-t1"]
    assert result.run_id == run_id
    assert store.load(run_id).status["status"] == "completed"
    assert store.report_path(run_id).read_text(encoding="utf-8").startswith("# T")


def test_brief_reaches_every_lead_call_and_survives_resume(tmp_path) -> None:
    store = RunStore(tmp_path)
    run_id = _crashed_run(store)
    lead = _Lead()

    result = asyncio.run(_orchestrator(store, lead, _Subagent("new")).resume(run_id))

    assert lead.briefs_written == 0
    assert [brief.sub_questions for brief in lead.briefs_seen] == [["sq one", "sq two"]] * 3
    assert result.brief is not None
    assert result.brief.sub_questions == ["sq one", "sq two"]


def test_rewrite_report_uses_stored_evidence_without_searching(tmp_path) -> None:
    store = RunStore(tmp_path)
    run_id = _crashed_run(store)
    lead = _Lead()

    asyncio.run(
        _orchestrator(store, lead, _NoSearch()).rewrite_report(run_id, "concise")
    )

    assert lead.planned == []
    assert lead.report_evidence_ids == ["old-t0", "old-t1"]
    assert lead.report_request is not None
    assert lead.report_request.detail_level == "concise"
    assert store.report_path(run_id).is_file()


def test_list_runs_newest_first_and_rejects_path_ids(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.create("a", ResearchRequest(query="first"), None)
    store.create("b", ResearchRequest(query="second"), None)
    store.save_report("b", "# Second", [], {"citation_count": 3, "usd_spent": 0.5})

    runs = store.list_runs()

    assert [run.run_id for run in runs] == ["b", "a"]
    assert runs[0].citation_count == 3
    assert runs[0].report_path == tmp_path / "b" / "report.md"
    assert runs[1].report_path is None
    with pytest.raises(RunNotFoundError):
        store.report_path("../a")


def test_engine_inspect_reads_run_store(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.create("r1", ResearchRequest(query="stored question"), {"model": "m"})
    store.save_report("r1", "# R", [], {"iterations": 2})
    store.set_status("r1", "completed")
    engine = ShanduEngine(
        runtime=SimpleNamespace(inspect_run=lambda run_id: {"exists": False}),
        orchestrator=SimpleNamespace(),
        ai_search_service=SimpleNamespace(),
        run_store=store,
    )

    payload = engine.inspect_run("r1")

    assert payload["status"] == "completed"
    assert payload["input"]["query"] == "stored question"
    assert payload["output_json"] == {"run_stats": {"iterations": 2}}
    assert engine.inspect_run("missing") == {"exists": False}
