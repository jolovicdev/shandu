from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from shandu.agents.researcher import (
    _BUDGET_SPENT,
    _READS_SPENT,
    _SEARCHES_SPENT,
    ResearcherAgent,
    ResearcherByRequest,
)
from shandu.agents.search_subagent import _ExtractionPayload
from shandu.contracts import (
    EvidenceRecord,
    FinalReportDraft,
    IterationPlan,
    IterationSynthesis,
    ResearchBrief,
    ResearchRequest,
    SubagentTask,
)
from shandu.orchestration.lead_orchestrator import LeadOrchestrator
from shandu.services.report import ReportService
from shandu.services.run_store import RunStore
from shandu.services.scrape import ScrapedPage
from shandu.services.scrape.models import PageLink
from shandu.services.search import SearchHit

_BLOG = "https://blog.example.com/post"
_PAPER = "https://arxiv.org/abs/2501.00001"


class _Search:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del max_results
        self.queries.append(query)
        return [SearchHit(query=query, url=_BLOG, title="Blog post", snippet="summary")]


class _Scrape:
    def __init__(self) -> None:
        self.requested: list[str] = []

    async def scrape_many(self, urls):
        self.requested.extend(urls)
        return [
            ScrapedPage(
                requested_url=url,
                url=url,
                title=url,
                text="page text",
                domain="example.com",
                links=[PageLink(url=_PAPER, text="the original paper")] if url == _BLOG else [],
            )
            for url in urls
        ], 0


class _ModelDesk:
    """Plays the model: calls the researcher's tools, then returns a memo."""

    def __init__(self, script) -> None:
        self.script = script
        self.replies: list[str] = []

    async def arun(self, worker, job):
        if worker.name == "SubagentExtractor":
            return SimpleNamespace(
                status="completed",
                data=_ExtractionPayload(
                    snippet="s",
                    extracted_text="evidence text",
                    confidence=0.5,
                    source_class="unknown",
                    authorship="anonymous",
                    is_promotional=False,
                    summarizes_inaccessible_source=False,
                    claims=[],
                ),
                metrics={},
            )
        tools = {item.name: item.callable for item in worker.tools()}
        await self.script(tools, self.replies)
        return SimpleNamespace(
            status="completed",
            content="KIVI saves 2.6x memory [arxiv]",
            metrics={"usage": {"total_tokens": 10}, "cost_usd": 0.001},
        )


def _researcher(desk, search=None, scrape=None) -> ResearcherAgent:
    runtime = SimpleNamespace(settings=SimpleNamespace(model="m"), desk=desk)
    return ResearcherAgent(runtime, search or _Search(), scrape or _Scrape())


def _task() -> SubagentTask:
    return SubagentTask(task_id="t1", focus="KIVI memory", search_queries=["kivi"])


def test_researcher_follows_link_to_primary_source_and_reports_memo() -> None:
    async def script(tools, replies):
        await tools["web_search"](query="kivi kv cache")
        blog = json.loads(await tools["read_page"](url=_BLOG))
        replies.append(await tools["read_page"](url=blog["links"][0]["url"]))

    traces: list[tuple[str, dict]] = []

    async def on_trace(kind, payload) -> None:
        traces.append((kind, payload))

    evidence = asyncio.run(
        _researcher(_ModelDesk(script)).execute_task(
            "run:r", _task(), ResearchRequest(query="q"), progress_callback=on_trace
        )
    )

    assert [record.requested_url for record in evidence] == [_BLOG, _PAPER]
    assert evidence[0].search_query == "kivi kv cache"
    memo = next(payload for kind, payload in traces if kind == "task_memo")
    assert memo["memo"] == "KIVI saves 2.6x memory [arxiv]"
    assert (memo["searches"], memo["reads"]) == (1, 2)
    assert memo["llm_usage"]["cost_usd"] == 0.001


class _LoopFailsDesk(_ModelDesk):
    """Plays a tool loop that the desk stops before the model writes a memo."""

    async def arun(self, worker, job):
        if worker.name == "SubagentExtractor":
            return await super().arun(worker, job)
        tools = {item.name: item.callable for item in worker.tools()}
        await self.script(tools, self.replies)
        return SimpleNamespace(
            status="failed", errors=["Max iterations exceeded"], content=None, metrics={}
        )


def test_researcher_stopped_without_a_memo_reports_why_and_keeps_its_evidence() -> None:
    async def script(tools, replies):
        replies.append(await tools["read_page"](url=_BLOG))

    traces: list[tuple[str, dict]] = []

    async def on_trace(kind, payload) -> None:
        traces.append((kind, payload))

    evidence = asyncio.run(
        _researcher(_LoopFailsDesk(script)).execute_task(
            "run:r", _task(), ResearchRequest(query="q"), progress_callback=on_trace
        )
    )

    assert [record.requested_url for record in evidence] == [_BLOG]
    failed = next(payload for kind, payload in traces if kind == "researcher_failed")
    assert failed == {"task_id": "t1", "reason": "Max iterations exceeded"}


def test_researcher_reads_within_budget_and_repeats_are_free() -> None:
    async def script(tools, replies):
        replies.append(await tools["read_page"](url=_BLOG))
        replies.append(await tools["read_page"](url=_BLOG))
        replies.append(await tools["read_page"](url=_PAPER))
        replies.append(await tools["read_page"](url="https://seen.example.org/a"))

    desk = _ModelDesk(script)
    scrape = _Scrape()
    evidence = asyncio.run(
        _researcher(desk, scrape=scrape).execute_task(
            "run:r",
            _task(),
            ResearchRequest(query="q", max_pages_per_task=1),
            extracted_urls={"https://seen.example.org/a"},
        )
    )

    assert scrape.requested == [_BLOG]
    assert len(evidence) == 1
    assert desk.replies[1] == desk.replies[0]
    assert desk.replies[2] == _READS_SPENT.format(searches=6)
    assert desk.replies[3].startswith("Already read")


def test_researcher_out_of_searches_is_told_to_use_its_reads() -> None:
    async def script(tools, replies):
        await tools["web_search"](query="first")
        replies.append(await tools["web_search"](query="second"))
        await tools["read_page"](url=_BLOG)
        replies.append(await tools["web_search"](query="third"))

    desk = _ModelDesk(script)
    search = _Search()
    asyncio.run(
        _researcher(desk, search=search).execute_task(
            "run:r",
            _task(),
            ResearchRequest(query="q", max_searches_per_task=1, max_pages_per_task=1),
        )
    )

    assert search.queries == ["first"]
    assert desk.replies == [_SEARCHES_SPENT.format(reads=1), _BUDGET_SPENT]


def test_researcher_by_request_routes_on_request_field() -> None:
    calls: list[str] = []

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

        async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None, brief=None):
            calls.append(self.name)
            return []

    router = ResearcherByRequest(pipeline=_Named("pipeline"), agentic=_Named("agentic"))
    asyncio.run(router.execute_task("run:r", _task(), ResearchRequest(query="q")))
    asyncio.run(
        router.execute_task("run:r", _task(), ResearchRequest(query="q", researcher="agentic"))
    )

    assert calls == ["pipeline", "agentic"]


class _MemoLead:
    fallback_count = 0

    def __init__(self) -> None:
        self.task_memos = None

    async def create_brief(self, request):
        return ResearchBrief(scope="s", sub_questions=["q"])

    async def create_iteration_plan(self, request, iteration, prior_summaries, memory_context, brief=None):
        return IterationPlan(
            iteration_index=iteration,
            subagent_tasks=[_task()],
            continue_loop=False,
        )

    async def synthesize_iteration(self, request, iteration, iteration_evidence, prior_summaries, brief=None, task_memos=None):
        self.task_memos = task_memos
        return IterationSynthesis(summary="s")

    async def build_final_report(self, request, iteration_summaries, evidence_payload, citations_payload, brief=None):
        return FinalReportDraft(title="T", executive_summary="E", markdown="# T")


class _MemoSubagent:
    async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None, brief=None):
        await progress_callback("task_memo", {"task_id": task.task_id, "focus": task.focus, "memo": "found it"})
        return [
            EvidenceRecord(
                evidence_id="e1",
                task_id=task.task_id,
                query=task.focus,
                requested_url=_PAPER,
                title="Paper",
                snippet="s",
                extracted_text="x",
            )
        ]


class _NoCitations:
    async def build_citations(self, query, evidence):
        return []


class _MemolessSubagent:
    async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None, brief=None):
        await progress_callback(
            "researcher_failed", {"task_id": task.task_id, "reason": "Max iterations exceeded"}
        )
        return []


def test_orchestrator_counts_a_researcher_stopped_without_a_memo(tmp_path) -> None:
    store = RunStore(tmp_path)
    orchestrator = LeadOrchestrator(
        lead_agent=_MemoLead(),
        search_subagent=_MemolessSubagent(),
        citation_agent=_NoCitations(),
        run_store=store,
        report_service=ReportService(),
    )

    result = asyncio.run(
        orchestrator.run(ResearchRequest(query="q", max_iterations=1, parallelism=1))
    )

    assert result.run_stats["agent_fallbacks"] == 1
    assert result.run_stats["fallback_reasons"] == ["Max iterations exceeded"]
    errors = [event for event in store.events(result.run_id) if event["stage"] == "error"]
    assert [event["message"] for event in errors] == ["Task t1 researcher stopped without a memo"]


def test_orchestrator_hands_task_memos_to_synthesizer(tmp_path) -> None:
    lead = _MemoLead()
    orchestrator = LeadOrchestrator(
        lead_agent=lead,
        search_subagent=_MemoSubagent(),
        citation_agent=_NoCitations(),
        run_store=RunStore(tmp_path),
        report_service=ReportService(),
    )

    asyncio.run(orchestrator.run(ResearchRequest(query="q", max_iterations=1, parallelism=1)))

    assert lead.task_memos == [{"task_id": "t1", "focus": "KIVI memory", "memo": "found it"}]


def test_parallel_researchers_do_not_read_one_page_twice() -> None:
    async def script(tools, replies):
        replies.append(await tools["read_page"](url=_PAPER))

    shared: set[str] = set()
    scrape = _Scrape()
    first, second = _ModelDesk(script), _ModelDesk(script)

    asyncio.run(
        _researcher(first, scrape=scrape).execute_task(
            "run:r", _task(), ResearchRequest(query="q"), extracted_urls=shared
        )
    )
    asyncio.run(
        _researcher(second, scrape=scrape).execute_task(
            "run:r", _task(), ResearchRequest(query="q"), extracted_urls=shared
        )
    )

    assert scrape.requested == [_PAPER]
    assert second.replies[0].startswith("Already read in this run")


def test_researcher_paper_search_ignores_domain_scoping() -> None:
    papers = _Search()

    async def script(tools, replies):
        replies.append(await tools["search_papers"](query="kivi quantization"))

    runtime = SimpleNamespace(settings=SimpleNamespace(model="m"), desk=_ModelDesk(script))
    researcher = ResearcherAgent(runtime, _Search(), _Scrape(), papers)
    asyncio.run(
        researcher.execute_task(
            "run:r",
            _task(),
            ResearchRequest(query="q", include_domains=["arxiv.org", "nih.gov"]),
        )
    )

    assert papers.queries == ["kivi quantization"]


def test_researcher_reads_only_allowed_domains() -> None:
    async def script(tools, replies):
        replies.append(await tools["read_page"](url=_BLOG))
        replies.append(await tools["read_page"](url=_PAPER))

    desk = _ModelDesk(script)
    scrape = _Scrape()
    asyncio.run(
        _researcher(desk, scrape=scrape).execute_task(
            "run:r", _task(), ResearchRequest(query="q", include_domains=["arxiv.org"])
        )
    )

    assert scrape.requested == [_PAPER]
    assert desk.replies[0].startswith("Outside the allowed domains")
