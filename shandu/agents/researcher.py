from __future__ import annotations

import json
import logging
from collections.abc import Collection, Sequence
from datetime import date
from typing import Any

from blackgeorge import Job, Worker
from blackgeorge.tools import Tool, tool
from pydantic import BaseModel

from ..contracts import EvidenceRecord, ResearchRequest, SubagentTask
from ..interfaces import SearchServiceLike, SearchSubagentLike
from ..prompts import researcher_instructions, researcher_job
from ..runtime.costing import collect_llm_usage
from .search_subagent import (
    _MAX_QUERIES_PER_TASK,
    SearchSubagent,
    SearchTraceCallback,
    _dated_before,
    _host_matches,
    _report_reason,
)

logger = logging.getLogger(__name__)

_BUDGET_SPENT = "Budget spent. Stop calling tools and write your memo."
# The researcher sees the extractor's evidence, not the raw page, which keeps
# a dozen reads inside the loop's context.
_EVIDENCE_CHARS_FOR_MODEL = 4000
_LINKS_FOR_MODEL = 15


class _FindingsMemo(BaseModel):
    memo: str


class ResearcherAgent(SearchSubagent):
    """Runs one task as a tool loop instead of a fixed search-then-read pass.

    The model searches, reads, follows links toward primary sources, and
    stops when the task is covered. Every page it reads goes through the same
    extraction and source scoring as the pipeline, so evidence records and
    trace events keep their shape. The step budget lives in the tools: the
    desk fails a job that exceeds its own tool-call limit, while a tool that
    reports the budget spent lets the model finish with a memo.
    """

    async def execute_task(
        self,
        run_scope: str,
        task: SubagentTask,
        request: ResearchRequest,
        progress_callback: SearchTraceCallback | None = None,
        extracted_urls: Collection[str] | None = None,
    ) -> list[EvidenceRecord]:
        del run_scope
        evidence: list[EvidenceRecord] = []
        hits_by_url: dict[str, dict[str, str]] = {}
        replies: dict[str, str] = {}
        known = {self._canonicalize_url(url) or url for url in extracted_urls or ()}
        budget = {"searches": _MAX_QUERIES_PER_TASK, "reads": request.max_pages_per_task}

        def allowed(url: str) -> bool:
            if request.include_domains and not _host_matches(url, request.include_domains):
                return False
            return not _host_matches(url, request.exclude_domains)

        async def run_search(service: SearchServiceLike, query: str, source: str) -> str:
            if budget["searches"] <= 0:
                return _BUDGET_SPENT
            budget["searches"] -= 1
            scoped = (
                [f"{query} site:{domain}" for domain in request.include_domains]
                if request.include_domains
                else [query]
            )
            found: list[dict[str, Any]] = []
            for search_query in scoped:
                await self._emit_trace(
                    progress_callback,
                    "query_started",
                    {
                        "task_id": task.task_id,
                        "focus": task.focus,
                        "query": search_query,
                        "source": source,
                        "max_results": request.max_results_per_query,
                    },
                )
                hits: Sequence[Any] = await service.search(
                    search_query, request.max_results_per_query
                )
                error_for = getattr(service, "last_error_for", None)
                search_error = (
                    error_for(search_query, request.max_results_per_query)
                    if error_for
                    else None
                )
                if search_error:
                    await self._emit_trace(
                        progress_callback,
                        "search_failed",
                        {"task_id": task.task_id, "query": search_query, "reason": search_error},
                    )
                kept = [
                    hit
                    for hit in hits
                    if allowed(hit.url)
                    and not (
                        request.since is not None
                        and _dated_before(getattr(hit, "published_at", None), request.since)
                    )
                ]
                await self._emit_trace(
                    progress_callback,
                    "query_completed",
                    {
                        "task_id": task.task_id,
                        "query": search_query,
                        "hits": len(kept),
                        "urls": [hit.url for hit in kept[:8]],
                    },
                )
                for hit in kept:
                    entry = {
                        "url": hit.url,
                        "title": hit.title,
                        "snippet": hit.snippet,
                        "query": search_query,
                    }
                    hits_by_url.setdefault(hit.url, entry)
                    found.append(
                        {
                            "url": hit.url,
                            "title": hit.title,
                            "snippet": hit.snippet[:300],
                            "published_at": getattr(hit, "published_at", None),
                        }
                    )
            if search_error and not found:
                return f"Search failed: {search_error}"
            return json.dumps(found, ensure_ascii=False)

        @tool()
        async def web_search(query: str) -> str:
            """Search the web. Returns hits with url, title, snippet, and date."""
            return await run_search(self._search, query, "web")

        @tool()
        async def read_page(url: str) -> str:
            """Read a page. Returns task evidence, source credibility, and outbound links."""
            canonical = self._canonicalize_url(url) or url
            if canonical in replies:
                return replies[canonical]
            if canonical in known:
                return "Already read earlier in this run; choose another source."
            if not allowed(url):
                return "Outside the allowed domains; choose another source."
            if budget["reads"] <= 0:
                return _BUDGET_SPENT
            budget["reads"] -= 1
            await self._emit_trace(
                progress_callback,
                "scrape_started",
                {"task_id": task.task_id, "url_count": 1, "urls": [url]},
            )
            pages, _ = await self._scrape.scrape_many([url])
            page = pages[0] if pages else None
            too_old = (
                page is not None
                and page.fetch_error is None
                and request.since is not None
                and _dated_before(page.published_at, request.since)
            )
            readable = page is not None and page.fetch_error is None and not too_old
            await self._emit_trace(
                progress_callback,
                "scrape_completed",
                {
                    "task_id": task.task_id,
                    "scraped": 1 if readable else 0,
                    "missed": 0 if page is not None and page.fetch_error is None else 1,
                    "dated_before_since": 1 if too_old else 0,
                    "urls": [page.url] if readable and page is not None else [],
                },
            )
            if page is None or page.fetch_error is not None:
                reply = f"Could not read the page ({page.fetch_error if page else 'scrape_failed'})."
            elif too_old:
                reply = "The page is dated before the research cutoff; choose another source."
            else:
                record = await self._page_evidence(
                    task, page, hits_by_url.get(url), progress_callback
                )
                evidence.append(record)
                reply = json.dumps(
                    {
                        "url": page.url,
                        "title": page.title,
                        "published_at": page.published_at,
                        "source_class": record.source_class,
                        "credibility": record.credibility_score,
                        "evidence": record.extracted_text[:_EVIDENCE_CHARS_FOR_MODEL],
                        "links": [
                            {"url": link.url, "text": link.text}
                            for link in getattr(page, "links", [])[:_LINKS_FOR_MODEL]
                        ],
                    },
                    ensure_ascii=False,
                )
            replies[canonical] = reply
            return reply

        tools: list[Tool] = [web_search, read_page]
        scholarly = self._scholarly
        if scholarly is not None:

            @tool()
            async def search_papers(query: str) -> str:
                """Search arXiv, OpenAlex, and PubMed for papers and preprints."""
                return await run_search(scholarly, query, "scholarly")

            tools.append(search_papers)
        worker = Worker(
            name="Researcher",
            model=self._runtime.settings.model,
            tools=tools,
            instructions=researcher_instructions(),
        )
        job = Job(
            input=researcher_job(
                {
                    "task_focus": task.focus,
                    "expected_output": task.expected_output,
                    "suggested_queries": task.search_queries,
                    "papers_likely": task.scholarly,
                    "budget": dict(budget),
                    "today": date.today().isoformat(),
                }
            ),
            response_schema=_FindingsMemo,
        )
        memo = ""
        llm_usage: dict[str, Any] | None = None
        try:
            report = await self._runtime.desk.arun(worker, job)
            llm_usage = collect_llm_usage(report)
            if report.status == "completed" and isinstance(report.data, _FindingsMemo):
                memo = report.data.memo.strip()
            else:
                logger.warning(
                    "Researcher did not finish %s: %s", task.task_id, _report_reason(report)
                )
        except Exception:
            logger.warning("Researcher failed on %s", task.task_id, exc_info=True)

        memo_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "focus": task.focus,
            "memo": memo,
            "searches": _MAX_QUERIES_PER_TASK - budget["searches"],
            "reads": request.max_pages_per_task - budget["reads"],
        }
        if llm_usage:
            memo_payload["llm_usage"] = llm_usage
        await self._emit_trace(progress_callback, "task_memo", memo_payload)
        return evidence


class ResearcherByRequest:
    """Routes each task to the researcher its request names."""

    def __init__(self, pipeline: SearchSubagentLike, agentic: SearchSubagentLike) -> None:
        self._pipeline = pipeline
        self._agentic = agentic

    async def execute_task(
        self,
        run_scope: str,
        task: SubagentTask,
        request: ResearchRequest,
        progress_callback: SearchTraceCallback | None = None,
        extracted_urls: Collection[str] | None = None,
    ) -> list[EvidenceRecord]:
        researcher = self._agentic if request.researcher == "agentic" else self._pipeline
        return await researcher.execute_task(
            run_scope, task, request, progress_callback, extracted_urls
        )
