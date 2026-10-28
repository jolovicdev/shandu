from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from typing import Any

from blackgeorge import Job, Worker
from blackgeorge.tools import Tool, tool

from ..contracts import EvidenceRecord, ResearchBrief, ResearchRequest, SubagentTask
from ..interfaces import SearchServiceLike, SearchSubagentLike
from ..prompts import researcher_instructions, researcher_job
from ..runtime.calls import researcher_call
from ..runtime.costing import collect_llm_usage
from ..runtime.roles import role_model
from .search_subagent import (
    SearchSubagent,
    SearchTraceCallback,
    _dated_before,
    _report_reason,
)

logger = logging.getLogger(__name__)

_BUDGET_SPENT = "Budget spent. Stop calling tools and write your memo."
# Searches and reads are budgeted apart. A reply that only says "stop" when one
# runs out makes the model abandon the other, so it names what is left.
_SEARCHES_SPENT = (
    "Search budget spent. You can still read {reads} pages: read the best "
    "results you have, then write your memo."
)
_READS_SPENT = (
    "Read budget spent. You can still run {searches} searches, or write your memo."
)
# The researcher sees the extractor's evidence, not the raw page, which keeps
# a dozen reads inside the loop's context.
_EVIDENCE_CHARS_FOR_MODEL = 4000
_LINKS_FOR_MODEL = 15


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
        extracted_urls: set[str] | None = None,
        brief: ResearchBrief | None = None,
    ) -> list[EvidenceRecord]:
        del run_scope
        evidence: list[EvidenceRecord] = []
        claimed = extracted_urls if extracted_urls is not None else set()
        hits_by_url: dict[str, dict[str, str]] = {}
        replies: dict[str, str] = {}
        known = {self._canonicalize_url(url) or url for url in extracted_urls or ()}
        budget = {"searches": request.max_searches_per_task, "reads": request.max_pages_per_task}

        async def run_search(service: SearchServiceLike, query: str, source: str) -> str:
            if budget["searches"] <= 0:
                return _SEARCHES_SPENT.format(**budget) if budget["reads"] > 0 else _BUDGET_SPENT
            budget["searches"] -= 1
            # Paper search ignores site: scoping, so it runs the query once.
            scoped = self._scoped_queries([query], request) if source == "web" else [query]
            outcomes = await asyncio.gather(
                *(
                    self._traced_search(
                        service, search_query, source, task, request, progress_callback
                    )
                    for search_query in scoped
                )
            )
            found: list[dict[str, Any]] = []
            for search_query, (hits, _) in zip(scoped, outcomes):
                for hit in hits:
                    hits_by_url.setdefault(
                        hit.url,
                        {
                            "title": hit.title,
                            "snippet": hit.snippet,
                            "query": search_query,
                        },
                    )
                    found.append(
                        {
                            "url": hit.url,
                            "title": hit.title,
                            "snippet": hit.snippet[:300],
                            "published_at": getattr(hit, "published_at", None),
                        }
                    )
            errors = [error for _, error in outcomes if error]
            if errors and not found:
                return f"Search failed: {errors[-1]}"
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
            if canonical in known or canonical in claimed:
                return (
                    "Already read in this run; its evidence reaches the report. "
                    "Choose another source."
                )
            if not self._in_scope(url, None, request):
                return "Outside the allowed domains; choose another source."
            if budget["reads"] <= 0:
                return _READS_SPENT.format(**budget) if budget["searches"] > 0 else _BUDGET_SPENT
            budget["reads"] -= 1
            claimed.add(canonical)
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
                    task, page, hits_by_url.get(url), progress_callback, brief
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
            model=role_model(self._runtime.settings, "researcher"),
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
            # The memo is prose, so it stays plain text: DeepSeek in thinking
            # mode rejects the forced tool_choice that structured retries send.
            expected_output="A findings memo in markdown.",
        )
        memo = ""
        failure: str | None = None
        llm_usage: dict[str, Any] | None = None
        try:
            report = await researcher_call(self._runtime, worker, job)
            llm_usage = collect_llm_usage(report)
            content = getattr(report, "content", None)
            if report.status == "completed" and isinstance(content, str) and content.strip():
                memo = content.strip()
            else:
                failure = _report_reason(report)
                logger.warning("Researcher did not finish %s: %s", task.task_id, failure)
        except TimeoutError:
            failure = "researcher loop timed out"
            logger.warning("Researcher loop timed out on %s", task.task_id)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("Researcher failed on %s", task.task_id, exc_info=True)
        # The pages read before the loop stopped still count as evidence; the
        # run records that this task ended without its memo.
        if failure is not None:
            await self._emit_trace(
                progress_callback,
                "researcher_failed",
                {"task_id": task.task_id, "reason": failure},
            )

        memo_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "focus": task.focus,
            "memo": memo,
            "searches": request.max_searches_per_task - budget["searches"],
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
        extracted_urls: set[str] | None = None,
        brief: ResearchBrief | None = None,
    ) -> list[EvidenceRecord]:
        researcher = self._agentic if request.researcher == "agentic" else self._pipeline
        return await researcher.execute_task(
            run_scope, task, request, progress_callback, extracted_urls, brief
        )
