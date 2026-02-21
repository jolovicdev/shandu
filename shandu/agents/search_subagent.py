from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from blackgeorge import Job, Worker
from blackgeorge.utils import new_id
from pydantic import BaseModel, Field

from ..contracts import EvidenceRecord, ResearchRequest, SubagentTask
from ..interfaces import RuntimeExecutionLike, ScrapeServiceLike, SearchServiceLike

SearchTraceCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class _ExtractionPayload(BaseModel):
    snippet: str
    extracted_text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchSubagent:
    def __init__(
        self,
        runtime: RuntimeExecutionLike,
        search_service: SearchServiceLike,
        scrape_service: ScrapeServiceLike,
    ) -> None:
        self._runtime = runtime
        self._search = search_service
        self._scrape = scrape_service

    async def execute_task(
        self,
        run_scope: str,
        task: SubagentTask,
        request: ResearchRequest,
        progress_callback: SearchTraceCallback | None = None,
    ) -> list[EvidenceRecord]:
        del run_scope
        all_hits: list[dict[str, str]] = []
        seen: set[str] = set()

        for query in task.search_queries or [task.focus]:
            await self._emit_trace(
                progress_callback,
                "query_started",
                {
                    "task_id": task.task_id,
                    "focus": task.focus,
                    "query": query,
                    "max_results": request.max_results_per_query,
                },
            )
            hits = await self._search.search(query, request.max_results_per_query)
            await self._emit_trace(
                progress_callback,
                "query_completed",
                {
                    "task_id": task.task_id,
                    "query": query,
                    "hits": len(hits),
                    "urls": [hit.url for hit in hits[:8]],
                },
            )
            for hit in hits:
                if hit.url in seen:
                    continue
                seen.add(hit.url)
                all_hits.append(
                    {
                        "url": hit.url,
                        "title": hit.title,
                        "snippet": hit.snippet,
                    }
                )

        urls = [entry["url"] for entry in all_hits[: request.max_pages_per_task]]
        await self._emit_trace(
            progress_callback,
            "scrape_started",
            {
                "task_id": task.task_id,
                "url_count": len(urls),
                "urls": urls,
            },
        )
        pages = await self._scrape.scrape_many(urls)
        await self._emit_trace(
            progress_callback,
            "scrape_completed",
            {
                "task_id": task.task_id,
                "scraped": len(pages),
                "missed": max(0, len(urls) - len(pages)),
                "urls": [page.url for page in pages],
            },
        )
        pages_by_url = {page.url: page for page in pages}
        hits_by_url = {entry["url"]: entry for entry in all_hits}

        evidence: list[EvidenceRecord] = []
        for page in pages:
            await self._emit_trace(
                progress_callback,
                "extract_started",
                {
                    "task_id": task.task_id,
                    "url": page.url,
                    "title": page.title,
                },
            )
            extraction = await self._extract(task, page.url, page.title, page.text)
            await self._emit_trace(
                progress_callback,
                "extract_completed",
                {
                    "task_id": task.task_id,
                    "url": page.url,
                    "title": page.title,
                    "confidence": extraction.confidence,
                },
            )
            evidence.append(
                EvidenceRecord(
                    evidence_id=new_id(),
                    task_id=task.task_id,
                    query=task.focus,
                    url=page.url,
                    title=page.title,
                    snippet=extraction.snippet,
                    extracted_text=extraction.extracted_text,
                    confidence=extraction.confidence,
                )
            )

        for url in urls:
            if url in pages_by_url:
                continue
            hit_payload = hits_by_url.get(url)
            if hit_payload is None:
                continue
            snippet = str(hit_payload.get("snippet", "")).strip()
            title = str(hit_payload.get("title", "")).strip() or url
            extracted_text = snippet or title
            evidence.append(
                EvidenceRecord(
                    evidence_id=new_id(),
                    task_id=task.task_id,
                    query=task.focus,
                    url=url,
                    title=title,
                    snippet=snippet or title,
                    extracted_text=extracted_text,
                    confidence=0.33,
                )
            )
            await self._emit_trace(
                progress_callback,
                "fallback_evidence",
                {
                    "task_id": task.task_id,
                    "url": url,
                    "title": title,
                    "confidence": 0.33,
                },
            )

        return evidence

    async def _emit_trace(
        self,
        callback: SearchTraceCallback | None,
        trace_type: str,
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        result = callback(trace_type, payload)
        if isinstance(result, Awaitable):
            await result

    async def _extract(
        self,
        task: SubagentTask,
        url: str,
        title: str,
        text: str,
    ) -> _ExtractionPayload:
        payload = {
            "task_focus": task.focus,
            "task_expected_output": task.expected_output,
            "url": url,
            "title": title,
            "text": text[:7000],
        }
        worker = Worker(
            name=f"SubagentExtractor_{task.task_id}",
            model=self._runtime.settings.model,
            instructions=(
                "You are EvidenceExtractor for a research subagent. "
                "Produce a concise, factual snippet and a richer extracted evidence body. "
                "Prioritize relevance to task focus, preserve dates/numbers/names, and avoid generic filler. "
                "Confidence should reflect specificity, factual density, and match to task intent."
            ),
        )
        job = Job(
            input=(
                "Extract a concise snippet and evidence body from this scraped page.\n"
                "Requirements:\n"
                "- snippet: 1-3 sentences with strongest relevant claim(s).\n"
                "- extracted_text: focused, source-grounded body for downstream synthesis.\n"
                "- Do not include fabricated information.\n"
                f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
            response_schema=_ExtractionPayload,
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            if report.status == "completed" and isinstance(report.data, _ExtractionPayload):
                return report.data
        except Exception:
            pass

        fallback_snippet = text[:320].strip()
        fallback_body = text[:2200].strip()
        return _ExtractionPayload(
            snippet=fallback_snippet or title,
            extracted_text=fallback_body or title,
            confidence=0.45,
        )
