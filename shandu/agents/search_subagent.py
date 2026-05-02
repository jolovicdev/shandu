from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

from blackgeorge import Job, Worker
from blackgeorge.utils import new_id
from pydantic import BaseModel, Field

from ..contracts import EvidenceRecord, ResearchRequest, SubagentTask
from ..interfaces import RuntimeExecutionLike, ScrapeServiceLike, SearchServiceLike
from ..prompts import extractor_instructions, extractor_job

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
        canonical_urls = {url: self._canonicalize_url(url) or url for url in urls}
        await self._emit_trace(
            progress_callback,
            "scrape_started",
            {
                "task_id": task.task_id,
                "url_count": len(urls),
                "urls": urls,
            },
        )
        pages, scrape_missed = await self._scrape.scrape_many(urls)
        successful_pages = [p for p in pages if p.fetch_error is None]
        missed_pages = [p for p in pages if p.fetch_error is not None]
        missed_count = max(scrape_missed, len(missed_pages), max(0, len(urls) - len(pages)))
        await self._emit_trace(
            progress_callback,
            "scrape_completed",
            {
                "task_id": task.task_id,
                "scraped": len(successful_pages),
                "missed": missed_count,
                "urls": [page.url for page in successful_pages],
            },
        )
        pages_by_url = {page.requested_url: page for page in pages}
        hits_by_url: dict[str, dict[str, str]] = {}
        for entry in all_hits:
            hits_by_url[entry["url"]] = entry
            canonical_hit_url = self._canonicalize_url(entry["url"])
            if canonical_hit_url:
                hits_by_url.setdefault(canonical_hit_url, entry)

        evidence: list[EvidenceRecord] = []
        for page in pages:
            if page.fetch_error is not None:
                hit_payload = hits_by_url.get(page.requested_url)
                snippet = str(hit_payload.get("snippet", "") if hit_payload else "").strip()
                title = str(hit_payload.get("title", "") if hit_payload else page.title).strip() or page.url
                extracted_text = snippet or title
                evidence.append(
                    EvidenceRecord(
                        evidence_id=new_id(),
                        task_id=task.task_id,
                        query=task.focus,
                        requested_url=page.requested_url,
                        domain=page.domain,
                        title=title,
                        snippet=snippet or title,
                        extracted_text=extracted_text,
                        confidence=0.33,
                        source_type="search_snippet",
                        extraction_method="search_snippet_fallback",
                        fetch_error=page.fetch_error,
                    )
                )
                await self._emit_trace(
                    progress_callback,
                    "fallback_evidence",
                    {
                        "task_id": task.task_id,
                        "url": page.url,
                        "title": title,
                        "confidence": 0.33,
                        "fetch_error": page.fetch_error,
                    },
                )
                continue

            await self._emit_trace(
                progress_callback,
                "extract_started",
                {
                    "task_id": task.task_id,
                    "url": page.url,
                    "title": page.title,
                },
            )
            extraction = await self._extract(task, page.url, page.title, page.text, progress_callback)
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
            source_type, extraction_method = self._source_metadata(page.content_type, page.url)
            evidence.append(
                EvidenceRecord(
                    evidence_id=new_id(),
                    task_id=task.task_id,
                    query=task.focus,
                    requested_url=page.requested_url,
                    final_url=page.url,
                    domain=page.domain,
                    title=page.title,
                    snippet=extraction.snippet,
                    extracted_text=extraction.extracted_text,
                    confidence=extraction.confidence,
                    fetched_at=page.fetched_at,
                    published_at=page.published_at,
                    source_type=source_type,
                    extraction_method=extraction_method,
                )
            )

        for url in urls:
            canonical_url = canonical_urls.get(url, url)
            if url in pages_by_url or canonical_url in pages_by_url:
                continue
            hit_payload = hits_by_url.get(url) or hits_by_url.get(canonical_url)
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
                    requested_url=url,
                    domain=urlparse(url).netloc or None,
                    title=title,
                    snippet=snippet or title,
                    extracted_text=extracted_text,
                    confidence=0.33,
                    source_type="search_snippet",
                    extraction_method="search_snippet_fallback",
                    fetch_error="scrape_failed",
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

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return ""
        path = parts.path or "/"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))

    @staticmethod
    def _source_metadata(content_type: str | None, url: str) -> tuple[str, str]:
        ct = (content_type or "").lower()
        path = urlparse(url).path.lower()
        if "pdf" in ct or path.endswith(".pdf"):
            return "document", "pdf"
        if "wordprocessingml.document" in ct or "msword" in ct or path.endswith(".docx"):
            return "document", "docx"
        if "spreadsheetml.sheet" in ct or "excel" in ct or path.endswith(".xlsx"):
            return "data_table", "xlsx"
        if "csv" in ct or path.endswith(".csv"):
            return "data_table", "csv"
        if "text/plain" in ct or path.endswith(".txt"):
            return "document", "plaintext"
        if "text/markdown" in ct or "text/x-markdown" in ct or path.endswith(".md"):
            return "document", "markdown"
        return "webpage", "main_html"

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
        progress_callback: SearchTraceCallback | None = None,
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
            instructions=extractor_instructions(),
        )
        job = Job(
            input=extractor_job(payload),
            response_schema=_ExtractionPayload,
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            if report.status == "completed" and isinstance(report.data, _ExtractionPayload):
                return report.data
        except Exception:
            pass

        await self._emit_trace(
            progress_callback,
            "extraction_fallback",
            {"task_id": task.task_id, "url": url, "title": title},
        )

        fallback_snippet = text[:320].strip()
        fallback_body = text[:2200].strip()
        return _ExtractionPayload(
            snippet=fallback_snippet or title,
            extracted_text=fallback_body or title,
            confidence=0.45,
        )
