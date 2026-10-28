from __future__ import annotations

import asyncio
import calendar
import logging
import re
from collections.abc import Awaitable, Callable, Collection
from datetime import date
from itertools import zip_longest
from typing import Any, Literal
from urllib.parse import urlparse, urlsplit, urlunsplit

from blackgeorge import Job, Worker
from blackgeorge.utils import new_id
from pydantic import BaseModel, Field

from ..contracts import (
    Claim,
    EvidenceRecord,
    ResearchBrief,
    ResearchRequest,
    SourceClass,
    SubagentTask,
)
from ..interfaces import (
    RuntimeExecutionLike,
    ScrapedPageLike,
    ScrapeServiceLike,
    SearchServiceLike,
)
from ..prompts import (
    extractor_instructions,
    extractor_job,
    selector_instructions,
    selector_job,
)
from ..runtime.calls import ProviderRejectedError, structured_call
from ..runtime.costing import collect_llm_usage
from ..runtime.roles import role_model
from ..services.findings import verified_claims

logger = logging.getLogger(__name__)

SearchTraceCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]

_SNIPPET_FALLBACK_METHOD = "search_snippet_fallback"
_SNIPPET_ONLY_CREDIBILITY = 0.20
_EXTRACTION_CONCURRENCY = 8
# One scrape of the top candidates, then up to two rounds that replace the
# pages that could not be read.
_SCRAPE_ROUNDS = 3


# Search engines list an arXiv page under its <title>, "[2406.02069] PyramidKV: ...".
_ARXIV_ID_PREFIX = re.compile(r"^\[\d{4}\.\d{4,5}(?:v\d+)?\]\s*")


def _cased_title(page_title: str, hit_title: str) -> str:
    """Some PDFs set the title in capitals, which also hides words the title
    breaks across lines. The search hit that led to the page names the paper
    in its own casing and reads better when it names mostly the same words; a
    preprint's hit can word the title a little differently from its PDF."""
    def words(title: str) -> set[str]:
        return set(re.sub(r"[\W_]+", " ", title.casefold()).split())

    def mostly_capitals(title: str) -> bool:
        letters = [char for char in title if char.isalpha()]
        return bool(letters) and sum(char.isupper() for char in letters) >= 0.8 * len(letters)

    hit = _ARXIV_ID_PREFIX.sub("", hit_title).strip()
    if mostly_capitals(page_title) and hit and not mostly_capitals(hit):
        page_words, hit_words = words(page_title), words(hit)
        if len(page_words & hit_words) >= 0.6 * len(page_words | hit_words):
            return hit
    return page_title


def _host_matches(url: str, domains: Collection[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _dated_before(published_at: str | None, since: date) -> bool:
    """True only when the whole period a date names ends before ``since``.

    A bare year counts as its last day and a year-month as the month's last
    day, so a coarse date never excludes a page that may fall in range.
    """
    match = re.match(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", (published_at or "").strip())
    if match is None:
        return False
    year = int(match.group(1))
    month = int(match.group(2) or 12)
    try:
        day = int(match.group(3) or calendar.monthrange(year, month)[1])
        return date(year, month, day) < since
    except ValueError:
        return False


def _report_reason(report: Any) -> str:
    errors = getattr(report, "errors", None) or []
    detail = "; ".join(str(item) for item in errors if str(item).strip())
    if detail:
        return detail[:300]
    return f"status={getattr(report, 'status', 'unknown')}"


# Every field is required: a provider that fills only required fields would
# otherwise leave confidence at a constant and the source unclassified, and
# the credibility score built from them would be noise.
class _ExtractionPayload(BaseModel):
    snippet: str
    extracted_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_class: SourceClass
    authorship: Literal["named", "organizational", "anonymous"]
    is_promotional: bool
    summarizes_inaccessible_source: bool
    claims: list[Claim]


class _SelectionPayload(BaseModel):
    ranked_ids: list[int]


def assess_source_quality(
    source_class: SourceClass,
    authorship: str,
    is_promotional: bool,
    summarizes_inaccessible_source: bool,
    has_publish_date: bool,
) -> tuple[float, list[str]]:
    """Score an assessed source from its provenance signals.

    The score starts from a per-class tier and loses small amounts for missing
    date, missing author, promotional intent, and secondhand summaries. Every
    credibility constant lives here. Callers handle the non-assessed buckets
    (snippet-only, unassessed) directly.
    """
    tiers: dict[str, float] = {
        "peer_reviewed": 0.92,
        "primary": 0.90,
        "official": 0.88,
        "technical_reference": 0.80,
        "journalism": 0.72,
        "expert_commentary": 0.62,
        "corporate": 0.52,
        "aggregator": 0.42,
        "community": 0.42,
        "personal_blog": 0.36,
        "advocacy_marketing": 0.30,
        "social_profile": 0.26,
        "unknown": 0.40,
    }
    self_published_classes = {
        "personal_blog",
        "social_profile",
        "corporate",
        "advocacy_marketing",
    }

    score = tiers[source_class]
    flags: list[str] = []
    if not has_publish_date:
        score -= 0.10
        flags.append("undated")
    if authorship == "anonymous":
        score -= 0.10
        flags.append("no_author")
    if is_promotional:
        score -= 0.15
        flags.append("promotional")
    if summarizes_inaccessible_source:
        score -= 0.15
        flags.append("secondhand")
    if source_class in self_published_classes:
        flags.append("self_published")

    return max(0.0, min(1.0, score)), flags


class SearchSubagent:
    def __init__(
        self,
        runtime: RuntimeExecutionLike,
        search_service: SearchServiceLike,
        scrape_service: ScrapeServiceLike,
        scholarly_service: SearchServiceLike | None = None,
    ) -> None:
        self._runtime = runtime
        self._search = search_service
        self._scrape = scrape_service
        self._scholarly = scholarly_service
        self._extract_semaphore = asyncio.Semaphore(_EXTRACTION_CONCURRENCY)

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
        planned = (task.search_queries or [task.focus])[: request.max_searches_per_task]
        searches: list[tuple[str, SearchServiceLike, str]] = [
            (query, self._search, "web")
            for query in self._scoped_queries(planned, request)[: request.max_searches_per_task]
        ]
        # Paper search ignores site: scoping, so each planned query runs once.
        if task.scholarly and self._scholarly is not None:
            searches += [(query, self._scholarly, "scholarly") for query in planned]
        outcomes = await asyncio.gather(
            *(
                self._traced_search(service, query, source, task, request, progress_callback)
                for query, service, source in searches
            )
        )

        per_query: list[list[dict[str, str]]] = []
        seen: set[str] = set()
        for (query, _, _), (hits, _) in zip(searches, outcomes):
            entries: list[dict[str, str]] = []
            for hit in hits:
                if hit.url in seen:
                    continue
                seen.add(hit.url)
                entries.append(
                    {
                        "url": hit.url,
                        "title": hit.title,
                        "snippet": hit.snippet,
                        "query": query,
                    }
                )
            per_query.append(entries)

        # Round-robin across queries so later queries survive the page cap.
        interleaved: list[dict[str, str]] = []
        for round_hits in zip_longest(*per_query):
            for entry in round_hits:
                if entry is not None:
                    interleaved.append(entry)

        def page_key(url: str) -> str:
            return self._canonicalize_url(url) or url

        # Pages read in earlier iterations, or claimed by a sibling task, are
        # never read again; siblings claim pages while this task ranks, so
        # the claims are checked once more after ranking.
        read = {page_key(url) for url in extracted_urls or ()}
        unseen = [entry for entry in interleaved if page_key(entry["url"]) not in read]
        unseen = await self._rank_candidates(
            task, unseen, request.max_pages_per_task, progress_callback
        )
        read = {page_key(url) for url in extracted_urls or ()}
        all_hits = [entry for entry in unseen if page_key(entry["url"]) not in read]

        since = request.since
        queue = [entry["url"] for entry in all_hits]
        urls: list[str] = []
        pages: list[ScrapedPageLike] = []
        scrape_missed = 0
        successful_pages: list[ScrapedPageLike] = []
        too_old: list[ScrapedPageLike] = []
        # A blocked, empty, or too-old page gives the task nothing to read, so
        # the next-ranked candidates take its place until the task has its
        # pages, the candidates run out, or the rounds are spent.
        for _ in range(_SCRAPE_ROUNDS):
            batch: list[str] = []
            while queue and len(batch) < request.max_pages_per_task - len(successful_pages):
                url = queue.pop(0)
                if extracted_urls is None or page_key(url) not in extracted_urls:
                    batch.append(url)
            if not batch:
                break
            if extracted_urls is not None:
                extracted_urls.update(page_key(url) for url in batch)
            await self._emit_trace(
                progress_callback,
                "scrape_started",
                {
                    "task_id": task.task_id,
                    "url_count": len(batch),
                    "urls": batch,
                },
            )
            batch_pages, batch_missed = await self._scrape.scrape_many(batch)
            urls.extend(batch)
            pages.extend(batch_pages)
            scrape_missed += batch_missed
            too_old = [
                page
                for page in pages
                if page.fetch_error is None
                and since is not None
                and _dated_before(page.published_at, since)
            ]
            successful_pages = [
                page for page in pages if page.fetch_error is None and page not in too_old
            ]
        canonical_urls = {url: page_key(url) for url in urls}
        missed_pages = [p for p in pages if p.fetch_error is not None]
        missed_count = max(scrape_missed, len(missed_pages), max(0, len(urls) - len(pages)))
        await self._emit_trace(
            progress_callback,
            "scrape_completed",
            {
                "task_id": task.task_id,
                "scraped": len(successful_pages),
                "missed": missed_count,
                "dated_before_since": len(too_old),
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

        evidence: list[EvidenceRecord] = await asyncio.gather(
            *(
                self._page_evidence(
                    task, page, hits_by_url.get(page.requested_url), progress_callback, brief
                )
                for page in pages
                if page not in too_old
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
            credibility, quality_flags = _SNIPPET_ONLY_CREDIBILITY, ["snippet_only"]
            evidence.append(
                EvidenceRecord(
                    evidence_id=new_id(),
                    task_id=task.task_id,
                    query=task.focus,
                    search_query=hit_payload.get("query") if hit_payload else None,
                    requested_url=url,
                    domain=urlparse(url).netloc or None,
                    title=title,
                    snippet=snippet or title,
                    extracted_text=extracted_text,
                    confidence=0.33,
                    source_type="search_snippet",
                    extraction_method=_SNIPPET_FALLBACK_METHOD,
                    relevance_score=0.33,
                    credibility_score=credibility,
                    quality_flags=quality_flags,
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
    def _scoped_queries(queries: list[str], request: ResearchRequest) -> list[str]:
        """Each query scoped to each included domain, one domain at a time, so a
        search cap keeps every query on the first domain before any gets a second."""
        if not request.include_domains:
            return list(queries)
        return [
            f"{query} site:{domain}" for domain in request.include_domains for query in queries
        ]

    @staticmethod
    def _in_scope(url: str, published_at: str | None, request: ResearchRequest) -> bool:
        """Whether a source passes the request's domain and date filters."""
        if request.include_domains and not _host_matches(url, request.include_domains):
            return False
        if _host_matches(url, request.exclude_domains):
            return False
        return request.since is None or not _dated_before(published_at, request.since)

    async def _traced_search(
        self,
        service: SearchServiceLike,
        query: str,
        source: str,
        task: SubagentTask,
        request: ResearchRequest,
        progress_callback: SearchTraceCallback | None,
    ) -> tuple[list[Any], str | None]:
        """Run one search with its trace events; return in-scope hits and any error."""
        await self._emit_trace(
            progress_callback,
            "query_started",
            {
                "task_id": task.task_id,
                "focus": task.focus,
                "query": query,
                "source": source,
                "max_results": request.max_results_per_query,
            },
        )
        hits = await service.search(query, request.max_results_per_query)
        error_for = getattr(service, "last_error_for", None)
        error = error_for(query, request.max_results_per_query) if error_for else None
        if error:
            await self._emit_trace(
                progress_callback,
                "search_failed",
                {"task_id": task.task_id, "query": query, "reason": error},
            )
        kept = [
            hit
            for hit in hits
            if self._in_scope(hit.url, getattr(hit, "published_at", None), request)
        ]
        await self._emit_trace(
            progress_callback,
            "query_completed",
            {
                "task_id": task.task_id,
                "query": query,
                "hits": len(kept),
                "urls": [hit.url for hit in kept[:8]],
            },
        )
        return kept, error

    async def _page_evidence(
        self,
        task: SubagentTask,
        page: ScrapedPageLike,
        hit_payload: dict[str, str] | None,
        progress_callback: SearchTraceCallback | None,
        brief: ResearchBrief | None = None,
    ) -> EvidenceRecord:
        """Turn a scraped page into evidence through extraction and source scoring.

        A page that failed to fetch keeps its search snippet as weak evidence.
        """
        if page.fetch_error is not None:
            snippet = str(hit_payload.get("snippet", "") if hit_payload else "").strip()
            title = str(hit_payload.get("title", "") if hit_payload else page.title).strip() or page.url
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
            credibility, quality_flags = _SNIPPET_ONLY_CREDIBILITY, ["snippet_only"]
            return EvidenceRecord(
                evidence_id=new_id(),
                task_id=task.task_id,
                query=task.focus,
                search_query=hit_payload.get("query") if hit_payload else None,
                requested_url=page.requested_url,
                domain=page.domain,
                title=title,
                site_name=page.site_name,
                snippet=snippet or title,
                extracted_text=snippet or title,
                confidence=0.33,
                source_type="search_snippet",
                extraction_method=_SNIPPET_FALLBACK_METHOD,
                relevance_score=0.33,
                credibility_score=credibility,
                quality_flags=quality_flags,
                fetch_error=page.fetch_error,
            )

        await self._emit_trace(
            progress_callback,
            "extract_started",
            {
                "task_id": task.task_id,
                "url": page.url,
                "title": page.title,
            },
        )
        extraction, assessed, extract_usage = await self._extract(
            task, page.url, page.title, page.text, progress_callback, brief
        )
        source_type, extraction_method = self._source_metadata(page.content_type, page.url)
        if assessed:
            source_class = extraction.source_class
            credibility, quality_flags = assess_source_quality(
                extraction.source_class,
                extraction.authorship,
                extraction.is_promotional,
                extraction.summarizes_inaccessible_source,
                page.published_at is not None,
            )
        else:
            source_class = None
            credibility, quality_flags = None, ["unassessed"]
        completed_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "url": page.url,
            "title": page.title,
            "confidence": extraction.confidence,
            "credibility": credibility,
        }
        if extract_usage:
            completed_payload["llm_usage"] = extract_usage
        await self._emit_trace(
            progress_callback,
            "extract_completed",
            completed_payload,
        )
        return EvidenceRecord(
            evidence_id=new_id(),
            task_id=task.task_id,
            query=task.focus,
            search_query=hit_payload.get("query") if hit_payload else None,
            requested_url=page.requested_url,
            final_url=page.url,
            domain=page.domain,
            title=_cased_title(page.title, hit_payload.get("title", "") if hit_payload else ""),
            site_name=page.site_name,
            snippet=extraction.snippet,
            extracted_text=extraction.extracted_text,
            confidence=extraction.confidence,
            fetched_at=page.fetched_at,
            published_at=page.published_at,
            source_type=source_type,
            extraction_method=extraction_method,
            relevance_score=extraction.confidence,
            credibility_score=credibility,
            source_class=source_class,
            quality_flags=quality_flags,
            claims=verified_claims(
                extraction.claims,
                page.text,
                len(brief.sub_questions) if brief is not None else 0,
            ),
        )

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

    async def _rank_candidates(
        self,
        task: SubagentTask,
        candidates: list[dict[str, str]],
        limit: int,
        progress_callback: SearchTraceCallback | None = None,
    ) -> list[dict[str, str]]:
        """Order candidates by expected value for the task.

        Search rank says little about which hit carries the evidence the task
        needs, so the model picks from titles and snippets. The incoming
        order is kept when the pick is unnecessary or the call fails.
        """
        if len(candidates) <= limit:
            return candidates
        payload = {
            "task_focus": task.focus,
            "task_expected_output": task.expected_output,
            "select": limit,
            "candidates": [
                {
                    "id": index,
                    "url": entry["url"],
                    "title": entry["title"],
                    "snippet": entry["snippet"][:300],
                }
                for index, entry in enumerate(candidates)
            ],
        }
        worker = Worker(
            name="SubagentSelector",
            model=role_model(self._runtime.settings, "researcher"),
            instructions=selector_instructions(),
        )
        job = Job(input=selector_job(payload), response_schema=_SelectionPayload)
        ranked = candidates
        llm_usage: dict[str, Any] | None = None
        try:
            report = await structured_call(self._runtime, worker, job)
            llm_usage = collect_llm_usage(report)
            if report.status == "completed" and isinstance(report.data, _SelectionPayload):
                picked = [
                    index
                    for index in dict.fromkeys(report.data.ranked_ids)
                    if 0 <= index < len(candidates)
                ]
                chosen = set(picked)
                ranked = [candidates[index] for index in picked] + [
                    entry
                    for index, entry in enumerate(candidates)
                    if index not in chosen
                ]
            else:
                logger.warning(
                    "URL selection did not complete for %s: %s",
                    task.task_id,
                    _report_reason(report),
                )
        except ProviderRejectedError:
            raise
        except Exception:
            logger.warning("URL selection failed for %s", task.task_id, exc_info=True)

        trace_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "candidates": len(candidates),
            "selected": [entry["url"] for entry in ranked[:limit]],
        }
        if llm_usage:
            trace_payload["llm_usage"] = llm_usage
        await self._emit_trace(progress_callback, "urls_ranked", trace_payload)
        return ranked

    async def _extract(
        self,
        task: SubagentTask,
        url: str,
        title: str,
        text: str,
        progress_callback: SearchTraceCallback | None = None,
        brief: ResearchBrief | None = None,
    ) -> tuple[_ExtractionPayload, bool, dict[str, Any] | None]:
        payload = {
            "task_focus": task.focus,
            "task_expected_output": task.expected_output,
            "sub_questions": [
                {"index": index, "question": question}
                for index, question in enumerate(
                    brief.sub_questions if brief is not None else [], start=1
                )
            ],
            "url": url,
            "title": title,
            "text": text,
        }
        worker = Worker(
            name="SubagentExtractor",
            model=role_model(self._runtime.settings, "extractor"),
            instructions=extractor_instructions(),
        )
        job = Job(
            input=extractor_job(payload),
            response_schema=_ExtractionPayload,
        )
        llm_usage: dict[str, Any] | None = None
        try:
            async with self._extract_semaphore:
                report = await structured_call(self._runtime, worker, job)
            llm_usage = collect_llm_usage(report)
            if report.status == "completed" and isinstance(report.data, _ExtractionPayload):
                return report.data, True, llm_usage
            reason = _report_reason(report)
            logger.warning("Extraction call did not complete for %s: %s", url, reason)
        except ProviderRejectedError:
            raise
        except Exception:
            reason = "extractor raised before completing"
            logger.warning("Extraction call failed for %s", url, exc_info=True)

        fallback_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "url": url,
            "title": title,
            "reason": reason,
        }
        if llm_usage:
            fallback_payload["llm_usage"] = llm_usage
        await self._emit_trace(
            progress_callback,
            "extraction_fallback",
            fallback_payload,
        )

        fallback_snippet = text[:320].strip()
        fallback_body = text[:2200].strip()
        return (
            _ExtractionPayload(
                snippet=fallback_snippet or title,
                extracted_text=fallback_body or title,
                confidence=0.45,
                source_class="unknown",
                authorship="anonymous",
                is_promotional=False,
                summarizes_inaccessible_source=False,
                claims=[],
            ),
            False,
            llm_usage,
        )
