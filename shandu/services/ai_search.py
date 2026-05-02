from __future__ import annotations

from blackgeorge import Job, Worker

from ..contracts import AISearchResult, AISearchSource
from ..interfaces import DetailLevel, RuntimeExecutionLike, ScrapeServiceLike, SearchServiceLike
from ..prompts import aisearch_expected_output, aisearch_instructions, aisearch_job


class AISearchService:
    def __init__(
        self,
        runtime: RuntimeExecutionLike,
        search_service: SearchServiceLike,
        scrape_service: ScrapeServiceLike,
    ) -> None:
        self._runtime = runtime
        self._search = search_service
        self._scrape = scrape_service

    async def search(
        self,
        query: str,
        max_results: int = 8,
        max_pages: int = 3,
        detail_level: DetailLevel = "standard",
    ) -> AISearchResult:
        hits = await self._search.search(query, max_results=max(1, min(max_results, 20)))
        urls = [hit.url for hit in hits[: max(1, min(max_pages, 10))]]
        scraped_pages, scrape_missed = await self._scrape.scrape_many(urls)
        scraped_by_url = {page.requested_url: page for page in scraped_pages}

        sources: list[AISearchSource] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.url in seen:
                continue
            seen.add(hit.url)
            page = scraped_by_url.get(hit.url)
            excerpt = ""
            if page is not None:
                excerpt = page.text[:1400].strip()
            snippet = hit.snippet.strip() if hit.snippet.strip() else excerpt[:300]
            sources.append(
                AISearchSource(
                    title=hit.title.strip() or hit.url,
                    url=hit.url,
                    snippet=snippet,
                    text_excerpt=excerpt,
                )
            )

        if not sources:
            return AISearchResult(
                query=query,
                answer_markdown=f"# {query}\n\nNo search results were returned for this query.",
                sources=[],
                run_stats={"sources": 0, "scraped_pages": 0},
            )

        min_words = self._word_target(detail_level)
        payload = {
            "query": query,
            "detail_level": detail_level,
            "sources": [source.model_dump(mode="json") for source in sources],
        }
        worker = Worker(
            name="AISearchAnalyst",
            model=self._runtime.settings.model,
            instructions=aisearch_instructions(),
        )
        job = Job(
            input=aisearch_job(payload, min_words),
            expected_output=aisearch_expected_output(),
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            content = getattr(report, "content", None)
            if report.status == "completed" and isinstance(content, str) and content.strip():
                return AISearchResult(
                    query=query,
                    answer_markdown=content.strip(),
                    sources=sources,
                    run_stats={
                        "sources": len(sources),
                        "scraped_pages": len(scraped_pages),
                        "scrape_missed": scrape_missed,
                    },
                )
        except Exception:
            pass

        fallback_lines = [f"# {query}", "", "## Answer", ""]
        for idx, source in enumerate(sources[:8], start=1):
            snippet = source.snippet or source.text_excerpt[:260]
            if not snippet:
                continue
            fallback_lines.append(f"{snippet} [{idx}]")
            fallback_lines.append("")
        fallback_lines.extend(["## Sources", ""])
        for idx, source in enumerate(sources, start=1):
            fallback_lines.append(f"[{idx}] {source.title} - {source.url}")
        return AISearchResult(
            query=query,
            answer_markdown="\n".join(fallback_lines).strip(),
            sources=sources,
            run_stats={
                "sources": len(sources),
                "scraped_pages": len(scraped_pages),
                "scrape_missed": scrape_missed,
            },
        )

    @staticmethod
    def _word_target(detail_level: DetailLevel) -> int:
        if detail_level == "concise":
            return 700
        if detail_level == "standard":
            return 1300
        return 2000
