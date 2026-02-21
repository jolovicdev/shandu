from __future__ import annotations

import asyncio
from types import SimpleNamespace

from shandu.services.ai_search import AISearchService
from shandu.services.search import SearchHit
from shandu.services.scrape import ScrapedPage


class FakeDesk:
    def __init__(self, content: str) -> None:
        self._content = content

    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(status="completed", content=self._content)


class FakeRuntime:
    def __init__(self, content: str) -> None:
        self.settings = SimpleNamespace(model="deepseek/deepseek-chat")
        self.desk = FakeDesk(content)


class FakeSearchService:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del query, max_results
        return [
            SearchHit(
                query="q",
                url="https://example.com/a",
                title="Example A",
                snippet="Snippet A",
            ),
            SearchHit(
                query="q",
                url="https://example.com/b",
                title="Example B",
                snippet="Snippet B",
            ),
        ]


class EmptySearchService:
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        del query, max_results
        return []


class FakeScrapeService:
    async def scrape_many(self, urls: list[str]) -> list[ScrapedPage]:
        return [
            ScrapedPage(
                url=url,
                title=f"Title {idx}",
                text=f"Long text for {url}",
                domain="example.com",
            )
            for idx, url in enumerate(urls, start=1)
        ]


def test_ai_search_returns_model_answer_when_available() -> None:
    service = AISearchService(
        runtime=FakeRuntime("# Result\n\n## Answer\nBody [1]"),
        search_service=FakeSearchService(),
        scrape_service=FakeScrapeService(),
    )

    result = asyncio.run(service.search("test query"))
    assert "## Answer" in result.answer_markdown
    assert len(result.sources) == 2


def test_ai_search_handles_empty_sources() -> None:
    service = AISearchService(
        runtime=FakeRuntime(""),
        search_service=EmptySearchService(),
        scrape_service=FakeScrapeService(),
    )

    result = asyncio.run(service.search("missing"))
    assert "No search results were returned" in result.answer_markdown
    assert result.sources == []
