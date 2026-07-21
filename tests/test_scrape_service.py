from __future__ import annotations

import asyncio

from shandu.services.scrape import ScrapeService, ScrapedPage


def test_scrape_service_canonicalizes_urls() -> None:
    service = ScrapeService()
    canonical = service._canonicalize_url("https://example.com/path?a=1#section")
    assert canonical == "https://example.com/path?a=1"


def test_scrape_service_extracts_main_content_and_drops_noise() -> None:
    service = ScrapeService()
    title, text = service._extract(
        """
        <html>
          <head>
            <title>Sample Article</title>
            <script>const token = "ignore me"</script>
          </head>
          <body>
            <header>header nav</header>
            <article>
              <p>This is a long paragraph that should be included in the extracted output because it is informative and content-rich.</p>
              <p>This is another long paragraph with additional context about the same topic and enough length to pass filtering.</p>
            </article>
          </body>
        </html>
        """
    )
    assert title == "Sample Article"
    assert "informative and content-rich" in text
    assert "ignore me" not in text
    assert "header nav" not in text


def test_safe_decode_falls_back_on_unknown_charset() -> None:
    from shandu.services.scrape.service import _safe_decode

    data = b"<html><body>ok</body></html>"
    assert _safe_decode(data, "utf-8") == "<html><body>ok</body></html>"
    assert _safe_decode(data, "utf8mb4") == "<html><body>ok</body></html>"
    assert _safe_decode(data, None) == "<html><body>ok</body></html>"


def test_scrape_many_reuses_one_session() -> None:
    service = ScrapeService()
    created: list[object] = []
    used: list[object] = []

    class FakeSession:
        closed = False

        async def close(self) -> None:
            self.closed = True

    async def fake_get_session() -> object:
        session = FakeSession()
        created.append(session)
        return session

    async def fake_scrape(url: str, session: object = None) -> ScrapedPage:
        used.append(session)
        return ScrapedPage(requested_url=url, url=url, title=url, text="t", domain="d")

    service._get_session = fake_get_session  # type: ignore[assignment]
    service.scrape = fake_scrape  # type: ignore[assignment]

    async def run() -> None:
        await service.scrape_many(["https://example.com/1"])
        await service.scrape_many(["https://example.com/2"])

    asyncio.run(run())
    assert len(created) == 1
    assert used[0] is used[1]


def test_scrape_does_not_close_caller_session() -> None:
    service = ScrapeService()

    class FakeSession:
        closed = False

        async def close(self) -> None:
            self.closed = True

        def get(self, *args, **kwargs):
            class FakeResponse:
                url = "https://example.com/"
                headers = {"content-type": "text/html"}
                status = 200

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                async def text(self, errors=None):
                    return "<html><body><p>enough words here to pass the content filter easily</p></body></html>"

            return FakeResponse()

    session = FakeSession()
    asyncio.run(service.scrape("https://example.com", session=session))
    assert session.closed is False


def test_page_cache_evicts_oldest() -> None:
    from shandu.services.scrape.service import _PAGE_CACHE_MAX

    service = ScrapeService()
    for i in range(_PAGE_CACHE_MAX + 10):
        service._store_page(
            f"k{i}",
            ScrapedPage(requested_url=f"k{i}", url=f"k{i}", title="t", text="x", domain="d"),
        )
    assert len(service._page_cache) == _PAGE_CACHE_MAX
    assert "k0" not in service._page_cache
    assert f"k{_PAGE_CACHE_MAX + 9}" in service._page_cache
