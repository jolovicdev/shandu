from __future__ import annotations

import asyncio

from shandu.services.scrape import ScrapeService


def test_scrape_timeout_increments_counter():
    service = ScrapeService()

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            class FakeResponse:
                async def __aenter__(self):
                    raise asyncio.TimeoutError()
                async def __aexit__(self, *args):
                    pass
            return FakeResponse()

    fake_session = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake_session))
    assert result is None
    assert service._timeout_count == 1
    assert service._total_scrapes == 1


def test_scrape_success_does_not_increment_timeout():
    service = ScrapeService()

    html = "<html><body><p>Hello world this is a test paragraph with enough words to pass filter</p></body></html>"

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html"}
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                def raise_for_status(self):
                    pass
                async def text(self, errors=None):
                    return html
            return FakeResponse()

    fake_session = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake_session))
    assert result is not None
    assert result.text == "Hello world this is a test paragraph with enough words to pass filter"
    assert service._timeout_count == 0
    assert service._total_scrapes == 1
