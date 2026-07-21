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
                status = 200
                async def __aenter__(self):
                    raise asyncio.TimeoutError()
                async def __aexit__(self, *args):
                    pass
            return FakeResponse()

    fake_session = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake_session))
    assert result is not None
    assert result.fetch_error == "timeout"
    # Timeouts get at most one retry: initial attempt + 1 retry = 2 timeouts.
    assert service._timeout_count == 2
    assert service._retry_count == 1
    assert service._total_scrapes == 1


def test_retryable_status_uses_full_attempt_budget():
    service = ScrapeService()
    service._backoff_delay = lambda attempt: 0.0
    call_count = 0

    class FakeSession:
        closed = False

        async def close(self):
            pass

        def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1

            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html"}
                status = 429

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            return FakeResponse()

    fake_session = FakeSession()
    result = asyncio.run(service.scrape("https://example.com", session=fake_session))
    assert result is not None
    assert result.http_status == 429
    # 429 is retryable but not a timeout, so it uses the full attempt budget.
    assert call_count == service._max_attempts
    assert service._retry_count == service._max_attempts - 1


def test_scrape_success_after_retry():
    service = ScrapeService()

    html = "<html><body><p>Hello world this is a test paragraph with enough words to pass filter</p></body></html>"
    call_count = 0

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html"}
                status = 200
                async def __aenter__(self):
                    nonlocal call_count
                    if call_count == 1:
                        raise asyncio.TimeoutError()
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
    assert result.fetch_error is None
    assert result.text == "Hello world this is a test paragraph with enough words to pass filter"
    assert service._timeout_count == 1
    assert service._retry_count == 1
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
                status = 200
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
    assert result.fetch_error is None
    assert result.text == "Hello world this is a test paragraph with enough words to pass filter"
    assert service._timeout_count == 0
    assert service._retry_count == 0
    assert service._total_scrapes == 1


def test_scrape_cross_task_deduplication():
    service = ScrapeService()

    html = "<html><body><p>Hello world this is a test paragraph with enough words to pass filter</p></body></html>"
    call_count = 0

    class FakeSession:
        closed = False
        async def close(self):
            pass
        def get(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            class FakeResponse:
                url = "https://example.com"
                headers = {"content-type": "text/html"}
                status = 200
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
    result1 = asyncio.run(service.scrape("https://example.com", session=fake_session))
    result2 = asyncio.run(service.scrape("https://example.com", session=fake_session))

    assert result1 is not None
    assert result2 is not None
    assert result1 is result2  # same cached object
    assert call_count == 1  # only one HTTP call
    assert service._total_scrapes == 1  # one actual scrape attempt
