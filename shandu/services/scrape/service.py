from __future__ import annotations

import asyncio
import logging
import random
from urllib.parse import urlparse, urlsplit, urlunsplit

import aiohttp

from ...config import config
from .constants import (
    _HEADERS,
    _MAX_DOWNLOAD_BYTES,
    _RETRYABLE_STATUSES,
    _USER_AGENTS,
)
from .extraction import (
    _detect_fetch_error,
    _error_for_status,
    _extract_html,
    _extract_published_at,
    _guess_format,
    _parse_csv,
    _parse_docx,
    _parse_pdf,
    _parse_plaintext,
    _parse_xlsx,
)
from .models import ScrapedPage, _FetchResult, _ParseError
from .scheduler import _DomainScheduler

logger = logging.getLogger(__name__)


async def _read_limited_response(response: aiohttp.ClientResponse) -> bytes | None:
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_DOWNLOAD_BYTES:
                return None
        except ValueError:
            pass

    stream = getattr(response, "content", None)
    if stream is not None and hasattr(stream, "iter_chunked"):
        chunks: list[bytes] = []
        total = 0
        async for chunk in stream.iter_chunked(64 * 1024):
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                return None
            chunks.append(chunk)
        return b"".join(chunks)
    if stream is not None and hasattr(stream, "read"):
        data = await stream.read()
        if len(data) > _MAX_DOWNLOAD_BYTES:
            return None
        return data

    if hasattr(response, "text"):
        text = await response.text(errors="ignore")
        try:
            data = text.encode(getattr(response, "charset", None) or "utf-8", errors="ignore")
        except LookupError:
            data = text.encode("utf-8", errors="ignore")
        if len(data) > _MAX_DOWNLOAD_BYTES:
            return None
        return data

    data = await response.read()
    if len(data) > _MAX_DOWNLOAD_BYTES:
        return None
    return data


def _canonicalize_url(url: str) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return ""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _safe_decode(data: bytes, charset: str | None) -> str:
    try:
        return data.decode(charset or "utf-8", errors="ignore")
    except LookupError:
        return data.decode("utf-8", errors="ignore")


class ScrapeService:
    def __init__(self) -> None:
        self._timeout = int(config.get("scraper", "timeout", 20))
        self._max_concurrent = int(config.get("scraper", "max_concurrent", 5))
        self._proxy = config.get("scraper", "proxy")
        self._semaphore = asyncio.Semaphore(max(1, min(self._max_concurrent, 12)))
        self._domain_scheduler = _DomainScheduler(
            max_concurrent_per_domain=int(config.get("scraper", "max_concurrent_per_domain", 2)),
            base_delay=float(config.get("scraper", "domain_base_delay", 0.5)),
        )
        self._max_attempts = max(1, min(int(config.get("scraper", "max_attempts", 3)), 5))
        self._timeout_count = 0
        self._total_scrapes = 0
        self._retry_count = 0
        self._page_cache: dict[str, ScrapedPage] = {}
        self._inflight: dict[str, asyncio.Task[ScrapedPage]] = {}
        self._headers = dict(_HEADERS)
        self._headers["User-Agent"] = _USER_AGENTS[0]

    async def scrape_many(self, urls: list[str]) -> tuple[list[ScrapedPage], int]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in urls:
            url = _canonicalize_url(raw)
            if not url or url in seen:
                continue
            seen.add(url)
            normalized.append(url)
        session = await self._get_session()
        try:
            tasks = [self.scrape(url, session=session) for url in normalized]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if not session.closed:
                await session.close()
        pages: list[ScrapedPage] = []
        for url, result in zip(normalized, results):
            if isinstance(result, ScrapedPage):
                pages.append(result)
            elif isinstance(result, BaseException):
                pages.append(
                    ScrapedPage(
                        requested_url=url,
                        url=url,
                        title=url,
                        text="",
                        domain=urlparse(url).netloc,
                        fetch_error="scrape_failed",
                    )
                )
        missed = sum(1 for p in pages if p.fetch_error is not None)
        return pages, missed

    async def scrape(
        self,
        url: str,
        session: aiohttp.ClientSession | None = None,
    ) -> ScrapedPage:
        normalized_url = _canonicalize_url(url)
        if not normalized_url:
            return ScrapedPage(
                requested_url=url,
                url=url,
                title=url,
                text="",
                domain=urlparse(url).netloc or "",
                fetch_error="scrape_failed",
            )

        cached = self._page_cache.get(normalized_url)
        if cached is not None:
            return cached

        in_flight = self._inflight.get(normalized_url)
        if in_flight is not None:
            return await in_flight

        task = asyncio.create_task(self._do_scrape(normalized_url, session))
        self._inflight[normalized_url] = task
        try:
            return await task
        finally:
            self._inflight.pop(normalized_url, None)

    async def _do_scrape(
        self,
        url: str,
        session: aiohttp.ClientSession | None = None,
    ) -> ScrapedPage:
        active_session = session or await self._get_session()
        owns_session = session is None
        self._total_scrapes += 1

        try:
            page = await self._scrape_with_retry(url, active_session, attempt=0)
        finally:
            if owns_session and not active_session.closed:
                await active_session.close()

        if page.fetch_error is None:
            self._page_cache[url] = page
            final_key = _canonicalize_url(page.url)
            if final_key != url:
                self._page_cache[final_key] = page.model_copy(update={"requested_url": final_key})
        return page

    async def _scrape_with_retry(
        self,
        url: str,
        session: aiohttp.ClientSession,
        attempt: int,
    ) -> ScrapedPage:
        result = await self._fetch_one(url, session, attempt)
        if result.retryable and attempt < self._max_attempts - 1:
            self._retry_count += 1
            delay = self._backoff_delay(attempt)
            await asyncio.sleep(delay)
            return await self._scrape_with_retry(url, session, attempt + 1)
        return result.page

    def _backoff_delay(self, attempt: int) -> float:
        base = 2 ** attempt
        jitter = random.random()
        return base + jitter

    async def _fetch_one(
        self,
        url: str,
        session: aiohttp.ClientSession,
        attempt: int,
    ) -> _FetchResult:
        domain = urlparse(url).netloc

        def _error_page(
            fetch_error: str, status: int | None = None, retryable: bool = False
        ) -> _FetchResult:
            return _FetchResult(
                ScrapedPage(
                    requested_url=url,
                    url=url,
                    title=url,
                    text="",
                    domain=domain,
                    fetch_error=fetch_error,
                    http_status=status,
                ),
                status=status,
                fetch_error=fetch_error,
                retryable=retryable,
            )

        await self._domain_scheduler.acquire(domain)
        try:
            async with self._semaphore:
                try:
                    headers = dict(self._headers)
                    if attempt > 0:
                        ua_index = attempt % len(_USER_AGENTS)
                        headers["User-Agent"] = _USER_AGENTS[ua_index]

                    kwargs: dict[str, object] = {"allow_redirects": True, "headers": headers}
                    if self._proxy:
                        kwargs["proxy"] = self._proxy

                    async with session.get(url, **kwargs) as response:
                        status = response.status

                        if status in _RETRYABLE_STATUSES:
                            self._domain_scheduler.bump_backoff(domain)
                            return _error_page(_error_for_status(status), status, retryable=True)

                        if status >= 400:
                            return _error_page(_error_for_status(status), status, retryable=False)

                        self._domain_scheduler.reset_backoff(domain)

                        content_type = response.headers.get("content-type", "").lower()
                        final_url = _canonicalize_url(str(response.url)) or url
                        fmt = _guess_format(final_url, content_type)

                        if fmt == "html":
                            data = await _read_limited_response(response)
                            if data is None:
                                return _error_page("non_text_content", status, retryable=False)
                            html = _safe_decode(data, getattr(response, "charset", None))
                            result = await asyncio.to_thread(_extract_html, html)
                            published_at = result.published_at or await asyncio.to_thread(
                                _extract_published_at, html
                            )
                            if not result.text.strip():
                                fetch_error = _detect_fetch_error(html, status, result.text) or "empty_content"
                            else:
                                fetch_error = _detect_fetch_error(html, status, result.text)
                            page = ScrapedPage(
                                requested_url=url,
                                url=final_url,
                                title=result.title or final_url,
                                text=result.text,
                                blocks=result.blocks,
                                domain=urlparse(final_url).netloc,
                                published_at=published_at,
                                content_type=content_type,
                                fetch_error=fetch_error,
                                http_status=status,
                            )
                            return _FetchResult(page, status, fetch_error, retryable=False)

                        if fmt == "":
                            return _error_page("non_text_content", status, retryable=False)

                        if fmt == "pdf":
                            data = await _read_limited_response(response)
                            if data is None:
                                return _error_page("non_text_content", status, retryable=False)
                            try:
                                result = await asyncio.to_thread(_parse_pdf, data)
                            except _ParseError as exc:
                                return _error_page(exc.fetch_error, status, retryable=False)
                            page = ScrapedPage(
                                requested_url=url,
                                url=final_url,
                                title=result.title or final_url,
                                text=result.text,
                                blocks=result.blocks,
                                domain=urlparse(final_url).netloc,
                                content_type=content_type,
                                http_status=status,
                            )
                            return _FetchResult(page, status, None, retryable=False)

                        if fmt == "docx":
                            data = await _read_limited_response(response)
                            if data is None:
                                return _error_page("non_text_content", status, retryable=False)
                            try:
                                result = await asyncio.to_thread(_parse_docx, data)
                            except _ParseError as exc:
                                return _error_page(exc.fetch_error, status, retryable=False)
                            page = ScrapedPage(
                                requested_url=url,
                                url=final_url,
                                title=result.title or final_url,
                                text=result.text,
                                blocks=result.blocks,
                                domain=urlparse(final_url).netloc,
                                content_type=content_type,
                                http_status=status,
                            )
                            return _FetchResult(page, status, None, retryable=False)

                        if fmt == "xlsx":
                            data = await _read_limited_response(response)
                            if data is None:
                                return _error_page("non_text_content", status, retryable=False)
                            try:
                                result = await asyncio.to_thread(_parse_xlsx, data)
                            except _ParseError as exc:
                                return _error_page(exc.fetch_error, status, retryable=False)
                            page = ScrapedPage(
                                requested_url=url,
                                url=final_url,
                                title=result.title or final_url,
                                text=result.text,
                                blocks=result.blocks,
                                domain=urlparse(final_url).netloc,
                                content_type=content_type,
                                http_status=status,
                            )
                            return _FetchResult(page, status, None, retryable=False)

                        if fmt == "csv":
                            data = await _read_limited_response(response)
                            if data is None:
                                return _error_page("non_text_content", status, retryable=False)
                            try:
                                result = await asyncio.to_thread(_parse_csv, data)
                            except _ParseError as exc:
                                return _error_page(exc.fetch_error, status, retryable=False)
                            page = ScrapedPage(
                                requested_url=url,
                                url=final_url,
                                title=result.title or final_url,
                                text=result.text,
                                blocks=result.blocks,
                                domain=urlparse(final_url).netloc,
                                content_type=content_type,
                                http_status=status,
                            )
                            return _FetchResult(page, status, None, retryable=False)

                        if fmt in ("txt", "md"):
                            data = await _read_limited_response(response)
                            if data is None:
                                return _error_page("non_text_content", status, retryable=False)
                            try:
                                result = await asyncio.to_thread(_parse_plaintext, data, content_type)
                            except _ParseError as exc:
                                return _error_page(exc.fetch_error, status, retryable=False)
                            page = ScrapedPage(
                                requested_url=url,
                                url=final_url,
                                title=result.title or final_url,
                                text=result.text,
                                blocks=result.blocks,
                                domain=urlparse(final_url).netloc,
                                content_type=content_type,
                                http_status=status,
                            )
                            return _FetchResult(page, status, None, retryable=False)

                        return _error_page("non_text_content", status, retryable=False)

                except asyncio.TimeoutError:
                    self._timeout_count += 1
                    logger.warning(
                        "Scrape timeout: %s (timeout=%ss, attempt=%s)",
                        url,
                        self._timeout,
                        attempt + 1,
                    )
                    return _error_page("timeout", None, retryable=True)
                except aiohttp.ClientResponseError as exc:
                    status = exc.status
                    if status in _RETRYABLE_STATUSES:
                        self._domain_scheduler.bump_backoff(domain)
                        return _error_page(_error_for_status(status), status, retryable=True)
                    return _error_page(_error_for_status(status), status, retryable=False)
                except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                    logger.warning("Retryable scrape exception for %s: %s", url, exc)
                    return _error_page("scrape_failed", None, retryable=True)
                except Exception as exc:
                    logger.warning("Scrape exception for %s: %s", url, exc)
                    return _error_page("scrape_failed", None, retryable=False)
        finally:
            await self._domain_scheduler.release(domain)

    def _canonicalize_url(self, url: str) -> str:
        return _canonicalize_url(url)

    def _extract(self, html: str) -> tuple[str, str]:
        result = _extract_html(html)
        return result.title, result.text

    @staticmethod
    def _extract_published_at(html: str) -> str | None:
        return _extract_published_at(html)

    @staticmethod
    def _extract_title(soup: object) -> str:
        from .extraction import _extract_title_from_soup

        return _extract_title_from_soup(soup)  # type: ignore[arg-type]

    async def _get_session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        connector = aiohttp.TCPConnector(
            limit=max(8, self._max_concurrent * 4), ttl_dns_cache=300
        )
        return aiohttp.ClientSession(timeout=timeout, connector=connector)
