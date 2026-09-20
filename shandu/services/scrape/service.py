from __future__ import annotations

import asyncio
import ipaddress
import logging
import random
import re
import socket
from collections import OrderedDict
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

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

_PAGE_CACHE_MAX = 128
_BLOCKED_PRIVATE_FETCH_ERROR = "blocked_private_host"
_MAX_REDIRECT_HOPS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class _PrivateAddressError(aiohttp.ClientError):
    """Every resolved address for a host is non-public."""


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    return ip.is_global


def _record_host_is_public(value: str) -> bool:
    try:
        return _ip_is_public(ipaddress.ip_address(value))
    except ValueError:
        pass
    try:
        normalized = socket.inet_ntoa(socket.inet_aton(value))
    except (OSError, ValueError):
        return False
    return _ip_is_public(ipaddress.ip_address(normalized))


def _literal_host_blocked(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            normalized = socket.inet_ntoa(socket.inet_aton(host))
        except (OSError, ValueError):
            return False
        ip = ipaddress.ip_address(normalized)
    return not _ip_is_public(ip)


def _blocked_host_in_url(url: str) -> str | None:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return url
    if host and _literal_host_blocked(host):
        return host
    return None


class _PublicAddressConnector(aiohttp.TCPConnector):
    async def _resolve_host(
        self, host: str, port: int, traces: Any = None
    ) -> list[dict[str, Any]]:
        records = await super()._resolve_host(host, port, traces=traces)
        public = [
            record
            for record in records
            if _record_host_is_public(str(record.get("host", "")))
        ]
        if not public:
            raise _PrivateAddressError(f"blocked non-public address for {host}")
        return public


async def _proxy_target_blocked(url: str) -> bool:
    try:
        parts = urlsplit(url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return True
    host = parts.hostname or ""
    if not host:
        return True
    resolver = aiohttp.DefaultResolver()
    try:
        try:
            records = await resolver.resolve(host, port)
        except Exception:
            # This check is the only private-address guard on the proxy
            # path, so an unresolvable host is refused.
            return True
        return not any(
            _record_host_is_public(str(record.get("host", ""))) for record in records
        )
    finally:
        await resolver.close()


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


_ARXIV_ABSTRACT_URL = re.compile(r"^https?://(?:www\.)?arxiv\.org/abs/(?P<paper>[^?#]+)")


def _fulltext_url(url: str) -> str:
    # An arXiv abstract page holds only the abstract; the paper is the PDF.
    match = _ARXIV_ABSTRACT_URL.match(url)
    if match:
        return f"https://arxiv.org/pdf/{match['paper']}"
    return url


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
        self._page_cache: OrderedDict[str, ScrapedPage] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[ScrapedPage]] = {}
        self._session: aiohttp.ClientSession | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None
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
        session = await self._shared_session()
        tasks = [self.scrape(url, session=session) for url in normalized]
        results = await asyncio.gather(*tasks, return_exceptions=True)
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
            self._page_cache.move_to_end(normalized_url)
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

        try:
            page = await self._scrape_with_retry(url, active_session, attempt=0)
        finally:
            if owns_session and not active_session.closed:
                await active_session.close()

        if page.fetch_error is None:
            self._store_page(url, page)
            final_key = _canonicalize_url(page.url)
            if final_key != url:
                self._store_page(final_key, page.model_copy(update={"requested_url": final_key}))
        return page

    def _store_page(self, key: str, page: ScrapedPage) -> None:
        self._page_cache[key] = page
        self._page_cache.move_to_end(key)
        while len(self._page_cache) > _PAGE_CACHE_MAX:
            self._page_cache.popitem(last=False)

    async def _scrape_with_retry(
        self,
        url: str,
        session: aiohttp.ClientSession,
        attempt: int,
    ) -> ScrapedPage:
        result = await self._fetch_one(url, session, attempt)
        max_attempts = (
            min(2, self._max_attempts)
            if result.fetch_error == "timeout"
            else self._max_attempts
        )
        if result.retryable and attempt < max_attempts - 1:
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
        request_url = _fulltext_url(url)
        hop = 0
        while True:
            outcome = await self._fetch_single_request(
                url, request_url, session, attempt, hop
            )
            if isinstance(outcome, _FetchResult):
                return outcome
            request_url = outcome
            hop += 1

    async def _fetch_single_request(
        self,
        url: str,
        request_url: str,
        session: aiohttp.ClientSession,
        attempt: int,
        hop: int,
    ) -> _FetchResult | str:
        domain = urlparse(request_url).netloc

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

        if _blocked_host_in_url(request_url) is not None:
            logger.warning("Refusing to fetch URL with non-public host: %s", request_url)
            return _error_page(_BLOCKED_PRIVATE_FETCH_ERROR, None, retryable=False)

        if self._proxy and await _proxy_target_blocked(request_url):
            logger.warning(
                "Refusing to fetch URL with non-public host via proxy: %s", request_url
            )
            return _error_page(_BLOCKED_PRIVATE_FETCH_ERROR, None, retryable=False)

        await self._domain_scheduler.acquire(domain)
        try:
            async with self._semaphore:
                try:
                    headers = dict(self._headers)
                    if attempt > 0:
                        ua_index = attempt % len(_USER_AGENTS)
                        headers["User-Agent"] = _USER_AGENTS[ua_index]

                    kwargs: dict[str, object] = {"allow_redirects": False, "headers": headers}
                    if self._proxy:
                        kwargs["proxy"] = self._proxy

                    async with session.get(request_url, **kwargs) as response:
                        status = response.status

                        if status in _REDIRECT_STATUSES:
                            if hop >= _MAX_REDIRECT_HOPS:
                                return _error_page("scrape_failed", status, retryable=False)
                            location = response.headers.get("location", "")
                            next_url = (
                                _canonicalize_url(urljoin(request_url, location))
                                if location
                                else ""
                            )
                            if not next_url:
                                return _error_page("scrape_failed", status, retryable=False)
                            return next_url

                        if status in _RETRYABLE_STATUSES:
                            self._domain_scheduler.bump_backoff(domain)
                            return _error_page(_error_for_status(status), status, retryable=True)

                        if status >= 400:
                            return _error_page(_error_for_status(status), status, retryable=False)

                        self._domain_scheduler.reset_backoff(domain)

                        content_type = response.headers.get("content-type", "").lower()
                        final_url = _canonicalize_url(str(response.url)) or request_url
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
                                fetch_error = _detect_fetch_error(html, result.text) or "empty_content"
                            else:
                                fetch_error = _detect_fetch_error(html, result.text)
                            page = ScrapedPage(
                                requested_url=url,
                                url=final_url,
                                title=result.title or final_url,
                                text=result.text,
                                blocks=result.blocks,
                                domain=urlparse(final_url).netloc,
                                site_name=result.site_name,
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
                                published_at=result.published_at,
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
                                result = await asyncio.to_thread(_parse_plaintext, data)
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
                    logger.warning(
                        "Scrape timeout: %s (timeout=%ss, attempt=%s)",
                        request_url,
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
                    logger.warning("Retryable scrape exception for %s: %s", request_url, exc)
                    return _error_page("scrape_failed", None, retryable=True)
                except _PrivateAddressError:
                    logger.warning(
                        "Refusing to fetch URL with non-public host: %s", request_url
                    )
                    return _error_page(_BLOCKED_PRIVATE_FETCH_ERROR, None, retryable=False)
                except Exception as exc:
                    logger.warning("Scrape exception for %s: %s", request_url, exc)
                    return _error_page("scrape_failed", None, retryable=False)
        finally:
            await self._domain_scheduler.release(domain)

    async def _get_session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        # With a proxy the connector only resolves the operator-configured
        # proxy host, so targets are validated by precheck instead.
        connector_cls: type[aiohttp.TCPConnector] = (
            aiohttp.TCPConnector if self._proxy else _PublicAddressConnector
        )
        connector = connector_cls(
            limit=max(8, self._max_concurrent * 4), ttl_dns_cache=300
        )
        return aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def _shared_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        session = self._session
        if session is not None and not session.closed and self._session_loop is loop:
            return session
        self._session = await self._get_session()
        self._session_loop = loop
        return self._session

    async def aclose(self) -> None:
        session = self._session
        self._session = None
        self._session_loop = None
        if session is not None and not session.closed:
            await session.close()
