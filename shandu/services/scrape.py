from __future__ import annotations

import asyncio
from typing import Iterable
from urllib.parse import urlparse, urlsplit, urlunsplit

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ..config import config


class ScrapedPage(BaseModel):
    url: str
    title: str
    text: str
    domain: str


class ScrapeService:
    def __init__(self) -> None:
        self._timeout = int(config.get("scraper", "timeout", 20))
        self._max_concurrent = int(config.get("scraper", "max_concurrent", 5))
        self._proxy = config.get("scraper", "proxy")
        self._semaphore = asyncio.Semaphore(max(1, min(self._max_concurrent, 12)))
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def scrape_many(self, urls: list[str]) -> list[ScrapedPage]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in urls:
            url = self._canonicalize_url(raw)
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
        for result in results:
            if isinstance(result, ScrapedPage):
                pages.append(result)
        return pages

    async def scrape(
        self,
        url: str,
        session: aiohttp.ClientSession | None = None,
    ) -> ScrapedPage | None:
        normalized_url = self._canonicalize_url(url)
        if not normalized_url:
            return None
        active_session = session or await self._get_session()
        owns_session = session is None
        async with self._semaphore:
            try:
                if self._proxy:
                    response_ctx = active_session.get(
                        normalized_url,
                        allow_redirects=True,
                        headers=self._headers,
                        proxy=self._proxy,
                    )
                else:
                    response_ctx = active_session.get(
                        normalized_url,
                        allow_redirects=True,
                        headers=self._headers,
                    )
                async with response_ctx as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/" not in content_type and "html" not in content_type:
                        return None
                    html = await response.text(errors="ignore")
                    final_url = self._canonicalize_url(str(response.url)) or normalized_url
            except Exception:
                return None
            finally:
                if owns_session and not active_session.closed:
                    await active_session.close()

        title, text = self._extract(html)
        if not text:
            return None

        return ScrapedPage(
            url=final_url,
            title=title or final_url,
            text=text,
            domain=urlparse(final_url).netloc,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        connector = aiohttp.TCPConnector(limit=max(8, self._max_concurrent * 4), ttl_dns_cache=300)
        return aiohttp.ClientSession(timeout=timeout, connector=connector)

    def _extract(self, html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.select("script,style,noscript,header,footer,nav,aside,form,iframe,svg"):
            tag.decompose()

        title = self._extract_title(soup)
        section = (
            soup.find("article")
            or soup.find("main")
            or soup.select_one("[role='main']")
            or soup.find("body")
            or soup
        )

        blocks = list(
            self._clean_blocks(
                node.get_text(" ", strip=True)
                for node in section.select("p,li,h2,h3,blockquote")
            )
        )
        if len(" ".join(blocks).split()) < 120:
            blocks = list(self._clean_blocks(line for line in section.get_text("\n").splitlines()))
        text = "\n".join(blocks).strip()
        if len(text) > 18000:
            text = text[:18000].rstrip()
        return title, text

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return str(og_title["content"]).strip()
        title_tag = soup.find("title")
        if title_tag and title_tag.text:
            return title_tag.text.strip()
        h1 = soup.find("h1")
        if h1 and h1.text:
            return h1.text.strip()
        return ""

    @staticmethod
    def _clean_blocks(lines: Iterable[str]) -> Iterable[str]:
        seen: set[str] = set()
        for raw in lines:
            line = " ".join(raw.split()).strip()
            if len(line) < 35:
                continue
            lowered = line.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            yield line

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return ""
        path = parts.path or "/"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))
