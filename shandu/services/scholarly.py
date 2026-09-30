from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ElementTree
from itertools import zip_longest
from typing import Any

import aiohttp

from .search import SearchHit

logger = logging.getLogger(__name__)

_ARXIV_URL = "https://export.arxiv.org/api/query"
_OPENALEX_URL = "https://api.openalex.org/works"
_PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_ATOM = "{http://www.w3.org/2005/Atom}"
# arXiv asks API clients to leave 3 seconds between calls.
_ARXIV_INTERVAL_SECONDS = 3.0
# NCBI allows three E-utilities requests a second without an API key.
_PUBMED_INTERVAL_SECONDS = 0.34
_SNIPPET_CHARS = 400
_TIMEOUT = aiohttp.ClientTimeout(total=20)
_SITE_OPERATOR = re.compile(r"\s*\bsite:\S+")
# Provider records sometimes carry markup debris after the URL.
_CLEAN_URL = re.compile(r"https?://[^\s<>\"']+")
_RATE_LIMITED_RETRY_SECONDS = 2.0


class _Throttle:
    """Keeps a minimum interval between calls to one provider across tasks."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._last + self._interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _title_key(title: str) -> str:
    """Letters and digits of any script, so punctuation and case never
    separate two records of one paper."""
    return " ".join(re.sub(r"[\W_]+", " ", title.casefold()).split())


def _openalex_abstract(inverted: Any) -> str:
    if not isinstance(inverted, dict):
        return ""
    positions = [
        (position, word)
        for word, indexes in inverted.items()
        for position in indexes or []
        if isinstance(position, int)
    ]
    return " ".join(word for _, word in sorted(positions))


def _openalex_url(work: dict[str, Any]) -> str:
    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    for candidate in (
        best.get("pdf_url"),
        best.get("landing_page_url"),
        primary.get("landing_page_url"),
        work.get("doi"),
    ):
        if isinstance(candidate, str) and _CLEAN_URL.fullmatch(candidate):
            return candidate
    return ""


class ScholarlySearchService:
    """Searches arXiv, OpenAlex, and PubMed and merges their hits.

    It has the same interface as the web search service, so a research task
    sends its queries here as well when papers can answer it. A provider that
    fails drops out of that query; the query fails only when all of them do.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._arxiv_throttle = _Throttle(_ARXIV_INTERVAL_SECONDS)
        self._pubmed_throttle = _Throttle(_PUBMED_INTERVAL_SECONDS)
        self._errors: dict[str, str] = {}

    def last_error_for(self, query: str, max_results: int) -> str | None:
        return self._errors.get(f"{query}:{max_results}")

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        terms = _SITE_OPERATOR.sub("", query).strip() or query
        providers = {
            "arxiv": self._search_arxiv,
            "openalex": self._search_openalex,
            "pubmed": self._search_pubmed,
        }
        outcomes = await asyncio.gather(
            *(search(query, terms, max_results) for search in providers.values()),
            return_exceptions=True,
        )
        hit_lists: list[list[SearchHit]] = []
        failures: list[str] = []
        for name, outcome in zip(providers, outcomes):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome
                failures.append(f"{name}: {outcome}")
                logger.warning("Scholarly search on %s failed: %s", name, outcome)
            else:
                hit_lists.append(outcome)
        key = f"{query}:{max_results}"
        if len(failures) == len(providers):
            self._errors[key] = "; ".join(failures)[:300]
        else:
            self._errors.pop(key, None)

        merged: list[SearchHit] = []
        seen_titles: set[str] = set()
        for round_hits in zip_longest(*hit_lists):
            for hit in round_hits:
                if hit is None:
                    continue
                title_key = _title_key(hit.title)
                if title_key in seen_titles:
                    continue
                # A title of punctuation alone says nothing about identity.
                if title_key:
                    seen_titles.add(title_key)
                merged.append(hit)
        return merged[:max_results]

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get(self, url: str, params: dict[str, Any]) -> str:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=_TIMEOUT,
                headers={"User-Agent": "shandu (+https://github.com/jolovicdev/shandu)"},
            )
        async with self._session.get(url, params=params) as response:
            if response.status != 429:
                response.raise_for_status()
                return await response.text()
        # One retry covers the short rate-limit bursts these APIs return
        # under parallel tasks.
        await asyncio.sleep(_RATE_LIMITED_RETRY_SECONDS)
        async with self._session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.text()

    async def _search_arxiv(
        self, query: str, terms: str, max_results: int
    ) -> list[SearchHit]:
        await self._arxiv_throttle.wait()
        feed = ElementTree.fromstring(
            await self._get(
                _ARXIV_URL,
                {
                    "search_query": f"all:{terms}",
                    "max_results": max_results,
                    "sortBy": "relevance",
                },
            )
        )
        hits: list[SearchHit] = []
        for entry in feed.findall(f"{_ATOM}entry"):
            url = _clean(entry.findtext(f"{_ATOM}id")).replace("http://", "https://", 1)
            title = _clean(entry.findtext(f"{_ATOM}title"))
            if not url or not title:
                continue
            hits.append(
                SearchHit(
                    query=query,
                    url=url,
                    title=title,
                    snippet=_clean(entry.findtext(f"{_ATOM}summary"))[:_SNIPPET_CHARS],
                    published_at=_clean(entry.findtext(f"{_ATOM}published"))[:10] or None,
                )
            )
        return hits

    async def _search_openalex(
        self, query: str, terms: str, max_results: int
    ) -> list[SearchHit]:
        params: dict[str, Any] = {
            "search": terms,
            "per_page": max_results,
            "select": "display_name,doi,publication_date,abstract_inverted_index,"
            "best_oa_location,primary_location",
        }
        api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        if api_key:
            params["api_key"] = api_key
        payload = json.loads(await self._get(_OPENALEX_URL, params))
        hits: list[SearchHit] = []
        for work in payload.get("results") or []:
            title = _clean(work.get("display_name"))
            url = _openalex_url(work)
            if not title or not url:
                continue
            hits.append(
                SearchHit(
                    query=query,
                    url=url,
                    title=title,
                    snippet=_openalex_abstract(work.get("abstract_inverted_index"))[
                        :_SNIPPET_CHARS
                    ],
                    published_at=work.get("publication_date") or None,
                )
            )
        return hits

    async def _search_pubmed(
        self, query: str, terms: str, max_results: int
    ) -> list[SearchHit]:
        common: dict[str, Any] = {"db": "pubmed", "retmode": "json", "tool": "shandu"}
        api_key = os.getenv("NCBI_API_KEY", "").strip()
        if api_key:
            common["api_key"] = api_key
        await self._pubmed_throttle.wait()
        found = json.loads(
            await self._get(
                _PUBMED_SEARCH_URL,
                {**common, "term": terms, "retmax": max_results, "sort": "relevance"},
            )
        )
        ids = [str(pmid) for pmid in (found.get("esearchresult") or {}).get("idlist") or []]
        if not ids:
            return []
        await self._pubmed_throttle.wait()
        summary = json.loads(
            await self._get(_PUBMED_SUMMARY_URL, {**common, "id": ",".join(ids)})
        )
        documents = summary.get("result") or {}
        hits: list[SearchHit] = []
        for pmid in ids:
            document = documents.get(pmid) or {}
            title = _clean(document.get("title"))
            if not title:
                continue
            published = str(document.get("sortpubdate") or "")[:10].replace("/", "-")
            # PubMed article pages answer plain fetches with a cookie
            # challenge; the free full text on PMC reads cleanly.
            pmc_id = next(
                (
                    str(article.get("value"))
                    for article in document.get("articleids") or []
                    if article.get("idtype") == "pmc"
                    and str(article.get("value", "")).startswith("PMC")
                ),
                None,
            )
            hits.append(
                SearchHit(
                    query=query,
                    url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
                    if pmc_id
                    else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    title=title,
                    snippet=", ".join(
                        part
                        for part in (
                            _clean(document.get("fulljournalname")),
                            _clean(document.get("pubdate")),
                        )
                        if part
                    ),
                    published_at=published or None,
                )
            )
        return hits
