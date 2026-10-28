from __future__ import annotations

import re
from datetime import date
from urllib.parse import unquote, urlparse

from ..contracts import CitationEntry, EvidenceRecord

# Evidence below this credibility stays out of the ledger so weak pages can
# inform caveats without earning a reference entry. Snippet-only fallbacks
# (0.20) and penalized blogs/social/marketing pages land under it; a clean
# personal blog (0.36) or worst-case journalism (0.37) stays citable. If
# nothing clears the bar, the full corpus is used so reports keep citations.
_MIN_CITABLE_CREDIBILITY = 0.35
# The extractor rates a page it found off-topic or unusable below this, so a
# credible site that did not answer the task earns no reference either.
_MIN_CITABLE_CONFIDENCE = 0.35

# Same-work dedup only trusts titles long enough to be distinctive; short
# titles ("10-K", "FAQ") collide across unrelated pages on the same site.
_MIN_MERGE_TITLE_LEN = 12
_MARKUP = re.compile(r"<[^>]+>")
# Documentation themes end a heading with its anchor ("Quantized KV Cache¶").
# A "#" after a capital is part of a name such as C#.
_HEADING_ANCHOR = re.compile(r"(?:\s*\u00b6|(?<=[a-z])#)$")
# Copies of one paper share a DOI or an arXiv id, and a publisher's PDF sits
# beside its HTML page with ".pdf" added.
_DOI = re.compile(r"10\.\d{4,9}/[^\s?#]+", re.IGNORECASE)
_DOI_VIEW_SUFFIX = re.compile(r"(?:\.pdf|/(?:full|pdf|epdf|abstract|fulltext))+$")
_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}")
# The same long title on two sites names one work, such as a paper on its
# journal's site and on PubMed Central.
_MIN_CROSS_SITE_TITLE_LEN = 40
# Page titles often end with their site's name: "Incident report | CrowdStrike".
_SITE_SUFFIX = re.compile(r"^(?P<head>.*\S)\s+[|\-–—]\s+(?P<tail>[^|\-–—]+)$")
# Words a site or journal name may carry in lower case.
_NAME_CONNECTORS = frozenset({"&", "a", "and", "at", "de", "du", "for", "in", "la", "of", "on", "the", "und"})
# A country's second level, as in bbc.co.uk, which is not the site's brand.
_SECOND_LEVELS = frozenset({"ac", "co", "com", "edu", "gov", "net", "org"})


def _reads_as_a_name(text: str) -> bool:
    """Whether every word is capitalized or carries a capital ("bioRxiv"), is a
    domain, or is a connector, as in a site or journal name."""
    return all(
        any(char.isupper() or char.isdigit() for char in word)
        or "." in word
        or word in _NAME_CONNECTORS
        for word in text.split()
    )


def _name_other_works(ids: set[tuple[str, str]], seen: set[tuple[str, str]]) -> bool:
    """Whether two identifiers of one kind differ, which names two works
    whatever their titles. A DOI and an arXiv id can be one paper's journal
    and preprint versions."""
    return any(key not in seen and any(kind == key[0] for kind, _ in seen) for key in ids)


class CitationAgent:
    async def build_citations(
        self,
        query: str,
        evidence: list[EvidenceRecord],
    ) -> list[CitationEntry]:
        if not evidence:
            return []

        citable = [
            item
            for item in evidence
            if (item.credibility_score is None or item.credibility_score >= _MIN_CITABLE_CREDIBILITY)
            and item.confidence >= _MIN_CITABLE_CONFIDENCE
        ]
        if not citable:
            citable = evidence

        return self._build_ledger(citable)

    @staticmethod
    def _sanitize_title(title: str, fallback: str) -> str:
        # Publisher pages leave markup and soft hyphens inside titles.
        cleaned = " ".join(_MARKUP.sub("", title).replace("\u00ad", "").split())
        cleaned = _HEADING_ANCHOR.sub("", cleaned)
        if len(cleaned) < 3 or cleaned.lower().startswith(
            ("http://", "https://", "www.")
        ):
            return fallback
        # Some pages ship a whole lede as <title>; cap so the reference list
        # stays scannable.
        if len(cleaned) > 160:
            cleaned = cleaned[:157].rstrip() + "..."
        return cleaned

    @staticmethod
    def _same_work_keys(url: str, title: str, publisher: str) -> list[tuple[str, str]]:
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        path = unquote(parsed.path).rstrip("/")
        keys: list[tuple[str, str]] = []
        doi = _DOI.search(path)
        arxiv = _ARXIV_ID.search(path) if host.endswith("arxiv.org") else None
        if doi:
            value = _DOI_VIEW_SUFFIX.sub("", doi.group(0).lower())
            if value.startswith("10.48550/arxiv."):
                keys.append(("arxiv", value.removeprefix("10.48550/arxiv.")))
            else:
                keys.append(("doi", value))
        elif arxiv:
            keys.append(("arxiv", arxiv.group()))
        elif not parsed.query:
            keys.append(("page", host + path.removesuffix(".pdf")))
        normalized = " ".join(re.sub(r"[\W_]+", " ", title.casefold()).split())
        # Fallback titles equal the publisher; merging on them would fuse
        # unrelated pages from the same site.
        if host and title != publisher and len(normalized) >= _MIN_MERGE_TITLE_LEN:
            keys.append((host, normalized))
            if len(normalized) >= _MIN_CROSS_SITE_TITLE_LEN:
                keys.append(("title", normalized))
        return keys

    @staticmethod
    def _without_site_suffix(title: str, publisher: str, host: str) -> str:
        """Drop a trailing site or journal name that starts with the publisher's name.

        "... - Nature Medicine" from nature.com loses its suffix, which also
        lets it match the same paper's title on PubMed Central.
        """
        match = _SITE_SUFFIX.match(title)
        if match is None or len(match.group("head").split()) < 3:
            return title
        labels = host.split(".")
        if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVELS:
            labels = labels[:-1]
        brand = labels[-2] if len(labels) >= 2 else host
        names = {re.sub(r"[\W_]+", "", name.casefold()) for name in (publisher, brand)}
        tail = re.sub(r"[\W_]+", "", match.group("tail").casefold())
        named = _reads_as_a_name(match.group("tail")) and any(
            len(name) >= 3 and tail.startswith(name) for name in names
        )
        return match.group("head") if named else title

    @staticmethod
    def _title_from_url(url: str) -> str:
        """A document served without a title is named after its file."""
        stem = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]).rsplit(".", 1)[0]
        words = " ".join(re.sub(r"[-_+]+", " ", stem).split())
        return words[:1].upper() + words[1:] if sum(char.isalpha() for char in words) >= 8 else ""

    def _build_ledger(self, evidence: list[EvidenceRecord]) -> list[CitationEntry]:
        grouped: dict[str, list[EvidenceRecord]] = {}
        for item in evidence:
            grouped.setdefault(item.requested_url, []).append(item)

        citations: list[CitationEntry] = []
        merged: dict[tuple[str, str], CitationEntry] = {}
        # The DOIs and arXiv ids each reference has been seen under.
        work_ids: dict[int, set[tuple[str, str]]] = {}
        accessed = date.today().isoformat()
        for url, items in grouped.items():
            first = items[0]
            host = urlparse(url).netloc.removeprefix("www.")
            publisher = first.site_name or host or "unknown"
            title = self._sanitize_title(first.title, "")
            from_url = self._title_from_url(url)
            # A one-word title is usually the site's section ("Solutions")
            # when the address names the article itself. Scripts that set
            # words without spaces, as Japanese does, put a whole title in one.
            if (
                not title
                or title.casefold() == host.casefold()
                or (len(title.split()) == 1 and title.isascii() and len(from_url.split()) >= 3)
            ):
                title = from_url or publisher
            title = self._without_site_suffix(title, publisher, host)
            evidence_ids = sorted({entry.evidence_id for entry in items})
            keys = self._same_work_keys(url, title, publisher)
            ids = {key for key in keys if key[0] in ("doi", "arxiv")}
            entry = next(
                (
                    merged[key]
                    for key in keys
                    if key in merged and not _name_other_works(ids, work_ids[merged[key].citation_id])
                ),
                None,
            )
            if entry is None:
                entry = CitationEntry(
                    citation_id=len(citations) + 1,
                    evidence_ids=evidence_ids,
                    url=url,
                    title=title,
                    publisher=publisher,
                    accessed_at=accessed,
                )
                citations.append(entry)
            else:
                entry.evidence_ids = sorted(set(entry.evidence_ids) | set(evidence_ids))
            work_ids.setdefault(entry.citation_id, set()).update(ids)
            for key in keys:
                merged.setdefault(key, entry)
        return citations
