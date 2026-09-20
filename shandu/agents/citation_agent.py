from __future__ import annotations

from datetime import date
from urllib.parse import urlparse

from ..contracts import CitationEntry, EvidenceRecord

# Evidence below this credibility stays out of the ledger so weak pages can
# inform caveats without earning a reference entry. Snippet-only fallbacks
# (0.20) and penalized blogs/social/marketing pages land under it; a clean
# personal blog (0.36) or worst-case journalism (0.37) stays citable. If
# nothing clears the bar, the full corpus is used so reports keep citations.
_MIN_CITABLE_CREDIBILITY = 0.35

# Same-work dedup only trusts titles long enough to be distinctive; short
# titles ("10-K", "FAQ") collide across unrelated pages on the same site.
_MIN_MERGE_TITLE_LEN = 12


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
            if item.credibility_score is None
            or item.credibility_score >= _MIN_CITABLE_CREDIBILITY
        ]
        if not citable:
            citable = evidence

        return self._build_ledger(citable)

    @staticmethod
    def _sanitize_title(title: str, fallback: str) -> str:
        cleaned = " ".join(title.split())
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
    def _merge_key(url: str, title: str) -> tuple[str, str] | None:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        normalized = " ".join(title.split()).casefold()
        if not host or len(normalized) < _MIN_MERGE_TITLE_LEN:
            return None
        return host, normalized

    def _build_ledger(self, evidence: list[EvidenceRecord]) -> list[CitationEntry]:
        grouped: dict[str, list[EvidenceRecord]] = {}
        for item in evidence:
            grouped.setdefault(item.requested_url, []).append(item)

        citations: list[CitationEntry] = []
        merged: dict[tuple[str, str], CitationEntry] = {}
        accessed = date.today().isoformat()
        for url, items in grouped.items():
            first = items[0]
            host = urlparse(url).netloc.removeprefix("www.")
            publisher = first.site_name or host or "unknown"
            title = self._sanitize_title(first.title, publisher)
            evidence_ids = sorted({entry.evidence_id for entry in items})
            # Fallback titles equal the publisher; merging on them would fuse
            # unrelated pages from the same site.
            key = self._merge_key(url, title) if title != publisher else None
            if key is not None and key in merged:
                entry = merged[key]
                entry.evidence_ids = sorted(set(entry.evidence_ids) | set(evidence_ids))
                continue
            entry = CitationEntry(
                citation_id=len(citations) + 1,
                evidence_ids=evidence_ids,
                url=url,
                title=title,
                publisher=publisher,
                accessed_at=accessed,
            )
            if key is not None:
                merged[key] = entry
            citations.append(entry)
        return citations
