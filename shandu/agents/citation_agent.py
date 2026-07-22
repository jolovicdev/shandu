from __future__ import annotations

import json
from datetime import date
from urllib.parse import urlparse

from blackgeorge import Job, Worker
from pydantic import BaseModel, Field

from ..contracts import CitationEntry, EvidenceRecord
from ..interfaces import RuntimeExecutionLike
from ..prompts import citation_instructions, citation_job

# Evidence below this credibility stays out of the ledger so weak pages can
# inform caveats without earning a reference entry. Snippet-only fallbacks
# (0.20) and penalized blogs/social/marketing pages land under it; a clean
# personal blog (0.36) or worst-case journalism (0.37) stays citable. If
# nothing clears the bar, the full corpus is used so reports keep citations.
_MIN_CITABLE_CREDIBILITY = 0.35

# Same-work dedup only trusts titles long enough to be distinctive; short
# titles ("10-K", "FAQ") collide across unrelated pages on the same site.
_MIN_MERGE_TITLE_LEN = 12


class _CitationCandidate(BaseModel):
    evidence_ids: list[str] = Field(default_factory=list)
    url: str
    title: str
    publisher: str


class _CitationBundle(BaseModel):
    citations: list[_CitationCandidate] = Field(default_factory=list)


class CitationAgent:
    def __init__(self, runtime: RuntimeExecutionLike) -> None:
        self._runtime = runtime

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

        evidence_json = json.dumps(
            [self._project(item) for item in citable], ensure_ascii=False
        )
        worker = Worker(
            name="CitationSubagent",
            model=self._runtime.settings.model,
            instructions=citation_instructions(),
        )
        job = Job(
            input=citation_job(query, evidence_json),
            response_schema=_CitationBundle,
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            if report.status == "completed" and isinstance(report.data, _CitationBundle):
                normalized = self._normalize(report.data.citations, citable)
                if normalized:
                    return normalized
        except Exception:
            pass

        return self._fallback(citable)

    @staticmethod
    def _project(item: EvidenceRecord) -> dict[str, str]:
        return {
            "evidence_id": item.evidence_id,
            "requested_url": item.requested_url,
            "final_url": item.final_url or "",
            "title": item.title,
            "domain": item.domain or "",
            "snippet": (item.snippet or "")[:280],
        }

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

    def _normalize(
        self,
        candidates: list[_CitationCandidate],
        evidence: list[EvidenceRecord],
    ) -> list[CitationEntry]:
        if not candidates:
            return []
        by_url: dict[str, set[str]] = {}
        for item in evidence:
            by_url.setdefault(item.requested_url, set()).add(item.evidence_id)

        normalized: list[CitationEntry] = []
        seen_urls: set[str] = set()
        merged: dict[tuple[str, str], CitationEntry] = {}
        accessed = date.today().isoformat()
        for candidate in candidates:
            url = candidate.url.strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            evidence_ids = list(by_url.get(url, set())) or candidate.evidence_ids
            publisher = candidate.publisher.strip() or urlparse(url).netloc
            title = self._sanitize_title(candidate.title, publisher or "Untitled")
            # Fallback titles equal the publisher; merging on them would fuse
            # unrelated pages from the same site.
            key = self._merge_key(url, title) if title != publisher else None
            if key is not None and key in merged:
                entry = merged[key]
                entry.evidence_ids = sorted(set(entry.evidence_ids) | set(evidence_ids))
                continue
            entry = CitationEntry(
                citation_id=len(normalized) + 1,
                evidence_ids=sorted(set(evidence_ids)),
                url=url,
                title=title,
                publisher=publisher,
                accessed_at=accessed,
            )
            if key is not None:
                merged[key] = entry
            normalized.append(entry)
        return normalized

    def _fallback(self, evidence: list[EvidenceRecord]) -> list[CitationEntry]:
        grouped: dict[str, list[EvidenceRecord]] = {}
        for item in evidence:
            grouped.setdefault(item.requested_url, []).append(item)

        citations: list[CitationEntry] = []
        merged: dict[tuple[str, str], CitationEntry] = {}
        accessed = date.today().isoformat()
        for url, items in grouped.items():
            first = items[0]
            publisher = urlparse(url).netloc or "unknown"
            title = self._sanitize_title(first.title, publisher)
            evidence_ids = sorted({entry.evidence_id for entry in items})
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
