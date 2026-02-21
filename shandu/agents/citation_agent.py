from __future__ import annotations

import json
from datetime import date
from urllib.parse import urlparse

from blackgeorge import Job, Worker
from pydantic import BaseModel, Field

from ..contracts import CitationEntry, EvidenceRecord
from ..interfaces import RuntimeExecutionLike


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

        worker = Worker(
            name="CitationSubagent",
            model=self._runtime.settings.model,
            instructions=(
                "You are CitationSubagent. "
                "Generate a clean bibliography from evidence without inventing fields. "
                "Deduplicate sources by URL, preserve evidence linkage, and normalize publisher/title text. "
                "If metadata is weak, use safe fallbacks from URL/domain."
            ),
        )
        job = Job(
            input=(
                "Build citation entries from evidence as structured output.\n"
                "Requirements:\n"
                "- Return one citation candidate per unique URL whenever possible.\n"
                "- evidence_ids must reference provided evidence only.\n"
                "- Do not invent URLs, titles, publishers, or evidence IDs.\n"
                f"Query: {query}\n"
                f"Evidence JSON:\n{json.dumps([item.model_dump(mode='json') for item in evidence], ensure_ascii=False)}"
            ),
            response_schema=_CitationBundle,
        )
        try:
            report = await self._runtime.desk.arun(worker, job)
            if report.status == "completed" and isinstance(report.data, _CitationBundle):
                normalized = self._normalize(report.data.citations, evidence)
                if normalized:
                    return normalized
        except Exception:
            pass

        return self._fallback(evidence)

    def _normalize(
        self,
        candidates: list[_CitationCandidate],
        evidence: list[EvidenceRecord],
    ) -> list[CitationEntry]:
        if not candidates:
            return []
        by_url: dict[str, set[str]] = {}
        for item in evidence:
            by_url.setdefault(item.url, set()).add(item.evidence_id)

        normalized: list[CitationEntry] = []
        seen: set[str] = set()
        accessed = date.today().isoformat()
        for idx, candidate in enumerate(candidates, start=1):
            url = candidate.url.strip()
            if not url or url in seen:
                continue
            seen.add(url)
            evidence_ids = list(by_url.get(url, set())) or candidate.evidence_ids
            publisher = candidate.publisher.strip() or urlparse(url).netloc
            title = candidate.title.strip() or "Untitled"
            normalized.append(
                CitationEntry(
                    citation_id=idx,
                    evidence_ids=sorted(set(evidence_ids)),
                    url=url,
                    title=title,
                    publisher=publisher,
                    accessed_at=accessed,
                )
            )
        return normalized

    def _fallback(self, evidence: list[EvidenceRecord]) -> list[CitationEntry]:
        grouped: dict[str, list[EvidenceRecord]] = {}
        for item in evidence:
            grouped.setdefault(item.url, []).append(item)

        citations: list[CitationEntry] = []
        accessed = date.today().isoformat()
        for idx, (url, items) in enumerate(grouped.items(), start=1):
            first = items[0]
            citations.append(
                CitationEntry(
                    citation_id=idx,
                    evidence_ids=sorted({entry.evidence_id for entry in items}),
                    url=url,
                    title=first.title or "Untitled",
                    publisher=urlparse(url).netloc or "unknown",
                    accessed_at=accessed,
                )
            )
        return citations
