from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from itertools import batched
from typing import Any, Literal

from blackgeorge import Job, Worker
from pydantic import BaseModel

from ..contracts import CitationEntry, CitationVerdict, EvidenceRecord
from ..interfaces import RuntimeExecutionLike
from ..prompts import auditor_instructions, auditor_job
from ..runtime.calls import ProviderRejectedError, structured_call
from ..runtime.costing import add_usage, collect_llm_usage
from ..runtime.roles import role_model
from .search_subagent import _report_reason

logger = logging.getLogger(__name__)

_UNITS_PER_BATCH = 12
_BATCH_CONCURRENCY = 4
# Enough for the extractor's whole evidence text plus its quotes: the writer
# saw that much, and a shorter cut marks true citations unsupported.
_SOURCE_CHARS = 12_000
_MARKER = re.compile(r"\[(\d+)\]")
# A sentence ends at terminal punctuation (optionally wrapped in closing quotes,
# bold/italic marks, or parentheses) followed by space and the start of a new
# sentence. Markers right after the punctuation belong to the sentence they close.
_SENTENCE_END = re.compile(
    r"(?<=[.!?])([*_\"'”’)\]]*(?:\s*\[\d+\])*[*_\"'”’)\]]*)\s+(?!\[\d+\])(?=[*_\"'“‘(\[\x00]*[A-Z0-9\x00])"
)
_ABBREVIATION = re.compile(
    r"(?:^|[\s(\[\"“‘])(?:dr|mr|mrs|ms|prof|jr|sr|st|vs|fig|no|eq|vol|al|e\.g|i\.e|u\.s|u\.k)\.$",
    re.IGNORECASE,
)
_CODE_SPAN = re.compile(r"(`+).+?\1")
_FENCE_OPEN = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_LIST_OR_QUOTE_PREFIX = re.compile(r"^\s*(?:[-*+>]|(?:\d+[.)]))\s+$")


def _mask_code_and_links(line: str) -> str:
    """Replace inline code spans and markdown links (with balanced parens) with NULs."""
    spans: list[tuple[int, int]] = [
        (match.start(), match.end()) for match in _CODE_SPAN.finditer(line)
    ]
    for match in re.finditer(r"\[[^\[\]\n]*\]\(", line):
        depth = 1
        index = match.end()
        while index < len(line) and depth > 0:
            if line[index] == "(":
                depth += 1
            elif line[index] == ")":
                depth -= 1
            index += 1
        if depth == 0:
            spans.append((match.start(), index))
    if not spans:
        return line
    spans.sort()
    chars = list(line)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = "\x00"
    return "".join(chars)


_Verdict = Literal["supported", "partial", "unsupported", "unchecked"]


class _MarkerVerdict(BaseModel):
    citation_id: int
    verdict: Literal["supported", "partial", "unsupported"]


class _UnitVerdict(BaseModel):
    unit_id: int
    markers: list[_MarkerVerdict]


class _AuditPayload(BaseModel):
    units: list[_UnitVerdict]


@dataclass(frozen=True, slots=True)
class _Unit:
    unit_id: int
    line: int
    start: int
    end: int
    text: str
    # (start, end, citation id) of each marker, relative to ``text``.
    markers: tuple[tuple[int, int, int], ...]

    @property
    def citation_ids(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(citation_id for _, _, citation_id in self.markers))


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    markdown: str
    verdicts: list[CitationVerdict]
    llm_usage: dict[str, Any] | None
    calls: int
    # Why each failed batch left its markers unchecked.
    failures: tuple[str, ...] = ()


def _cited_units(markdown: str) -> list[_Unit]:
    """Sentences, bullet lines, and table rows that carry citation markers.

    Fenced code is skipped, and inline code and links are masked, so digits
    in brackets there are never read as markers or sentence ends.
    """
    units: list[_Unit] = []
    fence_char = ""
    fence_len = 0
    for line_index, line in enumerate(markdown.split("\n")):
        stripped = line.strip()
        fence_match = _FENCE_OPEN.match(line)
        if fence_match:
            marker, info = fence_match.group(1), fence_match.group(2)
            if fence_char:
                if marker[0] == fence_char and len(marker) >= fence_len and not info.strip():
                    fence_char = ""
                    fence_len = 0
                continue
            if not (marker[0] == "`" and "`" in info):
                fence_char = marker[0]
                fence_len = len(marker)
                continue
        if fence_char or not stripped or stripped.startswith("#"):
            continue
        masked = _mask_code_and_links(line)
        if stripped.startswith("|"):
            spans = [(0, len(line))]
        else:
            spans = []
            cursor = 0
            for match in _SENTENCE_END.finditer(masked):
                before = line[cursor : match.start()]
                if _ABBREVIATION.search(before):
                    continue
                # If the next sentence starts with a masked code/link span,
                # confirm the unmasked span starts with an identifier/word
                # rather than punctuation.
                next_char = line[match.end()] if match.end() < len(line) else ""
                if masked[match.end()] == "\x00" and next_char not in ("`", "["):
                    continue
                spans.append((cursor, match.end(1)))
                cursor = match.end()
            spans.append((cursor, len(line)))
        for start, end in spans:
            markers = tuple(
                (match.start() - start, match.end() - start, int(match.group(1)))
                for match in _MARKER.finditer(masked, start, end)
            )
            if markers:
                units.append(
                    _Unit(len(units) + 1, line_index, start, end, line[start:end], markers)
                )
    return units


class CitationAuditor:
    """Checks each cited sentence against the evidence behind its markers.

    Unsupported markers are removed from the sentence that carries them;
    partial and supported markers stay. A batch whose call fails leaves its
    markers untouched and records them as unchecked.
    """

    def __init__(self, runtime: RuntimeExecutionLike) -> None:
        self._runtime = runtime

    async def audit(
        self,
        markdown: str,
        citations: list[CitationEntry],
        evidence: list[EvidenceRecord],
    ) -> AuditOutcome:
        body, separator, references = markdown.rpartition("\n## References\n")
        if not separator:
            body, references = references, ""
        units = _cited_units(body)
        by_id = {entry.citation_id: entry for entry in citations}
        records = {record.evidence_id: record for record in evidence}

        def source_text(citation_id: int) -> str:
            entry = by_id.get(citation_id)
            if entry is None:
                return ""
            matched = [records[eid] for eid in entry.evidence_ids if eid in records]
            if not matched:
                return ""
            per_record = max(1200, _SOURCE_CHARS // len(matched))
            parts: list[str] = []
            for record in matched:
                quotes = [f'"{claim.quote}"' for claim in record.claims]
                block = "\n".join([record.extracted_text, *quotes])
                if len(block) > per_record:
                    quote_block = "\n".join(quotes)
                    keep_quotes = quote_block[: per_record // 3] if quote_block else ""
                    room = per_record - len(keep_quotes) - (1 if keep_quotes else 0)
                    body_head = record.extracted_text[: max(0, room)]
                    block = (
                        f"{body_head}\n{keep_quotes}".strip()
                        if keep_quotes
                        else body_head
                    )
                parts.append(block)
            return "\n".join(parts)[:_SOURCE_CHARS]

        def source_dates(citation_id: int) -> list[str]:
            entry = by_id.get(citation_id)
            if entry is None:
                return []
            dates: set[str] = set()
            for eid in entry.evidence_ids:
                record = records.get(eid)
                if record is not None and record.published_at:
                    dates.add(record.published_at)
            return sorted(dates)

        semaphore = asyncio.Semaphore(_BATCH_CONCURRENCY)
        usage: dict[str, Any] | None = None
        calls = 0
        failures: list[str] = []

        async def judge(batch: tuple[_Unit, ...]) -> dict[int, dict[int, _Verdict]]:
            nonlocal usage, calls
            batch_ids = {unit.unit_id for unit in batch}
            cited = sorted({citation_id for unit in batch for citation_id in unit.citation_ids})
            payload = {
                "sources": [
                    {
                        "citation_id": citation_id,
                        "title": by_id[citation_id].title if citation_id in by_id else "",
                        "publisher": by_id[citation_id].publisher if citation_id in by_id else "",
                        "url": by_id[citation_id].url if citation_id in by_id else "",
                        "published_at": source_dates(citation_id),
                        "text": source_text(citation_id),
                    }
                    for citation_id in cited
                ],
                "units": [
                    {"unit_id": unit.unit_id, "text": unit.text, "citation_ids": unit.citation_ids}
                    for unit in batch
                ],
            }
            worker = Worker(
                name="CitationAuditor",
                model=role_model(self._runtime.settings, "auditor"),
                instructions=auditor_instructions(),
            )
            async with semaphore:
                try:
                    report = await structured_call(
                        self._runtime,
                        worker,
                        Job(input=auditor_job(payload), response_schema=_AuditPayload),
                    )
                except ProviderRejectedError:
                    raise
                except Exception as exc:
                    logger.warning("Citation audit batch failed", exc_info=True)
                    failures.append(f"{type(exc).__name__}: {exc}"[:300])
                    return {}
            calls += 1
            usage = add_usage(usage, collect_llm_usage(report))
            if report.status != "completed" or not isinstance(report.data, _AuditPayload):
                reason = _report_reason(report)
                logger.warning("Citation audit batch did not complete: %s", reason)
                failures.append(reason)
                return {}
            return {
                item.unit_id: {marker.citation_id: marker.verdict for marker in item.markers}
                for item in report.data.units
                if item.unit_id in batch_ids
            }

        batches = batched(units, _UNITS_PER_BATCH)
        judged: dict[int, dict[int, _Verdict]] = {}
        for result in await asyncio.gather(*(judge(batch) for batch in batches)):
            judged.update(result)

        lines = body.split("\n")
        verdicts: list[CitationVerdict] = []
        # Right to left, so earlier offsets on a line stay valid.
        for unit in sorted(units, key=lambda item: (item.line, item.start), reverse=True):
            unit_verdicts = judged.get(unit.unit_id, {})
            sentence = " ".join(_MARKER.sub("", unit.text).split())
            text = unit.text
            unit_list: list[CitationVerdict] = []
            for citation_id in unit.citation_ids:
                verdict: _Verdict = unit_verdicts.get(citation_id, "unchecked")
                entry = by_id.get(citation_id)
                unit_list.append(
                    CitationVerdict(
                        sentence=sentence, url=entry.url if entry else "", verdict=verdict
                    )
                )
            for start, end, citation_id in reversed(unit.markers):
                if unit_verdicts.get(citation_id) == "unsupported":
                    # Take the space before the marker along with it, unless a
                    # surviving marker follows or it separates a list/quote prefix.
                    after_prefix = bool(
                        unit.start == 0 and _LIST_OR_QUOTE_PREFIX.match(text[:start])
                    )
                    if after_prefix:
                        if end < len(text) and text[end] == " ":
                            end += 1
                    elif (
                        start > 0
                        and text[start - 1] == " "
                        and not (end < len(text) and text[end] == "[")
                    ):
                        start -= 1
                    text = text[:start] + text[end:]
            verdicts[0:0] = unit_list
            if text != unit.text:
                line = lines[unit.line]
                lines[unit.line] = line[: unit.start] + text + line[unit.end :]
        return AuditOutcome(
            markdown="\n".join(lines) + separator + references,
            verdicts=verdicts,
            llm_usage=usage,
            calls=calls,
            failures=tuple(failures),
        )
