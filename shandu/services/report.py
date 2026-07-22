from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

from ..contracts import CitationEntry, FinalReportDraft, ResearchRequest

# Table headers whose whole purpose is provenance; the reporter prompt bans
# them, and this renderer pass removes any that slip through.
_BANNED_PROVENANCE_HEADERS = {
    "source",
    "sources",
    "evidence source",
    "reference",
    "references",
    "ref",
    "citation",
    "citations",
    "provenance",
    "links",
    "key sources",
    "key proponents / sources",
}


@dataclass(frozen=True, slots=True)
class RenderedReport:
    markdown: str
    citations: list[CitationEntry]


class ReportService:
    def render(
        self,
        request: ResearchRequest,
        draft: FinalReportDraft,
        citations: list[CitationEntry],
    ) -> str:
        return self.render_result(request, draft, citations).markdown

    def render_result(
        self,
        request: ResearchRequest,
        draft: FinalReportDraft,
        citations: list[CitationEntry],
    ) -> RenderedReport:
        markdown = (
            draft.markdown.strip()
            if draft.markdown and draft.markdown.strip()
            else self._render_from_sections(request, draft)
        )
        markdown = self._strip_provenance_columns(markdown)
        markdown = self._strip_horizontal_rules(markdown)
        normalized = self._normalize_citation_markers(markdown, citations)
        normalized, normalized_citations = self._reindex_citation_numbers(
            normalized, citations
        )
        body = self._strip_references_section(normalized)
        body, normalized_citations = self._filter_and_reindex_used_citations(
            body,
            normalized_citations,
        )
        reference_lines = self._reference_lines(normalized_citations)
        if not reference_lines:
            return RenderedReport(markdown=body.strip(), citations=normalized_citations)
        rendered = "\n".join(
            [body.strip(), "", "## References", "", *reference_lines]
        ).strip()
        return RenderedReport(markdown=rendered, citations=normalized_citations)

    def _render_from_sections(
        self, request: ResearchRequest, draft: FinalReportDraft
    ) -> str:
        lines: list[str] = []
        lines.append(f"# {draft.title.strip()}")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(draft.executive_summary.strip())
        lines.append("")
        lines.append("## Research Configuration")
        lines.append("")
        lines.append(f"- Query: {request.query}")
        lines.append(f"- Max iterations: {request.max_iterations}")
        lines.append(f"- Parallelism: {request.parallelism}")
        lines.append(f"- Detail level: {request.detail_level}")
        lines.append("")
        for section in draft.sections:
            heading = section.heading.strip()
            content = section.content.strip()
            if not heading or not content:
                continue
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip()

    def _reference_lines(self, citations: list[CitationEntry]) -> list[str]:
        return [
            f"- **[{entry.citation_id}] {entry.publisher}** - "
            f'"{entry.title}". [Source]({entry.url}) '
            f"(accessed {entry.accessed_at})"
            for entry in sorted(citations, key=lambda item: item.citation_id)
        ]

    def _strip_provenance_columns(self, markdown: str) -> str:
        lines = markdown.splitlines()
        output: list[str] = []
        in_fence = False
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                output.append(lines[index])
                index += 1
                continue
            if in_fence or not self._is_table_row(lines[index]):
                output.append(lines[index])
                index += 1
                continue
            block_end = index
            while block_end < len(lines) and self._is_table_row(lines[block_end]):
                block_end += 1
            output.extend(self._rewrite_table(lines[index:block_end]))
            index = block_end
        return "\n".join(output)

    @staticmethod
    def _is_table_row(line: str) -> bool:
        stripped = line.strip()
        return len(stripped) > 1 and stripped.startswith("|") and stripped.endswith("|")

    @staticmethod
    def _split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def _rewrite_table(self, rows: list[str]) -> list[str]:
        if len(rows) < 2:
            return rows
        header = self._split_row(rows[0])
        separator = self._split_row(rows[1])
        if len(separator) != len(header) or not all(
            re.fullmatch(r":?-{2,}:?", cell) for cell in separator
        ):
            return rows
        banned = {
            index
            for index, cell in enumerate(header)
            if cell.strip("*").strip().casefold() in _BANNED_PROVENANCE_HEADERS
        }
        if not banned or len(banned) >= len(header):
            return rows
        body = [self._split_row(row) for row in rows[2:]]
        if any(len(cells) != len(header) for cells in body):
            return rows

        marker_pattern = re.compile(r"\[[A-Za-z0-9_-]{1,64}\]")

        def join(cells: list[str]) -> str:
            return "| " + " | ".join(cells) + " |"

        rewritten = [
            join([cell for index, cell in enumerate(header) if index not in banned]),
            join([cell for index, cell in enumerate(separator) if index not in banned]),
        ]
        for cells in body:
            kept = [cell for index, cell in enumerate(cells) if index not in banned]
            markers = [
                marker
                for index in sorted(banned)
                for marker in marker_pattern.findall(cells[index])
            ]
            fresh = [
                marker
                for marker in dict.fromkeys(markers)
                if all(marker not in cell for cell in kept)
            ]
            if kept and fresh:
                kept[0] = f"{kept[0]} {' '.join(fresh)}".strip()
            rewritten.append(join(kept))
        return rewritten

    def _strip_references_section(self, markdown: str) -> str:
        heading_pattern = re.compile(
            r"^\s{0,3}(?:#{1,6}\s*)?(?:key\s+)?"
            r"(?:references?|sources?|bibliography|citations?)\s*:?\s*$",
            re.IGNORECASE,
        )
        lines = markdown.splitlines()
        for index, line in enumerate(lines):
            if heading_pattern.match(line) and self._looks_like_reference_block(
                lines[index + 1 :]
            ):
                return "\n".join(lines[:index]).strip()
        return markdown.strip()

    def _looks_like_reference_block(self, lines: list[str]) -> bool:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^\s{0,3}#{1,6}\s+\S+", line):
                return False
            return self._looks_like_reference_entry(stripped)
        return True

    def _looks_like_reference_entry(self, line: str) -> bool:
        return bool(
            re.match(r"^(?:[-*+]\s*)?(?:\[\d+\]|\d+[\.)])\s+", line)
            or re.search(r"https?://|www\.", line, re.IGNORECASE)
            or re.search(r"\[[^\]]+\]\(https?://", line, re.IGNORECASE)
        )

    @staticmethod
    def _strip_horizontal_rules(markdown: str) -> str:
        # Only blank-line-surrounded rules; a setext heading underline
        # ("Title\n---") has text directly above and must survive.
        return re.sub(
            r"(\n[ \t]*\n)(?:-{3,}|\*{3,}|_{3,})[ \t]*\n",
            r"\1",
            markdown,
        )

    def _normalize_citation_markers(
        self,
        markdown: str,
        citations: list[CitationEntry],
    ) -> str:
        # Split grouped markers ("[1, 2]") into single ones so each number is
        # validated below instead of bypassing the marker pipeline entirely.
        markdown = re.sub(
            r"\[(\d+(?:\s*,\s*\d+)+)\]",
            lambda match: "".join(
                f"[{int(token)}]" for token in re.split(r"\s*,\s*", match.group(1))
            ),
            markdown,
        )
        valid_numbers = {str(entry.citation_id) for entry in citations}
        evidence_to_number: dict[str, str] = {}
        for entry in citations:
            number = str(entry.citation_id)
            for evidence_id in entry.evidence_ids:
                if evidence_id:
                    evidence_to_number[evidence_id] = number

        marker_pattern = re.compile(r"\[([A-Za-z0-9_-]{1,64})\]")

        def replace(match: re.Match[str]) -> str:
            token = match.group(1).strip()
            if not token:
                return ""
            if token.isdigit():
                if token in valid_numbers:
                    return f"[{int(token)}]"
                return ""
            mapped = evidence_to_number.get(token)
            if mapped:
                return f"[{mapped}]"
            if re.fullmatch(r"[0-9a-fA-F]{32}", token):
                return ""
            return match.group(0)

        text = marker_pattern.sub(replace, markdown)
        text = re.sub(r"(\[(\d+)\])(?:\s*\[\2\])+", r"[\2]", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _reindex_citation_numbers(
        self,
        markdown: str,
        citations: list[CitationEntry],
    ) -> tuple[str, list[CitationEntry]]:
        if not citations:
            return markdown, []

        ordered = sorted(citations, key=lambda item: item.citation_id)
        id_map = {
            str(entry.citation_id): index
            for index, entry in enumerate(ordered, start=1)
        }

        pattern = re.compile(r"\[(\d+)\]")

        def replace(match: re.Match[str]) -> str:
            token = match.group(1)
            mapped = id_map.get(token)
            if mapped is None:
                return match.group(0)
            return f"[{mapped}]"

        normalized_markdown = pattern.sub(replace, markdown)
        normalized_markdown = re.sub(
            r"(\[(\d+)\])(?:\s*\[\2\])+", r"[\2]", normalized_markdown
        )

        normalized_citations: list[CitationEntry] = []
        for index, entry in enumerate(ordered, start=1):
            normalized_citations.append(entry.model_copy(update={"citation_id": index}))
        return normalized_markdown, normalized_citations

    def _filter_and_reindex_used_citations(
        self,
        body: str,
        citations: list[CitationEntry],
    ) -> tuple[str, list[CitationEntry]]:
        marker_pattern = re.compile(r"\[(\d+)\]")
        used_markers = [int(token) for token in marker_pattern.findall(body)]
        if not used_markers or not citations:
            return body, citations

        ordered_used = list(OrderedDict.fromkeys(used_markers))
        citation_by_id = {entry.citation_id: entry for entry in citations}
        kept_entries: list[CitationEntry] = []
        id_map: dict[int, int] = {}
        for new_id, old_id in enumerate(ordered_used, start=1):
            entry = citation_by_id.get(old_id)
            if entry is None:
                continue
            kept_entries.append(entry.model_copy(update={"citation_id": new_id}))
            id_map[old_id] = new_id

        def replace(match: re.Match[str]) -> str:
            old_id = int(match.group(1))
            mapped = id_map.get(old_id)
            if mapped is None:
                return ""
            return f"[{mapped}]"

        normalized_body = marker_pattern.sub(replace, body)
        normalized_body = re.sub(r"(\[(\d+)\])(?:\s*\[\2\])+", r"[\2]", normalized_body)
        normalized_body = re.sub(r"[ \t]+\n", "\n", normalized_body)
        normalized_body = re.sub(r"\n{3,}", "\n\n", normalized_body).strip()
        return normalized_body, kept_entries
