from __future__ import annotations

import re

from ..contracts import CitationEntry, FinalReportDraft, ResearchRequest


class ReportService:
    def render(
        self,
        request: ResearchRequest,
        draft: FinalReportDraft,
        citations: list[CitationEntry],
    ) -> str:
        markdown = (
            draft.markdown.strip()
            if draft.markdown and draft.markdown.strip()
            else self._render_from_sections(request, draft)
        )
        body = self._strip_references_section(
            self._normalize_citation_markers(markdown, citations)
        )
        reference_lines = self._reference_lines(citations)
        if not reference_lines:
            return body.strip()
        return "\n".join([body.strip(), "", "## References", "", *reference_lines]).strip()

    def _render_from_sections(self, request: ResearchRequest, draft: FinalReportDraft) -> str:
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
            f"[{entry.citation_id}] {entry.publisher}. \"{entry.title}\". {entry.url} (accessed {entry.accessed_at})"
            for entry in sorted(citations, key=lambda item: item.citation_id)
        ]

    def _strip_references_section(self, markdown: str) -> str:
        lines: list[str] = []
        for line in markdown.splitlines():
            if line.strip().lower().startswith("## references"):
                break
            lines.append(line)
        return "\n".join(lines).strip()

    def _normalize_citation_markers(
        self,
        markdown: str,
        citations: list[CitationEntry],
    ) -> str:
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
                return match.group(0)
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
