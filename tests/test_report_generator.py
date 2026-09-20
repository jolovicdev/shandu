from __future__ import annotations

import pytest

from shandu.contracts import (
    CitationEntry,
    FinalReportDraft,
    ReportSection,
    ResearchRequest,
)
from shandu.services.report import ReportService


def test_report_service_renders_expected_sections() -> None:
    service = ReportService()
    request = ResearchRequest(query="Test query")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[ReportSection(heading="Analysis", content="Body")],
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com",
            title="Example",
            publisher="example.com",
            accessed_at="2026-02-21",
        )
    ]

    rendered = service.render(request, draft, citations)
    assert "# Report" in rendered
    assert "## Analysis" in rendered
    assert "## References" in rendered


def test_report_service_respects_prebuilt_markdown() -> None:
    service = ReportService()
    request = ResearchRequest(query="Test query")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown="# Report\n\n## Executive Summary\n\nText",
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com",
            title="Example",
            publisher="example.com",
            accessed_at="2026-02-21",
        )
    ]

    rendered = service.render(request, draft, citations)
    assert rendered.startswith("# Report")
    assert "## References" in rendered


def test_report_service_normalizes_evidence_id_markers_to_numeric_citations() -> None:
    service = ReportService()
    request = ResearchRequest(query="Predict top jobs")
    evidence_id = "e1-alpha"
    orphan_md5 = "a93a4e1b65ff42009c95f52329c5179e"
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "## Executive Summary\n\n"
            f"Energy demand is rising [{evidence_id}][{evidence_id}] and market salaries are rising [1][99] despite [{orphan_md5}].\n\n"
            "## References\n\n"
            f"[{evidence_id}] random"
        ),
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=[evidence_id],
            url="https://energy.example/analysis",
            title="Energy Analysis",
            publisher="energy.example",
            accessed_at="2026-02-21",
        )
    ]

    rendered = service.render(request, draft, citations)

    assert f"[{evidence_id}]" not in rendered
    assert "rising [1]" in rendered
    assert rendered.count("[1]") >= 2
    assert "[99]" not in rendered
    assert orphan_md5 not in rendered
    assert "**[1] energy.example**" in rendered
    assert "[Source](https://energy.example/analysis)" in rendered


def test_report_service_reindexes_citation_numbers_without_gaps() -> None:
    service = ReportService()
    request = ResearchRequest(query="Compare X vs Y")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "## Executive Summary\n\n"
            "A is strong [1]. B is strong [3]. C is emerging [4].\n"
        ),
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com/a",
            title="A",
            publisher="example.com",
            accessed_at="2026-02-21",
        ),
        CitationEntry(
            citation_id=3,
            evidence_ids=["e3"],
            url="https://example.com/b",
            title="B",
            publisher="example.com",
            accessed_at="2026-02-21",
        ),
        CitationEntry(
            citation_id=4,
            evidence_ids=["e4"],
            url="https://example.com/c",
            title="C",
            publisher="example.com",
            accessed_at="2026-02-21",
        ),
    ]

    rendered = service.render(request, draft, citations)

    assert "[4]" not in rendered
    assert "A is strong [1]. B is strong [2]. C is emerging [3]." in rendered
    assert '**[1] example.com** - "A". [Source](https://example.com/a)' in rendered
    assert '**[2] example.com** - "B". [Source](https://example.com/b)' in rendered
    assert '**[3] example.com** - "C". [Source](https://example.com/c)' in rendered


def test_report_service_strips_plain_reference_sections_and_returns_used_citations() -> (
    None
):
    service = ReportService()
    request = ResearchRequest(query="Compare X vs Y")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "Finding A is supported [2].\n\n"
            "References\n\n"
            "[1] model-invented reference that should be stripped"
        ),
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com/unused",
            title="Unused",
            publisher="example.com",
            accessed_at="2026-02-21",
        ),
        CitationEntry(
            citation_id=2,
            evidence_ids=["e2"],
            url="https://example.com/used",
            title="Used",
            publisher="example.com",
            accessed_at="2026-02-21",
        ),
    ]

    result = service.render_result(request, draft, citations)

    assert "model-invented reference" not in result.markdown
    assert (
        '- **[1] example.com** - "Used". [Source](https://example.com/used)'
        in result.markdown
    )
    assert len(result.citations) == 1
    assert result.citations[0].title == "Used"


def test_report_service_preserves_narrative_sources_section() -> None:
    service = ReportService()
    request = ResearchRequest(query="Evaluate source quality")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "## Sources\n\n"
            "Source quality is mixed, with stronger peer-reviewed evidence "
            "for one claim [1].\n\n"
            "## Implications\n\n"
            "This section should not be truncated [1]."
        ),
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com/source-quality",
            title="Source Quality",
            publisher="example.com",
            accessed_at="2026-02-21",
        )
    ]

    rendered = service.render(request, draft, citations)

    assert "## Sources" in rendered
    assert "Source quality is mixed" in rendered
    assert "## Implications" in rendered
    assert "This section should not be truncated" in rendered


@pytest.mark.parametrize(
    "markdown,kept,dropped",
    [
        (
            "| Model | FLOP | Source |\n"
            "|---|---|---|\n"
            "| GPT-5 | 5e25 [2] | Epoch AI [2] |\n"
            "| V3.2 | n/a | researchaudio.io [3] |",
            ["| GPT-5 | 5e25 [2] |", "| V3.2 [3] | n/a |"],
            ["Source", "Epoch AI"],
        ),
        (
            "| Claim | **Sources** |\n"
            "| :--- | ---: |\n"
            "| X is true [1] | Publisher [1] |",
            ["| X is true [1] |"],
            ["Sources"],
        ),
    ],
)
def test_strip_provenance_column_drops_source_column(
    markdown, kept, dropped
) -> None:
    result = ReportService()._strip_provenance_columns(markdown)

    for text in kept:
        assert text in result
    for text in dropped:
        assert text not in result


def test_strip_provenance_leaves_malformed_and_clean_tables() -> None:
    service = ReportService()
    malformed = "| A | Source |\n|---|---|\n| only one cell |"
    clean = "| A | B |\n|---|---|\n| 1 | 2 |"

    assert service._strip_provenance_columns(malformed) == malformed
    assert service._strip_provenance_columns(clean) == clean


def test_report_service_strips_source_column_end_to_end() -> None:
    service = ReportService()
    request = ResearchRequest(query="Compare X vs Y")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "Claim [1].\n\n"
            "| Model | Value | Source |\n"
            "|---|---|---|\n"
            "| A | 10 | Epoch [1] |\n"
        ),
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com/a",
            title="A",
            publisher="example.com",
            accessed_at="2026-02-21",
        )
    ]

    rendered = service.render(request, draft, citations)

    assert "| Source |" not in rendered
    assert "| A [1] | 10 |" in rendered


def test_report_service_splits_valid_groups_and_leaves_invalid_literals() -> None:
    service = ReportService()
    request = ResearchRequest(query="Compare X vs Y")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown="# Report\n\nStrong claim [1, 2]. Weak claim [15, 16, 17].\n",
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com/a",
            title="A",
            publisher="example.com",
            accessed_at="2026-02-21",
        ),
        CitationEntry(
            citation_id=2,
            evidence_ids=["e2"],
            url="https://example.com/b",
            title="B",
            publisher="example.com",
            accessed_at="2026-02-21",
        ),
    ]

    rendered = service.render(request, draft, citations)

    assert "Strong claim [1][2]." in rendered
    assert "Weak claim [15, 16, 17]." in rendered


def test_report_service_strips_horizontal_rules_but_keeps_setext() -> None:
    service = ReportService()
    request = ResearchRequest(query="Compare X vs Y")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\nIntro [1].\n\n---\n\n## Next\n\nSetext Heading\n---\n\nMore [1].\n"
        ),
    )
    citations = [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com/a",
            title="A",
            publisher="example.com",
            accessed_at="2026-02-21",
        )
    ]

    rendered = service.render(request, draft, citations)

    assert "\n\n---\n\n" not in rendered
    assert "Setext Heading\n---" in rendered


def _single_citation() -> list[CitationEntry]:
    return [
        CitationEntry(
            citation_id=1,
            evidence_ids=["e1"],
            url="https://example.com/a",
            title="A",
            publisher="example.com",
            accessed_at="2026-02-21",
        )
    ]


@pytest.mark.parametrize(
    "code",
    [
        "```python\nx = arr[0]\narr = [1, 2, 3]\n```",
        "~~~\nx = arr[0]\narr = [1, 2]\n~~~",
    ],
)
def test_marker_passes_leave_fenced_code_byte_for_byte(code) -> None:
    service = ReportService()
    request = ResearchRequest(query="q")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=f"# Report\n\nClaim [1].\n\n{code}\n",
    )

    rendered = service.render(request, draft, _single_citation())

    assert code in rendered
    assert "Claim [1]." in rendered


def test_marker_passes_leave_inline_code_and_intervals_untouched() -> None:
    service = ReportService()
    request = ResearchRequest(query="q")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "Read `arr[2]` then normalized to the range [0, 1].\n\n"
            "Claim [1].\n"
        ),
    )

    rendered = service.render(request, draft, _single_citation())

    assert "Read `arr[2]` then normalized to the range [0, 1]." in rendered
    assert "Claim [1]." in rendered


def test_marker_passes_leave_links_untouched() -> None:
    service = ReportService()
    request = ResearchRequest(query="q")
    link = "[manual](https://example.com/p_[2])"
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=f"# Report\n\nSee {link} and [1](https://example.com). Claim [1].\n",
    )

    rendered = service.render(request, draft, _single_citation())

    assert link in rendered
    assert "[1](https://example.com)" in rendered


def test_sources_heading_with_prose_url_keeps_following_sections() -> None:
    service = ReportService()
    request = ResearchRequest(query="q")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "## Sources\n\n"
            "See https://example.com for methodology.\n\n"
            "## Conclusions\n\n"
            "Done [1].\n"
        ),
    )

    rendered = service.render(request, draft, _single_citation())

    assert "See https://example.com for methodology." in rendered
    assert "## Conclusions" in rendered
    assert "Done [1]." in rendered


def test_bibliography_block_ends_at_next_same_level_heading() -> None:
    service = ReportService()
    request = ResearchRequest(query="q")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "Finding A is supported [1].\n\n"
            "## Sources\n\n"
            "[1] model-authored bibliography entry\n"
            "2. another invented entry\n\n"
            "## Conclusions\n\n"
            "Kept [1].\n"
        ),
    )

    rendered = service.render(request, draft, _single_citation())

    assert "model-authored bibliography entry" not in rendered
    assert "another invented entry" not in rendered
    assert "## Conclusions" in rendered
    assert "Kept [1]." in rendered


def test_bulleted_link_bibliography_is_stripped() -> None:
    service = ReportService()
    request = ResearchRequest(query="q")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "Finding A is supported [1].\n\n"
            "## Sources\n\n"
            "- [Title](https://a.example/x)\n"
            "- plain bullet without a link\n"
        ),
    )

    rendered = service.render(request, draft, _single_citation())

    assert "https://a.example/x" not in rendered
    assert "plain bullet without a link" not in rendered


def test_bare_references_heading_ends_at_next_heading() -> None:
    service = ReportService()
    request = ResearchRequest(query="q")
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "Finding A is supported [1].\n\n"
            "References\n\n"
            "[1] model-invented reference\n\n"
            "## Conclusions\n\n"
            "Kept [1].\n"
        ),
    )

    rendered = service.render(request, draft, _single_citation())

    assert "model-invented reference" not in rendered
    assert "## Conclusions" in rendered
