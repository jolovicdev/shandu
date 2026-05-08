from __future__ import annotations

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
    evidence_id = "a93a4e1b65ff42009c95f52329c5179e"
    draft = FinalReportDraft(
        title="Report",
        executive_summary="Summary",
        sections=[],
        markdown=(
            "# Report\n\n"
            "## Executive Summary\n\n"
            f"Energy demand is rising [{evidence_id}][{evidence_id}] and market salaries are rising [1][99].\n\n"
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
