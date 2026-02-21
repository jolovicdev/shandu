from __future__ import annotations

from shandu.contracts import CitationEntry, FinalReportDraft, ReportSection, ResearchRequest
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
    assert "[1] energy.example. \"Energy Analysis\". https://energy.example/analysis" in rendered
