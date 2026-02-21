from __future__ import annotations

from shandu.contracts import CitationEntry


def test_citation_entry_model() -> None:
    entry = CitationEntry(
        citation_id=1,
        evidence_ids=["e1", "e2"],
        url="https://example.com",
        title="Example",
        publisher="example.com",
        accessed_at="2026-02-21",
    )
    assert entry.citation_id == 1
    assert entry.publisher == "example.com"
