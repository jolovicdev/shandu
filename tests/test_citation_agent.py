from __future__ import annotations

import asyncio

from shandu.agents.citation_agent import CitationAgent
from shandu.contracts import EvidenceRecord

PAPER_TITLE = "The Illusion of Diminishing Returns: Measuring Long Horizon Execution"


def _record(
    evidence_id: str,
    url: str,
    title: str,
    credibility: float | None = None,
    site_name: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        task_id="t",
        query="q",
        requested_url=url,
        title=title,
        site_name=site_name,
        snippet="s",
        extracted_text="x",
        confidence=0.7,
        credibility_score=credibility,
    )


def test_ledger_groups_shared_url_and_unions_evidence_ids() -> None:
    agent = CitationAgent()
    evidence = [
        _record("e1", "https://example.com/a", "Title A"),
        _record("e2", "https://example.com/a", "Title A Second Record"),
        _record("e3", "https://another.net/b", "Title B"),
    ]

    citations = asyncio.run(agent.build_citations("query", evidence))

    assert len(citations) == 2
    assert citations[0].citation_id == 1
    assert citations[1].citation_id == 2
    assert set(citations[0].evidence_ids) == {"e1", "e2"}


def test_ledger_contains_only_evidence_urls() -> None:
    agent = CitationAgent()
    evidence = [_record("e1", "https://good.example/a", "A Solid Article Title")]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert {entry.url for entry in citations} == {"https://good.example/a"}
    assert "https://invented.example/" not in {entry.url for entry in citations}
    known_ids = {item.evidence_id for item in evidence}
    for entry in citations:
        assert set(entry.evidence_ids) <= known_ids


def test_ledger_merges_same_work_variants_and_unions_ids() -> None:
    agent = CitationAgent()
    evidence = [
        _record("e1", "https://arxiv.org/abs/2509.09677", PAPER_TITLE),
        _record("e2", "https://arxiv.org/html/2509.09677", PAPER_TITLE),
        _record("e3", "https://mirror.example/paper", PAPER_TITLE),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert [entry.citation_id for entry in citations] == [1, 2]
    assert citations[0].url == "https://arxiv.org/abs/2509.09677"
    assert set(citations[0].evidence_ids) == {"e1", "e2"}
    assert citations[1].url == "https://mirror.example/paper"


def test_ledger_short_titles_do_not_merge() -> None:
    agent = CitationAgent()
    evidence = [
        _record("e1", "https://sec.example/filings/a", "10-K"),
        _record("e2", "https://sec.example/filings/b", "10-K"),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert len(citations) == 2


def test_low_credibility_evidence_left_out_of_ledger() -> None:
    agent = CitationAgent()
    evidence = [
        _record("e1", "https://journal.example/a", "Strong Journal Article Title", 0.8),
        _record("e2", "https://blog.example/b", "Weak Blog Post Title Here", 0.2),
        _record("e3", "https://docs.example/c", "Unassessed Documentation Page", None),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert {entry.url for entry in citations} == {
        "https://journal.example/a",
        "https://docs.example/c",
    }


def test_all_weak_corpus_keeps_full_ledger() -> None:
    agent = CitationAgent()
    evidence = [
        _record("e1", "https://blog.example/a", "First Weak Blog Post Title", 0.2),
        _record("e2", "https://feed.example/b", "Second Weak Feed Page Title", 0.1),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert len(citations) == 2


def test_ledger_sanitizes_titles() -> None:
    agent = CitationAgent()
    evidence = [
        _record("e1", "https://example.com/a", "Line One\nLine   Two Extended Title"),
        _record("e2", "https://example.com/b", "https://example.com/b"),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    by_url = {entry.url: entry for entry in citations}
    assert by_url["https://example.com/a"].title == "Line One Line Two Extended Title"
    assert by_url["https://example.com/b"].title == "example.com"


def test_ledger_strips_markup_and_merges_variants_that_differ_only_by_it() -> None:
    agent = CitationAgent()
    evidence = [
        _record("e1", "https://journal.example/doi/full/1", "Efficacy of <fc>GLP</fc>\u2010 1 agonists in trials"),
        _record("e2", "https://journal.example/doi/pdf/1", "Efficacy of GLP-1 agonists in trials"),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert [entry.title for entry in citations] == ["Efficacy of GLP\u2010 1 agonists in trials"]
    assert set(citations[0].evidence_ids) == {"e1", "e2"}


def test_ledger_names_an_untitled_document_after_its_file() -> None:
    agent = CitationAgent()
    url = "https://gov.example/Documents/2025/Informe-no-confidencial-Comite-28A.pdf"

    citations = asyncio.run(agent.build_citations("q", [_record("e1", url, "gov.example")]))

    assert citations[0].title == "Informe no confidencial Comite 28A"


def test_ledger_caps_lede_length_titles() -> None:
    agent = CitationAgent()
    lede = "A british tokamak just came apart after many plasma pulses " * 6
    evidence = [_record("e1", "https://example.com/a", lede)]

    citations = asyncio.run(agent.build_citations("q", evidence))

    title = citations[0].title
    assert len(title) <= 160
    assert title.endswith("...")


def test_ledger_prefers_site_name_for_publisher() -> None:
    agent = CitationAgent()
    evidence = [
        _record(
            "e1",
            "https://news.example/articles/1",
            "A Named Publisher Article Title",
            site_name="Example News",
        ),
        _record("e2", "https://www.bare.example/page", "A Page Without Site Metadata"),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    by_url = {entry.url: entry for entry in citations}
    assert by_url["https://news.example/articles/1"].publisher == "Example News"
    assert by_url["https://www.bare.example/page"].publisher == "bare.example"


def test_empty_evidence_yields_empty_ledger() -> None:
    agent = CitationAgent()

    assert asyncio.run(agent.build_citations("q", [])) == []


def test_ledger_leaves_out_pages_the_extractor_judged_off_topic() -> None:
    agent = CitationAgent()
    useful = _record("e1", "https://example.org/study", "Field study of heat pumps", 0.8)
    off_topic = _record("e2", "https://utility.example/rates", "Electric Supply Rates", 0.6)
    off_topic.confidence = 0.2

    citations = asyncio.run(agent.build_citations("query", [useful, off_topic]))

    assert [entry.url for entry in citations] == ["https://example.org/study"]
