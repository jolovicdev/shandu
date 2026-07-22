from __future__ import annotations

import asyncio
from types import SimpleNamespace

from shandu.agents.citation_agent import (
    CitationAgent,
    _CitationBundle,
    _CitationCandidate,
)
from shandu.contracts import EvidenceRecord

PAPER_TITLE = "The Illusion of Diminishing Returns: Measuring Long Horizon Execution"


class FailingDesk:
    async def arun(self, worker, job):
        del worker, job
        raise RuntimeError("forced")


class BundleDesk:
    def __init__(self, bundle: _CitationBundle) -> None:
        self._bundle = bundle

    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(status="completed", data=self._bundle)


class FakeRuntime:
    def __init__(self, desk: object | None = None) -> None:
        self.settings = SimpleNamespace(model="deepseek/deepseek-v4-flash")
        self.desk = desk if desk is not None else FailingDesk()


def _record(
    evidence_id: str,
    url: str,
    title: str,
    credibility: float | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        task_id="t",
        query="q",
        requested_url=url,
        title=title,
        snippet="s",
        extracted_text="x",
        confidence=0.7,
        credibility_score=credibility,
    )


def test_citation_agent_falls_back_to_deterministic_entries() -> None:
    agent = CitationAgent(runtime=FakeRuntime())
    evidence = [
        EvidenceRecord(
            evidence_id="e1",
            task_id="t1",
            query="q",
            requested_url="https://example.com/a",
            title="A",
            snippet="s",
            extracted_text="x",
            confidence=0.8,
        ),
        EvidenceRecord(
            evidence_id="e2",
            task_id="t2",
            query="q",
            requested_url="https://example.com/a",
            title="A2",
            snippet="s2",
            extracted_text="x2",
            confidence=0.6,
        ),
        EvidenceRecord(
            evidence_id="e3",
            task_id="t3",
            query="q",
            requested_url="https://another.net/b",
            title="B",
            snippet="s3",
            extracted_text="x3",
            confidence=0.9,
        ),
    ]

    citations = asyncio.run(agent.build_citations("query", evidence))

    assert len(citations) == 2
    assert citations[0].citation_id == 1
    assert citations[1].citation_id == 2
    assert set(citations[0].evidence_ids) == {"e1", "e2"}


def test_citation_payload_omits_full_body_and_caps_snippet() -> None:
    class CapturingDesk:
        def __init__(self) -> None:
            self.captured: str | None = None

        async def arun(self, worker, job):
            del worker
            self.captured = job.input
            raise RuntimeError("stop")

    class Runtime:
        def __init__(self, desk: CapturingDesk) -> None:
            self.settings = SimpleNamespace(model="deepseek/deepseek-v4-flash")
            self.desk = desk

    desk = CapturingDesk()
    agent = CitationAgent(runtime=Runtime(desk))
    evidence = [
        EvidenceRecord(
            evidence_id="e1",
            task_id="t1",
            query="q",
            requested_url="https://example.com/a",
            title="A",
            snippet="S" * 400,
            extracted_text="SECRET_BODY_TEXT",
            confidence=0.8,
        )
    ]

    asyncio.run(agent.build_citations("query", evidence))

    assert desk.captured is not None
    assert "SECRET_BODY_TEXT" not in desk.captured
    assert "extracted_text" not in desk.captured
    assert "S" * 280 in desk.captured
    assert "S" * 281 not in desk.captured


def test_fallback_merges_same_work_url_variants() -> None:
    agent = CitationAgent(runtime=FakeRuntime())
    evidence = [
        _record("e1", "https://arxiv.org/abs/2509.09677", PAPER_TITLE),
        _record("e2", "https://arxiv.org/html/2509.09677", PAPER_TITLE),
        _record("e3", "https://mirror.example/paper", PAPER_TITLE),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert len(citations) == 2
    assert citations[0].url == "https://arxiv.org/abs/2509.09677"
    assert set(citations[0].evidence_ids) == {"e1", "e2"}
    assert citations[1].url == "https://mirror.example/paper"
    assert [entry.citation_id for entry in citations] == [1, 2]


def test_fallback_short_titles_do_not_merge() -> None:
    agent = CitationAgent(runtime=FakeRuntime())
    evidence = [
        _record("e1", "https://sec.example/filings/a", "10-K"),
        _record("e2", "https://sec.example/filings/b", "10-K"),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert len(citations) == 2


def test_low_credibility_evidence_left_out_of_ledger() -> None:
    agent = CitationAgent(runtime=FakeRuntime())
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
    agent = CitationAgent(runtime=FakeRuntime())
    evidence = [
        _record("e1", "https://blog.example/a", "First Weak Blog Post Title", 0.2),
        _record("e2", "https://feed.example/b", "Second Weak Feed Page Title", 0.1),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert len(citations) == 2


def test_fallback_sanitizes_titles() -> None:
    agent = CitationAgent(runtime=FakeRuntime())
    evidence = [
        _record("e1", "https://example.com/a", "Line One\nLine   Two Extended Title"),
        _record("e2", "https://example.com/b", "https://example.com/b"),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    by_url = {entry.url: entry for entry in citations}
    assert by_url["https://example.com/a"].title == "Line One Line Two Extended Title"
    assert by_url["https://example.com/b"].title == "example.com"


def test_fallback_caps_lede_length_titles() -> None:
    agent = CitationAgent(runtime=FakeRuntime())
    lede = "A british tokamak just came apart after many plasma pulses " * 6
    evidence = [_record("e1", "https://example.com/a", lede)]

    citations = asyncio.run(agent.build_citations("q", evidence))

    title = citations[0].title
    assert len(title) <= 160
    assert title.endswith("...")


def test_normalize_merges_variants_and_renumbers_sequentially() -> None:
    bundle = _CitationBundle(
        citations=[
            _CitationCandidate(
                evidence_ids=["e1"],
                url="https://arxiv.org/abs/2509.09677",
                title=PAPER_TITLE,
                publisher="arXiv",
            ),
            _CitationCandidate(
                evidence_ids=["e2"],
                url="https://arxiv.org/html/2509.09677",
                title=PAPER_TITLE,
                publisher="arXiv",
            ),
            _CitationCandidate(
                evidence_ids=["e3"],
                url="https://mirror.example/paper",
                title=PAPER_TITLE,
                publisher="Mirror",
            ),
        ]
    )
    agent = CitationAgent(runtime=FakeRuntime(desk=BundleDesk(bundle)))
    evidence = [
        _record("e1", "https://arxiv.org/abs/2509.09677", PAPER_TITLE),
        _record("e2", "https://arxiv.org/html/2509.09677", PAPER_TITLE),
        _record("e3", "https://mirror.example/paper", PAPER_TITLE),
    ]

    citations = asyncio.run(agent.build_citations("q", evidence))

    assert [entry.citation_id for entry in citations] == [1, 2]
    assert set(citations[0].evidence_ids) == {"e1", "e2"}
    assert citations[1].url == "https://mirror.example/paper"
