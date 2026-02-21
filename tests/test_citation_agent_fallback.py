from __future__ import annotations

import asyncio
from types import SimpleNamespace

from shandu.agents.citation_agent import CitationAgent
from shandu.contracts import EvidenceRecord


class FailingDesk:
    async def arun(self, worker, job):
        del worker, job
        raise RuntimeError("forced")


class FakeRuntime:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(model="deepseek/deepseek-chat")
        self.desk = FailingDesk()


def test_citation_agent_falls_back_to_deterministic_entries() -> None:
    agent = CitationAgent(runtime=FakeRuntime())
    evidence = [
        EvidenceRecord(
            evidence_id="e1",
            task_id="t1",
            query="q",
            url="https://example.com/a",
            title="A",
            snippet="s",
            extracted_text="x",
            confidence=0.8,
        ),
        EvidenceRecord(
            evidence_id="e2",
            task_id="t2",
            query="q",
            url="https://example.com/a",
            title="A2",
            snippet="s2",
            extracted_text="x2",
            confidence=0.6,
        ),
        EvidenceRecord(
            evidence_id="e3",
            task_id="t3",
            query="q",
            url="https://another.net/b",
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
