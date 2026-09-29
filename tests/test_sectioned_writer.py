from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from shandu.agents.lead import LeadAgent, _OutlinePayload, _OutlineSection
from shandu.contracts import Claim, EvidenceRecord, ResearchBrief, ResearchRequest


def _evidence(evidence_id: str, statement: str, sub_question: int) -> dict:
    return EvidenceRecord(
        evidence_id=evidence_id,
        task_id="t",
        query="q",
        requested_url=f"https://{evidence_id}.org/a",
        domain=f"{evidence_id}.org",
        title=evidence_id,
        snippet="s",
        extracted_text=f"text of {evidence_id}",
        credibility_score=0.9,
        claims=[Claim(statement=statement, quote="q", sub_question=sub_question)],
    ).model_dump(mode="json")


_EVIDENCE = [
    _evidence("e1", "KIVI keeps keys per channel", 1),
    _evidence("e2", "SGLang serves Llama at twice the rate", 2),
]
_CITATIONS = [
    {"citation_id": 1, "evidence_ids": ["e1"], "url": "https://e1.org/a", "title": "e1", "publisher": "e1.org"},
    {"citation_id": 2, "evidence_ids": ["e2"], "url": "https://e2.org/a", "title": "e2", "publisher": "e2.org"},
]


class _WriterDesk:
    def __init__(self, editor=None, outline_status: str = "completed") -> None:
        self.editor = editor
        self.outline_status = outline_status
        self.section_payloads: list[dict] = []
        self.outline_inputs: list[str] = []
        self.names: list[str] = []

    async def arun(self, worker, job):
        self.names.append(worker.name)
        if worker.name == "LeadOutliner":
            self.outline_inputs.append(job.input)
            return SimpleNamespace(
                status=self.outline_status,
                errors=[] if self.outline_status == "completed" else ["outline boom"],
                data=_OutlinePayload(
                    title="Serving and Caching",
                    sections=[
                        _OutlineSection(heading="## Caching", finding_ids=[1]),
                        _OutlineSection(heading="## Serving", finding_ids=[2]),
                    ],
                ),
                metrics={},
            )
        if worker.name == "SectionWriter":
            payload = json.loads(job.input.split("Input JSON:\n", 1)[1])
            self.section_payloads.append(payload)
            content = f"{payload['heading']}\n\nBody for {payload['heading'][3:]} [1]."
        elif worker.name == "LeadEditor":
            draft = json.loads(job.input.split("Input JSON:\n", 1)[1])["draft"]
            content = self.editor(draft)
        else:
            content = "# Single\n\n## Executive Summary\n\nOne call."
        return SimpleNamespace(status="completed", content=content, metrics={})


def _write(desk: _WriterDesk, effort: str = "medium"):
    agent = LeadAgent(SimpleNamespace(settings=SimpleNamespace(model="m"), desk=desk))
    request = ResearchRequest(query="q", writer="sectioned", effort=effort)
    brief = ResearchBrief(scope="s", sub_questions=["caching?", "serving?"])
    return asyncio.run(agent.build_final_report(request, [], _EVIDENCE, _CITATIONS, brief=brief))


def test_sections_see_only_their_evidence_and_the_edit_is_kept() -> None:
    desk = _WriterDesk(editor=lambda draft: f"# Serving and Caching\n\n## Executive Summary\n\nBoth.\n\n{draft}")

    draft = _write(desk)

    by_heading = {
        payload["heading"]: [item["title"] for item in payload["evidence"]]
        for payload in desk.section_payloads
    }
    assert by_heading == {"## Caching": ["e1"], "## Serving": ["e2"]}
    assert [
        [entry["citation_id"] for entry in payload["citations"]]
        for payload in desk.section_payloads
    ] == [[1], [2]]
    assert desk.section_payloads[0]["findings"][0]["citation_ids"] == [1]
    assert draft.markdown.startswith("# Serving and Caching\n\n## Executive Summary")
    assert "Body for Serving" in draft.markdown


@pytest.mark.parametrize(
    "lossy_edit",
    [
        lambda draft: draft.replace("## Serving", "Serving"),
        lambda draft: "# T\n\n## Caching\n\n## Serving",
    ],
    ids=["drops_a_heading", "cuts_the_content"],
)
def test_lossy_edit_is_rejected(lossy_edit) -> None:
    desk = _WriterDesk(editor=lossy_edit)

    draft = _write(desk)

    assert draft.markdown.startswith("# Serving and Caching\n\n## Caching")
    assert "## Serving" in draft.markdown


def test_failed_outline_falls_back_to_single_writer() -> None:
    desk = _WriterDesk(outline_status="failed")

    draft = _write(desk)

    assert draft.markdown == "# Single\n\n## Executive Summary\n\nOne call."
    assert desk.names == ["LeadOutliner", "LeadReporter"]


def test_word_budget_scales_with_effort() -> None:
    desk = _WriterDesk(editor=lambda draft: draft)

    _write(desk, effort="max")

    assert "about 7200 words" in desk.outline_inputs[0]


def test_sections_without_findings_draw_on_all_evidence() -> None:
    desk = _WriterDesk(editor=lambda draft: draft)
    agent = LeadAgent(SimpleNamespace(settings=SimpleNamespace(model="m"), desk=desk))
    claimless = [{**entry, "claims": []} for entry in _EVIDENCE]

    asyncio.run(
        agent.build_final_report(
            ResearchRequest(query="q", writer="sectioned"), [], claimless, _CITATIONS
        )
    )

    assert [sorted(item["title"] for item in payload["evidence"]) for payload in desk.section_payloads] == [
        ["e1", "e2"],
        ["e1", "e2"],
    ]
