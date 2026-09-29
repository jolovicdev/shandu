from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from shandu.agents.auditor import (
    AuditOutcome,
    CitationAuditor,
    _AuditPayload,
    _cited_units,
    _MarkerVerdict,
    _UnitVerdict,
)
from shandu.contracts import (
    Claim,
    CitationEntry,
    CitationVerdict,
    EvidenceRecord,
    FinalReportDraft,
    IterationPlan,
    IterationSynthesis,
    ResearchBrief,
    ResearchRequest,
    SubagentTask,
)
from shandu.orchestration.lead_orchestrator import LeadOrchestrator
from shandu.services.report import ReportService
from shandu.services.run_store import RunStore

_REPORT = (
    "# Title\n\n"
    "## Memory\n\n"
    "KIVI stores keys per channel [1][2]. It cuts memory 2.6x on Llama-2-7B [1].\n\n"
    "| Method | Saving |\n"
    "| --- | --- |\n"
    "| KIVI. Asymmetric | 2.6x [2] |\n\n"
    "## References\n\n"
    "- **[1] arxiv.org** - \"KIVI\". [Source](https://arxiv.org/abs/1) (accessed 2026-09-29)\n"
    "- **[2] blog.example.com** - \"Blog\". [Source](https://blog.example.com/a) (accessed 2026-09-29)"
)
_CITATIONS = [
    CitationEntry(citation_id=1, evidence_ids=["e1"], url="https://arxiv.org/abs/1", title="KIVI", publisher="arxiv.org", accessed_at="2026-09-29"),
    CitationEntry(citation_id=2, evidence_ids=["e2"], url="https://blog.example.com/a", title="Blog", publisher="blog.example.com", accessed_at="2026-09-29"),
]


def _record(evidence_id: str, text: str, *quotes: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        task_id="t",
        query="q",
        requested_url=f"https://{evidence_id}.org",
        title=evidence_id,
        snippet="s",
        extracted_text=text,
        claims=[Claim(statement="c", quote=quote) for quote in quotes],
    )


_EVIDENCE = [
    _record("e1", "KIVI quantizes keys per channel.", "keys are quantized per channel"),
    _record("e2", "A blog about caching."),
]


def test_cited_units_split_sentences_and_keep_table_rows_whole() -> None:
    units = _cited_units(_REPORT.split("\n## References\n")[0])

    assert [(unit.text, unit.citation_ids) for unit in units] == [
        ("KIVI stores keys per channel [1][2].", (1, 2)),
        ("It cuts memory 2.6x on Llama-2-7B [1].", (1,)),
        ("| KIVI. Asymmetric | 2.6x [2] |", (2,)),
    ]


class _AuditDesk:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[dict] = []

    async def arun(self, worker, job):
        self.payloads.append(json.loads(job.input.split("Input JSON:\n", 1)[1]))
        if self.fail:
            return SimpleNamespace(status="failed", errors=["boom"], data=None, metrics={})
        return SimpleNamespace(
            status="completed",
            data=_AuditPayload(
                units=[
                    _UnitVerdict(
                        unit_id=1,
                        markers=[
                            _MarkerVerdict(citation_id=1, verdict="supported"),
                            _MarkerVerdict(citation_id=2, verdict="unsupported"),
                        ],
                    ),
                    _UnitVerdict(unit_id=3, markers=[_MarkerVerdict(citation_id=2, verdict="unsupported")]),
                ]
            ),
            metrics={"usage": {"total_tokens": 7}, "cost_usd": 0.001},
        )


def _auditor(desk) -> CitationAuditor:
    return CitationAuditor(SimpleNamespace(settings=SimpleNamespace(model="m"), desk=desk))


def test_audit_strips_only_unsupported_markers_where_they_appear() -> None:
    desk = _AuditDesk()

    outcome = asyncio.run(_auditor(desk).audit(_REPORT, _CITATIONS, _EVIDENCE))

    assert "KIVI stores keys per channel [1]. It cuts memory 2.6x on Llama-2-7B [1]." in outcome.markdown
    assert "| KIVI. Asymmetric | 2.6x |" in outcome.markdown
    assert outcome.markdown.endswith(_REPORT.split("\n## References\n")[1])
    assert [(item.url, item.verdict) for item in outcome.verdicts] == [
        ("https://arxiv.org/abs/1", "supported"),
        ("https://blog.example.com/a", "unsupported"),
        ("https://arxiv.org/abs/1", "unchecked"),
        ("https://blog.example.com/a", "unsupported"),
    ]
    sources = {source["citation_id"]: source["text"] for source in desk.payloads[0]["sources"]}
    assert sources == {
        1: 'KIVI quantizes keys per channel.\n"keys are quantized per channel"',
        2: "A blog about caching.",
    }
    assert (outcome.calls, outcome.llm_usage["cost_usd"]) == (1, 0.001)


def test_auditor_sees_long_evidence_whole() -> None:
    desk = _AuditDesk()
    long_text = "KIVI result sentence. " * 400

    asyncio.run(
        _auditor(desk).audit(_REPORT, _CITATIONS, [_record("e1", long_text), _EVIDENCE[1]])
    )

    source = next(item for item in desk.payloads[0]["sources"] if item["citation_id"] == 1)
    assert source["text"] == long_text


def test_failed_audit_batch_leaves_markers_unchecked() -> None:
    outcome = asyncio.run(_auditor(_AuditDesk(fail=True)).audit(_REPORT, _CITATIONS, _EVIDENCE))

    assert outcome.markdown == _REPORT
    assert {item.verdict for item in outcome.verdicts} == {"unchecked"}


class _Lead:
    fallback_count = 0

    async def create_brief(self, request):
        return ResearchBrief(scope="s", sub_questions=["q"])

    async def create_iteration_plan(self, request, iteration, prior_summaries, memory_context, brief=None):
        return IterationPlan(
            iteration_index=iteration,
            subagent_tasks=[SubagentTask(task_id="t", focus="f")],
            continue_loop=False,
        )

    async def synthesize_iteration(self, request, iteration, iteration_evidence, prior_summaries, brief=None, task_memos=None):
        return IterationSynthesis(summary="s")

    async def build_final_report(self, request, iteration_summaries, evidence_payload, citations_payload, brief=None):
        return FinalReportDraft(title="T", executive_summary="E", markdown="# T\n\nA claim [1]. Another [2].")


class _Subagent:
    async def execute_task(self, run_scope, task, request, progress_callback=None, extracted_urls=None, brief=None):
        return list(_EVIDENCE)


class _Citations:
    async def build_citations(self, query, evidence):
        return list(_CITATIONS)


class _StripSecond:
    def __init__(self) -> None:
        self.calls = 0

    async def audit(self, markdown, citations, evidence):
        self.calls += 1
        return AuditOutcome(
            markdown=markdown.replace(" [2]", ""),
            verdicts=[
                CitationVerdict(sentence="A claim", url=citations[0].url, verdict="supported"),
                CitationVerdict(sentence="Another", url=citations[1].url, verdict="unsupported"),
            ],
            llm_usage=None,
            calls=1,
        )


def _run(tmp_path, audit: bool):
    auditor = _StripSecond()
    orchestrator = LeadOrchestrator(
        lead_agent=_Lead(),
        search_subagent=_Subagent(),
        citation_agent=_Citations(),
        run_store=RunStore(tmp_path / "runs"),
        report_service=ReportService(),
        auditor=auditor,
    )
    result = asyncio.run(
        orchestrator.run(ResearchRequest(query="q", max_iterations=1, parallelism=1, audit=audit))
    )
    return result, auditor


def test_orchestrator_rerenders_after_audit_and_counts_verdicts(tmp_path) -> None:
    result, auditor = _run(tmp_path, audit=True)

    assert auditor.calls == 1
    assert [entry.url for entry in result.citations] == ["https://arxiv.org/abs/1"]
    assert "[2]" not in result.report_markdown
    assert result.run_stats["citation_audit"] == {
        "supported": 1,
        "partial": 0,
        "unsupported": 1,
        "unchecked": 0,
    }
    assert [item.verdict for item in result.citation_audit] == ["supported", "unsupported"]
    stored = RunStore(tmp_path / "runs").load_result(result.run_id)
    assert stored.citation_audit == result.citation_audit


def test_audit_off_leaves_report_unchecked(tmp_path) -> None:
    result, auditor = _run(tmp_path, audit=False)

    assert auditor.calls == 0
    assert len(result.citations) == 2
    assert "citation_audit" not in result.run_stats


def test_cited_units_ignore_code_and_links_and_keep_trailing_markers() -> None:
    body = (
        "Revenue grew 12% in 2024. [1] Margins fell to 8% [2].\n"
        "Costs rose sharply. [3]\n"
        "```python\n"
        "x = values[1]\n"
        "```\n"
        "Use `items[2]` and see [the guide](https://example.org/a[3]) for details [4]."
    )

    units = _cited_units(body)

    assert [(unit.text, unit.citation_ids) for unit in units] == [
        ("Revenue grew 12% in 2024. [1]", (1,)),
        ("Margins fell to 8% [2].", (2,)),
        ("Costs rose sharply. [3]", (3,)),
        ("Use `items[2]` and see [the guide](https://example.org/a[3]) for details [4].", (4,)),
    ]


def test_audit_never_edits_code_that_looks_like_a_marker() -> None:
    report = "Intro claim [2].\n\nUse `data[2]` as shown [2].\n\n## References\n\n- [2] x"

    class _Unsupported:
        async def arun(self, worker, job):
            payload = json.loads(job.input.split("Input JSON:\n", 1)[1])
            return SimpleNamespace(
                status="completed",
                data=_AuditPayload(
                    units=[
                        _UnitVerdict(
                            unit_id=unit["unit_id"],
                            markers=[_MarkerVerdict(citation_id=2, verdict="unsupported")],
                        )
                        for unit in payload["units"]
                    ]
                ),
                metrics={},
            )

    outcome = asyncio.run(_auditor(_Unsupported()).audit(report, _CITATIONS, _EVIDENCE))

    assert outcome.markdown.startswith("Intro claim.\n\nUse `data[2]` as shown.")
