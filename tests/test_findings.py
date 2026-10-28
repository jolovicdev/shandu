from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from shandu.agents.search_subagent import SearchSubagent, _ExtractionPayload
from shandu.cli import ShanduEngine, cli
from shandu.contracts import (
    Claim,
    EvidenceRecord,
    ResearchBrief,
    ResearchRequest,
    ResearchRunResult,
    SubagentTask,
)
from shandu.services.findings import build_findings, coverage_matrix, verified_claims
from shandu.services.scrape import ScrapedPage
from shandu.services.search import SearchHit

_PAGE = "KIVI quantizes the key cache per channel.\n\nIt cuts peak memory by  2.6x on Llama-2-7B."


def _record(evidence_id: str, domain: str, credibility: float, *claims: Claim) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        task_id="t",
        query="q",
        requested_url=f"https://{domain}/{evidence_id}",
        domain=domain,
        title="T",
        snippet="s",
        extracted_text="x",
        credibility_score=credibility,
        claims=list(claims),
    )


def test_verified_claims_keep_verbatim_quotes_only() -> None:
    claims = [
        Claim(statement="Memory drops 2.6x", quote="it cuts peak memory by 2.6x", sub_question=2),
        Claim(statement="Paraphrased", quote="It reduces memory 2.6 times", sub_question=1),
        Claim(statement="Out of range", quote="quantizes the key cache per channel", sub_question=7),
    ]

    kept = verified_claims(claims, _PAGE, sub_question_count=3)

    assert [(claim.statement, claim.sub_question) for claim in kept] == [
        ("Memory drops 2.6x", 2),
        ("Out of range", None),
    ]


def test_verified_claims_reject_quotes_too_short_to_prove_anything() -> None:
    claims = [
        Claim(statement="Drug X cut events by 20%", quote="the"),
        Claim(statement="Memory drops", quote="peak memory by 2.6x"),
    ]

    kept = verified_claims(claims, "the drug cut peak memory by 2.6x on the model", 1)

    assert [claim.statement for claim in kept] == ["Memory drops"]


def test_build_findings_merges_restated_claims_across_pages() -> None:
    same = "KIVI cuts peak memory by 2.6x on Llama-2-7B"
    findings = build_findings(
        [
            _record("e1", "arxiv.org", 0.9, Claim(statement=same, quote="q1", sub_question=1)),
            _record(
                "e2",
                "blog.example.com",
                0.4,
                Claim(statement="KIVI cuts peak memory by 2.6x on Llama-2-7B.", quote="q2", sub_question=2),
                Claim(statement="KIVI needs no fine-tuning", quote="q3"),
            ),
        ]
    )

    assert len(findings) == 2
    merged = findings[0]
    assert merged.evidence_ids == ["e1", "e2"]
    assert merged.quotes == ["q1", "q2"]
    assert merged.domains == ["arxiv.org", "blog.example.com"]
    assert merged.sub_questions == [1, 2]
    assert merged.confidence == 0.9


def test_coverage_matrix_rates_independent_credible_support() -> None:
    brief = ResearchBrief(scope="s", sub_questions=["q1", "q2", "q3", "q4"])
    evidence = [
        _record("e1", "a.org", 0.9, Claim(statement="alpha one", quote="x", sub_question=1)),
        _record("e2", "b.org", 0.8, Claim(statement="alpha two", quote="x", sub_question=1)),
        _record("e3", "a.org", 0.9, Claim(statement="beta one", quote="x", sub_question=2)),
        _record("e4", "a.org", 0.9, Claim(statement="beta two", quote="x", sub_question=2)),
        _record("e5", "c.org", 0.3, Claim(statement="gamma", quote="x", sub_question=3)),
    ]

    matrix = coverage_matrix(brief, evidence)

    assert [(item.strength, item.credible_sources) for item in matrix] == [
        ("strong", 2),
        ("moderate", 1),
        ("weak", 0),
        ("none", 0),
    ]


class _ClaimDesk:
    async def arun(self, worker, job):
        del worker, job
        return SimpleNamespace(
            status="completed",
            data=_ExtractionPayload(
                snippet="s",
                extracted_text="t",
                confidence=0.5,
                source_class="unknown",
                authorship="anonymous",
                is_promotional=False,
                summarizes_inaccessible_source=False,
                claims=[
                    Claim(statement="per-channel keys", quote="quantizes the key cache per channel", sub_question=1),
                    Claim(statement="invented", quote="it doubles throughput", sub_question=1),
                ],
            ),
            metrics={},
        )


class _OneHit:
    async def search(self, query, max_results):
        return [SearchHit(query=query, url="https://arxiv.org/abs/1", title="KIVI", snippet="s")]


class _OnePage:
    async def scrape_many(self, urls):
        return [
            ScrapedPage(requested_url=url, url=url, title="KIVI", text=_PAGE, domain="arxiv.org")
            for url in urls
        ], 0


def test_extraction_stores_only_claims_found_in_the_page() -> None:
    runtime = SimpleNamespace(settings=SimpleNamespace(model="m"), desk=_ClaimDesk())
    subagent = SearchSubagent(runtime, _OneHit(), _OnePage())

    evidence = asyncio.run(
        subagent.execute_task(
            "run:r",
            SubagentTask(task_id="t", focus="f", search_queries=["kivi"]),
            ResearchRequest(query="q"),
            brief=ResearchBrief(scope="s", sub_questions=["how does KIVI quantize?"]),
        )
    )

    assert [claim.statement for claim in evidence[0].claims] == ["per-channel keys"]


def test_effort_preset_sets_budgets_and_flags_override_it(monkeypatch) -> None:
    seen: dict[str, ResearchRequest] = {}

    async def run(request, progress_callback=None):
        seen["request"] = request
        return ResearchRunResult(run_id="r", request=request, report_markdown="Body")

    engine = SimpleNamespace(run=run, close=lambda: None, report_path=lambda run_id: Path("missing"))
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(
        cli, ["run", "q", "--effort", "max", "--max-iterations", "2"]
    )

    assert result.exit_code == 0, result.output
    request = seen["request"]
    assert request.effort == "max"
    assert (request.max_iterations, request.parallelism, request.max_pages_per_task) == (2, 6, 8)
    assert request.max_searches_per_task == 10


def test_medium_effort_matches_the_default_budgets(monkeypatch) -> None:
    seen: list[ResearchRequest] = []

    async def run(request, progress_callback=None):
        seen.append(request)
        return ResearchRunResult(run_id="r", request=request, report_markdown="Body")

    engine = SimpleNamespace(run=run, close=lambda: None, report_path=lambda run_id: Path("missing"))
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    for args in (["run", "q"], ["run", "q", "--effort", "medium"]):
        result = CliRunner(mix_stderr=False).invoke(cli, args)
        assert result.exit_code == 0, result.output

    fields = {
        "effort",
        "max_iterations",
        "parallelism",
        "max_pages_per_task",
        "max_results_per_query",
        "max_searches_per_task",
    }
    default, medium = (request.model_dump(include=fields) for request in seen)
    assert default == medium
