from __future__ import annotations

import re
from collections.abc import Iterable

from ..contracts import (
    Claim,
    EvidenceRecord,
    Finding,
    ResearchBrief,
    SubQuestionCoverage,
)

# A page at or above this credibility counts toward coverage; it admits
# technical references, journalism, and better, and keeps blogs, forums,
# and marketing out.
_CREDIBLE = 0.6
# Two claims whose word sets overlap this much state the same fact.
_SAME_CLAIM_OVERLAP = 0.8
_WORD = re.compile(r"[0-9a-z]+(?:\.[0-9]+)?")
# A shorter quote appears in almost any page, so it proves nothing.
_MIN_QUOTE_WORDS = 4


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def verified_claims(
    claims: Iterable[Claim], page_text: str, sub_question_count: int
) -> list[Claim]:
    """Keep claims whose quote appears verbatim in the text the model read.

    Matching ignores whitespace and case only, so a paraphrased quote fails,
    and a quote under four words is rejected. A sub-question index outside
    the brief is cleared, not trusted.
    """
    haystack = _normalize(page_text)
    kept: list[Claim] = []
    for claim in claims:
        quote = " ".join(claim.quote.split())
        if (
            not claim.statement.strip()
            or len(quote.split()) < _MIN_QUOTE_WORDS
            or _normalize(quote) not in haystack
        ):
            continue
        index = claim.sub_question
        kept.append(
            claim.model_copy(
                update={
                    "statement": claim.statement.strip(),
                    "quote": quote,
                    "sub_question": index
                    if index is not None and 1 <= index <= sub_question_count
                    else None,
                }
            )
        )
    return kept


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.casefold()))


def build_findings(evidence: Iterable[EvidenceRecord]) -> list[Finding]:
    """Merge claims across pages into one finding per distinct statement."""
    findings: list[Finding] = []
    words: list[frozenset[str]] = []
    for record in evidence:
        confidence = record.credibility_score if record.credibility_score is not None else 0.4
        for claim in record.claims:
            claim_words = _words(claim.statement)
            match = next(
                (
                    index
                    for index, existing in enumerate(words)
                    if claim_words
                    and len(claim_words & existing) / len(claim_words | existing)
                    >= _SAME_CLAIM_OVERLAP
                ),
                None,
            )
            if match is None:
                findings.append(Finding(statement=claim.statement))
                words.append(claim_words)
                match = len(findings) - 1
            finding = findings[match]
            if record.evidence_id not in finding.evidence_ids:
                finding.evidence_ids.append(record.evidence_id)
            if claim.quote not in finding.quotes:
                finding.quotes.append(claim.quote)
            if record.domain and record.domain not in finding.domains:
                finding.domains.append(record.domain)
            if claim.sub_question is not None and claim.sub_question not in finding.sub_questions:
                finding.sub_questions.append(claim.sub_question)
            finding.confidence = max(finding.confidence, confidence)
    return findings


def coverage_matrix(
    brief: ResearchBrief, evidence: list[EvidenceRecord]
) -> list[SubQuestionCoverage]:
    """Rate each sub-question by how many independent credible sites support it."""
    findings = build_findings(evidence)
    credibility = {
        record.evidence_id: record.credibility_score or 0.0 for record in evidence
    }
    domains = {record.evidence_id: record.domain or record.requested_url for record in evidence}
    matrix: list[SubQuestionCoverage] = []
    for index, question in enumerate(brief.sub_questions, start=1):
        supporting = [finding for finding in findings if index in finding.sub_questions]
        credible_sites = {
            domains[evidence_id]
            for finding in supporting
            for evidence_id in finding.evidence_ids
            if credibility.get(evidence_id, 0.0) >= _CREDIBLE
        }
        if not supporting:
            strength = "none"
        elif not credible_sites:
            strength = "weak"
        elif len(credible_sites) == 1:
            strength = "moderate"
        else:
            strength = "strong"
        matrix.append(
            SubQuestionCoverage(
                index=index,
                question=question,
                strength=strength,
                findings=len(supporting),
                credible_sources=len(credible_sites),
            )
        )
    return matrix
