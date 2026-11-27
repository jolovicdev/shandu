from __future__ import annotations

import pytest

from shandu.agents.auditor import _AuditPayload
from shandu.agents.lead import _ClarifierPayload, _PlanPayload, _SynthesisPayload
from shandu.agents.search_subagent import _ExtractionPayload, _SelectionPayload

_OWN_NESTED = ("_PlannedTask", "_UnitVerdict", "_MarkerVerdict")


@pytest.mark.parametrize(
    "schema",
    [
        _PlanPayload,
        _SynthesisPayload,
        _ClarifierPayload,
        _ExtractionPayload,
        _SelectionPayload,
        _AuditPayload,
    ],
)
def test_schemas_a_model_fills_require_every_field(schema) -> None:
    document = schema.model_json_schema()
    nested = [document.get("$defs", {}).get(name) for name in _OWN_NESTED]

    for model in [document, *(item for item in nested if item)]:
        assert set(model["required"]) == set(model["properties"])


def test_lead_normalize_tasks_unique_ids_and_fallback_url() -> None:
    from shandu.agents.lead import LeadAgent
    from shandu.contracts import ResearchRequest, SubagentTask

    tasks = LeadAgent._normalize_tasks(
        [
            SubagentTask(task_id="iter_1_task_2", focus="First"),
            SubagentTask(task_id="iter_1_task_2", focus="Second"),
        ],
        request=ResearchRequest(query="q", parallelism=4),
        iteration=0,
    )
    assert len({t.task_id for t in tasks}) == 2


def test_extraction_payload_requires_claim_sub_question_in_schema() -> None:
    schema = _ExtractionPayload.model_json_schema()
    claim_schema = next(iter(schema["$defs"].values()))
    assert "sub_question" in claim_schema["required"]
