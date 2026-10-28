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
