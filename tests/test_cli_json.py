from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from shandu.cli import ShanduEngine, cli
from shandu.config import config
from shandu.contracts import ResearchRequest, ResearchRunResult, RunEvent


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path):
    previous = config.get("runtime", "storage_dir")
    config.set("runtime", "storage_dir", str(tmp_path / "storage"))
    try:
        yield
    finally:
        config.set("runtime", "storage_dir", previous)


def _stub_engine(monkeypatch, emit_event: bool = True) -> None:
    def fake_run_sync(request, progress_callback=None):
        if emit_event and progress_callback is not None:
            progress_callback(
                RunEvent(stage="search", message="diagnostic event line")
            )
        return ResearchRunResult(
            run_id="run-1",
            request=request if isinstance(request, ResearchRequest) else ResearchRequest(query="q"),
            report_markdown="# Report\n\nBody.",
        )

    engine = SimpleNamespace(run_sync=fake_run_sync, close=lambda: None)
    monkeypatch.setattr(
        ShanduEngine, "from_config", classmethod(lambda cls: engine)
    )


def test_run_json_output_stdout_parses_as_json(monkeypatch) -> None:
    _stub_engine(monkeypatch)

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "q", "--json-output"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "run-1"
    assert payload["report_markdown"] == "# Report\n\nBody."
    assert "diagnostic event line" not in result.stdout
    assert "diagnostic event line" in result.stderr


def test_run_default_output_keeps_diagnostics_on_stdout(monkeypatch) -> None:
    _stub_engine(monkeypatch)

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "q"])

    assert result.exit_code == 0, result.output
    assert "diagnostic event line" in result.stdout


def test_run_without_output_persists_report_export(monkeypatch, tmp_path) -> None:
    _stub_engine(monkeypatch, emit_event=False)

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "q"])

    assert result.exit_code == 0, result.output
    export = tmp_path / "storage" / "exports" / "run_1.md"
    assert export.exists()
    assert export.read_text(encoding="utf-8") == "# Report\n\nBody."
    assert "Report saved to" in result.stdout
    assert "run_1.md" in result.stdout


def test_run_depth_policy_flag_reaches_request(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_run_sync(request, progress_callback=None):
        del progress_callback
        seen["depth_policy"] = request.depth_policy
        return ResearchRunResult(
            run_id="run-1", request=request, report_markdown="# Report"
        )

    engine = SimpleNamespace(run_sync=fake_run_sync, close=lambda: None)
    monkeypatch.setattr(
        ShanduEngine, "from_config", classmethod(lambda cls: engine)
    )

    result = CliRunner(mix_stderr=False).invoke(
        cli, ["run", "q", "--depth-policy", "fixed"]
    )

    assert result.exit_code == 0, result.output
    assert seen["depth_policy"] == "fixed"
