from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from shandu.cli import ShanduEngine, cli
from shandu.config import config
from shandu.contracts import ResearchRequest, ResearchRunResult
from shandu.services.run_store import RunSummary


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = config.get("runtime", "storage_dir")
    config.set("runtime", "storage_dir", str(tmp_path / "storage"))
    try:
        yield
    finally:
        config.set("runtime", "storage_dir", previous)


def test_report_rewrites_with_writer_model_and_detail_level(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    async def rewrite_report(run_id, overrides=None, progress_callback=None):
        seen["rewrite"] = (run_id, overrides)
        return ResearchRunResult(
            run_id=run_id, request=ResearchRequest(query="q"), report_markdown="Rewritten body"
        )

    engine = SimpleNamespace(
        load_run=lambda run_id: ResearchRunResult(
            run_id=run_id, request=ResearchRequest(query="q"), report_markdown="# Old"
        ),
        rewrite_report=rewrite_report,
        report_path=lambda run_id: tmp_path / "missing.md",
        close=lambda: None,
    )

    def from_config(cls, writer_model=None):
        seen["writer_model"] = writer_model
        return engine

    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(from_config))

    result = CliRunner(mix_stderr=False).invoke(
        cli,
        [
            "report",
            "r1",
            "--model",
            "openai/gpt-x",
            "--detail-level",
            "concise",
            "--no-audit",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "writer_model": "openai/gpt-x",
        "rewrite": ("r1", {"detail_level": "concise", "audit": False}),
    }
    assert "Rewritten body" in result.stdout


def test_list_prints_stored_runs(monkeypatch) -> None:
    runs = [
        RunSummary(
            run_id="473677ec1d73484f919f35932bbdeddd",
            query="How much memory do KV cache compression methods save at 128k context?",
            status="completed",
            created_at="2026-09-29T10:00:00+00:00",
            citation_count=12,
            usd_spent=0.33,
            report_path=None,
        )
    ]
    engine = SimpleNamespace(list_runs=lambda: runs, close=lambda: None)
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(cli, ["list"])

    assert result.exit_code == 0, result.output
    assert "473677ec1d73484f919f35932bbdeddd" in result.stdout
    assert "completed" in result.stdout
    assert "How much" in result.stdout


def test_legacy_storage_in_cwd_prints_notice(monkeypatch, tmp_path) -> None:
    (tmp_path / ".blackgeorge").mkdir()
    engine = SimpleNamespace(list_runs=lambda: [], close=lambda: None)
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(cli, ["list"])

    assert result.exit_code == 0, result.output
    assert "legacy storage" in result.stderr


def test_run_clarify_sends_answered_questions_with_request(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def clarifying_questions(query):
        return ["Which region?", "Which year?"]

    async def run(request, progress_callback=None):
        seen["clarifications"] = request.clarifications
        return ResearchRunResult(run_id="r1", request=request, report_markdown="Body")

    engine = SimpleNamespace(
        clarifying_questions=clarifying_questions,
        run=run,
        report_path=lambda run_id: Path("missing.md"),
        close=lambda: None,
    )
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(
        cli, ["run", "q", "--clarify"], input="EU\n\n"
    )

    assert result.exit_code == 0, result.output
    assert seen["clarifications"] == ["Which region? EU"]
