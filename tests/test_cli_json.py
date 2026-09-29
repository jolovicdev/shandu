from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from shandu.cli import ShanduEngine, cli
from shandu.config import config
from shandu.contracts import ResearchRequest, ResearchRunResult, RunEvent

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path):
    previous = config.get("runtime", "storage_dir")
    config.set("runtime", "storage_dir", str(tmp_path / "storage"))
    try:
        yield
    finally:
        config.set("runtime", "storage_dir", previous)


def _fake_engine(run) -> SimpleNamespace:
    runs_dir = Path(str(config.get("runtime", "storage_dir"))) / "runs"
    return SimpleNamespace(
        run=run,
        close=lambda: None,
        report_path=lambda run_id: runs_dir / run_id / "report.md",
    )


def _stub_engine(monkeypatch, emit_event: bool = True) -> None:
    async def fake_run(request, progress_callback=None):
        if emit_event and progress_callback is not None:
            progress_callback(
                RunEvent(stage="search", message="diagnostic event line")
            )
        report = engine.report_path("run-1")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# Report\n\nBody.", encoding="utf-8")
        return ResearchRunResult(
            run_id="run-1",
            request=request if isinstance(request, ResearchRequest) else ResearchRequest(query="q"),
            report_markdown="# Report\n\nBody.",
        )

    engine = _fake_engine(fake_run)
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


def test_run_without_output_names_saved_report(monkeypatch) -> None:
    _stub_engine(monkeypatch, emit_event=False)

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "q"])

    assert result.exit_code == 0, result.output
    assert "Report saved to" in result.stdout
    assert "report.md" in result.stdout


def test_run_depth_policy_flag_reaches_request(monkeypatch) -> None:
    seen: dict[str, str] = {}

    async def fake_run(request, progress_callback=None):
        del progress_callback
        seen["depth_policy"] = request.depth_policy
        return ResearchRunResult(
            run_id="run-1", request=request, report_markdown="# Report"
        )

    engine = _fake_engine(fake_run)
    monkeypatch.setattr(
        ShanduEngine, "from_config", classmethod(lambda cls: engine)
    )

    result = CliRunner(mix_stderr=False).invoke(
        cli, ["run", "q", "--depth-policy", "fixed"]
    )

    assert result.exit_code == 0, result.output
    assert seen["depth_policy"] == "fixed"


def test_run_ctrl_c_cancels_and_exits_130(monkeypatch) -> None:
    import concurrent.futures

    class FakeFuture:
        def __init__(self) -> None:
            self.cancel_calls = 0
            self.results = 0

        def cancel(self) -> bool:
            self.cancel_calls += 1
            return True

        def result(self, timeout=None):
            del timeout
            self.results += 1
            if self.results == 1:
                raise KeyboardInterrupt
            raise concurrent.futures.CancelledError

    future = FakeFuture()

    async def fake_run(request, progress_callback=None):
        del request, progress_callback
        raise AssertionError("must go through the runner")

    def fake_submit(coro):
        coro.close()
        return future

    engine = _fake_engine(fake_run)
    monkeypatch.setattr(
        ShanduEngine, "from_config", classmethod(lambda cls: engine)
    )
    monkeypatch.setattr(
        "shandu.cli.get_async_runner",
        lambda: SimpleNamespace(submit=fake_submit),
    )

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "q"])

    assert result.exit_code == 130, result.output
    assert future.cancel_calls == 1


def test_info_reports_provider_managed_credentials() -> None:
    previous_model = config.get("api", "model")
    previous_env = config.get("api", "api_key_env")
    config.set("api", "model", "bedrock/anthropic.claude-3-sonnet")
    config.set("api", "api_key_env", "")
    try:
        result = CliRunner().invoke(cli, ["info"])
    finally:
        config.set("api", "model", previous_model)
        config.set("api", "api_key_env", previous_env)

    assert result.exit_code == 0
    assert "provider-managed" in result.output


def test_cli_loads_dotenv_from_working_directory(tmp_path) -> None:
    (tmp_path / ".env").write_text("SHANDU_MODEL=probe/x\n", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("from shandu.cli import cli\ncli()\n", encoding="utf-8")
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SHANDU_MODEL", None)
    env.pop("OPENAI_MODEL_NAME", None)

    completed = subprocess.run(
        [sys.executable, str(runner), "info"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0
    assert "probe/x" in completed.stdout
