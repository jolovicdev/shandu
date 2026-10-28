from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from shandu.cli import ShanduEngine, cli
from shandu.config import config
from shandu.contracts import ResearchRequest, ResearchRunResult
from shandu.services.run_store import RunNotFoundError, RunSummary


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
        list_runs=lambda: [],
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
    assert "473677ec " in result.stdout
    assert "completed" in result.stdout
    assert "How much memory do KV cache" in result.stdout


def _stored_run(run_id: str, status: str = "completed") -> RunSummary:
    return RunSummary(
        run_id=run_id,
        query="q",
        status=status,
        created_at="2026-09-29T10:00:00+00:00",
        citation_count=1,
        usd_spent=None,
        report_path=None,
    )


def test_show_finds_a_run_by_the_start_of_its_id(monkeypatch) -> None:
    loaded: list[str] = []

    def load_run(run_id):
        loaded.append(run_id)
        return ResearchRunResult(run_id=run_id, request=ResearchRequest(query="q"), report_markdown="# R")

    runs = [_stored_run("473677ec1d73484f919f35932bbdeddd"), _stored_run("9b2c0d1e5f6a4b7c8d9e0f1a2b3c4d5e")]
    engine = SimpleNamespace(list_runs=lambda: runs, load_run=load_run, close=lambda: None)
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(cli, ["show", "473677ec"])

    assert result.exit_code == 0, result.output
    assert loaded == ["473677ec1d73484f919f35932bbdeddd"]


@pytest.mark.parametrize(
    "status,resume_hint",
    [("cancelled", True), ("failed", True), ("running", False), ("completed", False)],
)
def test_show_names_the_status_of_a_run_without_a_report(monkeypatch, status, resume_hint) -> None:
    run_id = "847436363ebb4bc6b549712011aa89cf"
    engine = SimpleNamespace(
        list_runs=lambda: [_stored_run(run_id, status)],
        load_run=lambda run_id: ResearchRunResult(
            run_id=run_id, request=ResearchRequest(query="q"), report_markdown=""
        ),
        close=lambda: None,
    )
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(cli, ["show", "84743636"])

    lines = [line.split() for line in result.stdout.splitlines()]
    assert result.exit_code == 0, result.output
    assert ["status", status] in lines
    assert "This run has no report yet." in result.stdout
    assert "iterations" not in result.stdout
    assert ("Continue with: shandu resume 84743636" in result.stdout) is resume_hint


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_list_refuses_a_limit_below_one(monkeypatch, limit) -> None:
    engine = SimpleNamespace(list_runs=lambda: [_stored_run("473677ec" + "0" * 24)], close=lambda: None)
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(cli, ["list", "--limit", limit])

    assert result.exit_code == 2
    assert "Invalid value for '--limit'" in result.stderr


def test_a_run_id_prefix_shared_by_two_runs_asks_for_more(monkeypatch) -> None:
    runs = [_stored_run("473677ec1d73484f919f35932bbdeddd"), _stored_run("4736aaaa1d73484f919f35932bbdeddd")]
    engine = SimpleNamespace(list_runs=lambda: runs, close=lambda: None)
    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(lambda cls: engine))

    result = CliRunner(mix_stderr=False).invoke(cli, ["show", "4736"])

    assert result.exit_code != 0
    assert "matches 2 runs" in result.stderr


def _store_engine(monkeypatch, run_ids: list[str]) -> SimpleNamespace:
    loaded: list[str] = []

    def load_run(run_id):
        loaded.append(run_id)
        if run_id not in run_ids:
            raise RunNotFoundError(run_id)
        return ResearchRunResult(
            run_id=run_id, request=ResearchRequest(query="q"), report_markdown="# R"
        )

    engine = SimpleNamespace(
        list_runs=lambda: [_stored_run(run_id) for run_id in run_ids],
        load_run=load_run,
        inspect_run=lambda run_id: {"exists": run_id in run_ids, "run_id": run_id},
        close=lambda: None,
        loaded=loaded,
    )
    monkeypatch.setattr(
        ShanduEngine, "from_config", classmethod(lambda cls, writer_model=None: engine)
    )
    return engine


@pytest.mark.parametrize("command", ["show", "inspect", "resume", "report"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_run_id_is_refused_instead_of_matching_every_run(
    monkeypatch, command, blank
) -> None:
    engine = _store_engine(monkeypatch, ["473677ec1d73484f919f35932bbdeddd"])

    result = CliRunner(mix_stderr=False).invoke(cli, [command, blank])

    assert result.exit_code == 2
    assert "Give a run id" in result.stderr
    assert engine.loaded == []


@pytest.mark.parametrize(
    "run_ids,message",
    [
        (
            ["4023c8fb" + "0" * 24, "4e4d5d31" + "0" * 24, "443b0221" + "0" * 24],
            "Run id 4 matches 3 runs: 4023c8fb, 4e4d5d31, 443b0221. Type more characters.",
        ),
        (
            [f"4{index}a" + "0" * 29 for index in range(7)],
            "Run id 4 matches 7 runs: 40a00000, 41a00000, 42a00000, 43a00000, 44a00000"
            " and 2 more. Type more characters.",
        ),
    ],
)
def test_an_ambiguous_run_id_names_up_to_five_candidates(monkeypatch, run_ids, message) -> None:
    _store_engine(monkeypatch, run_ids)

    result = CliRunner(mix_stderr=False).invoke(cli, ["show", "4"])

    assert result.exit_code == 1
    assert message in result.stderr


@pytest.mark.parametrize("command", ["show", "inspect", "resume", "report"])
def test_an_unknown_run_id_exits_1_and_points_to_list(monkeypatch, command) -> None:
    _store_engine(monkeypatch, ["473677ec1d73484f919f35932bbdeddd"])

    result = CliRunner(mix_stderr=False).invoke(cli, [command, "zzzz"])

    assert result.exit_code == 1
    assert "No run starts with zzzz. See `shandu list`." in result.stderr


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


def _stub_run_engine(monkeypatch) -> dict[str, object]:
    seen: dict[str, object] = {"engines": 0}

    async def run(request, progress_callback=None):
        seen["request"] = request
        return ResearchRunResult(run_id="r1", request=request, report_markdown="Body")

    engine = SimpleNamespace(
        run=run, report_path=lambda run_id: Path("missing.md"), close=lambda: None
    )

    def from_config(cls):
        seen["engines"] += 1
        return engine

    monkeypatch.setattr(ShanduEngine, "from_config", classmethod(from_config))
    return seen


def test_run_joins_unquoted_words_into_one_question(monkeypatch) -> None:
    seen = _stub_run_engine(monkeypatch)

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "What", "caused", "the", "outage?"])

    assert result.exit_code == 0, result.output
    assert seen["request"].query == "What caused the outage?"


@pytest.mark.parametrize("args", [["run"], ["run", ""], ["run", "  ", " "]])
def test_run_without_a_question_is_a_usage_error_that_starts_nothing(monkeypatch, args) -> None:
    seen = _stub_run_engine(monkeypatch)

    result = CliRunner(mix_stderr=False).invoke(cli, args)

    assert result.exit_code == 2
    assert 'Give a question to research, like: shandu run "' in result.stderr
    assert seen["engines"] == 0


@pytest.mark.parametrize(
    "flag,value,message",
    [
        ("--max-iterations", "9", "Invalid value for '--max-iterations': 9 is not in the range 1<=x<=8."),
        ("--parallelism", "0", "Invalid value for '--parallelism': 0 is not in the range 1<=x<=8."),
        ("--max-pages-per-task", "11", "Invalid value for '--max-pages-per-task'"),
        ("--max-results-per-query", "21", "Invalid value for '--max-results-per-query'"),
        ("--max-cost", "0", "Invalid value for '--max-cost'"),
        ("--max-cost", "inf", "Invalid value for '--max-cost': give an amount in USD, like 0.50."),
        ("--max-cost", "nan", "Invalid value for '--max-cost': give an amount in USD, like 0.50."),
        ("--since", "31/01/2025", "Invalid value for '--since': Use YYYY-MM-DD, like 2025-01-31."),
    ],
)
def test_run_names_the_flag_when_a_value_is_out_of_range(monkeypatch, flag, value, message) -> None:
    seen = _stub_run_engine(monkeypatch)

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "q", flag, value])

    assert result.exit_code == 2
    assert message in result.stderr
    assert seen["engines"] == 0


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], "2 iterations · 3 subagents"),
        (["--max-iterations", "1", "--parallelism", "1"], "1 iteration · 1 subagent"),
    ],
)
def test_run_header_counts_iterations_and_subagents(monkeypatch, flags, expected) -> None:
    _stub_run_engine(monkeypatch)

    result = CliRunner(mix_stderr=False).invoke(cli, ["run", "q", *flags])

    assert expected in " ".join(result.stdout.split())


def test_run_passes_a_valid_cost_cap_and_start_day_to_the_request(monkeypatch) -> None:
    seen = _stub_run_engine(monkeypatch)

    result = CliRunner(mix_stderr=False).invoke(
        cli, ["run", "q", "--max-cost", "0.5", "--since", "2025-01-31", "--max-iterations", "8"]
    )

    assert result.exit_code == 0, result.output
    request = seen["request"]
    assert (request.max_cost_usd, request.since, request.max_iterations) == (0.5, date(2025, 1, 31), 8)
