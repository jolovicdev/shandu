from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from shandu import __version__
from shandu.cli import cli


def _help(*args: str) -> str:
    result = CliRunner().invoke(cli, list(args), terminal_width=120)
    assert result.exit_code == 0, result.output
    return result.output


def test_help_describes_the_tool_and_lists_run_first() -> None:
    text = _help("--help")

    names = [line.split()[0] for line in text.split("Commands:\n", 1)[1].splitlines() if line.strip()]

    assert "Deep research from the terminal." in text
    assert names[0] == "run"
    assert sorted(names) == sorted(cli.commands)


def test_version_flag_prints_the_package_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"shandu, version {__version__}"


@pytest.mark.parametrize("command", [[], ["run"], ["list"]])
def test_h_is_the_short_form_of_help(command) -> None:
    assert _help(*command, "-h") == _help(*command, "--help")


def test_help_uses_up_to_100_columns_on_a_wide_terminal(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "140")
    group = cli.make_context("shandu", ["--help"], resilient_parsing=True)
    run = cli.commands["run"].make_context("run", [], parent=group, resilient_parsing=True)

    assert group.make_formatter().width == run.make_formatter().width == 98


@pytest.mark.parametrize("command", ["run", "resume", "report", "aisearch"])
def test_output_option_asks_for_a_file(command) -> None:
    assert "--output FILE" in _help(command, "--help")


def test_every_option_has_help_text() -> None:
    bare = [
        f"{name} {param.opts[0]}"
        for name, command in cli.commands.items()
        for param in command.params
        if isinstance(param, click.Option) and not param.help
    ]

    assert bare == []


def test_run_help_shows_the_default_of_each_option_that_has_one() -> None:
    text = _help("run", "--help")

    for option in ("--effort", "--detail-level", "--depth-policy", "--researcher"):
        block = text.split(option, 1)[1].split("\n  --", 1)[0]
        assert "[default:" in block, option
