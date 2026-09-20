from __future__ import annotations

from click.testing import CliRunner

from shandu.cli import cli
from shandu.config import config


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
