from __future__ import annotations

from click.testing import CliRunner

from shandu.cli import cli


def test_gui_share_without_auth_exits_with_error(monkeypatch) -> None:
    def fake_launch_gui(**kwargs) -> None:
        del kwargs

    monkeypatch.setattr("shandu.ui.gradio_app.launch_gui", fake_launch_gui)
    result = CliRunner().invoke(cli, ["gui", "--share"])

    assert result.exit_code != 0
    assert "--auth" in result.output


def test_gui_auth_reaches_launch(monkeypatch) -> None:
    captured: dict = {}

    def fake_launch_gui(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("shandu.ui.gradio_app.launch_gui", fake_launch_gui)
    result = CliRunner().invoke(cli, ["gui", "--auth", "owner:s3cret"])

    assert result.exit_code == 0
    assert captured.get("auth") == ("owner", "s3cret")


def test_gui_non_loopback_without_auth_warns(monkeypatch) -> None:
    def fake_launch_gui(**kwargs) -> None:
        del kwargs

    monkeypatch.setattr("shandu.ui.gradio_app.launch_gui", fake_launch_gui)
    result = CliRunner().invoke(cli, ["gui", "--host", "0.0.0.0"])

    assert result.exit_code == 0
    assert "auth" in result.output.lower()


def test_launch_gui_passes_auth_to_blocks_launch(monkeypatch) -> None:
    import shandu.ui.gradio.app as app_module

    launched: dict = {}

    class FakeDemo:
        def launch(self, **kwargs) -> None:
            launched.update(kwargs)

    monkeypatch.setattr(app_module, "build_gui", lambda: FakeDemo())
    app_module.launch_gui(auth=("owner", "s3cret"))

    assert launched.get("auth") == ("owner", "s3cret")
