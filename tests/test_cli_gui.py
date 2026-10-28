from __future__ import annotations

import socket

import gradio as gr
from click.testing import CliRunner

from shandu.cli import cli


def _tiny_gui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("x")
    return demo


def test_gui_on_a_busy_port_exits_1_and_suggests_the_next_port(monkeypatch) -> None:
    monkeypatch.setattr("shandu.ui.gradio.app.build_gui", _tiny_gui)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        result = CliRunner(mix_stderr=False).invoke(cli, ["gui", "--port", str(port)])

    assert result.exit_code == 1
    assert result.stderr.strip() == (
        f"Error: Port {port} is in use. Shandu may already be running at "
        f"http://127.0.0.1:{port}. Use `shandu gui --port {port + 1}`."
    )


def test_gui_reports_other_launch_failures_without_claiming_a_busy_port(monkeypatch) -> None:
    def failing_launch(**kwargs) -> None:
        raise OSError("certificate file unreadable")

    monkeypatch.setattr("shandu.ui.gradio_app.launch_gui", failing_launch)

    result = CliRunner(mix_stderr=False).invoke(cli, ["gui"])

    assert "GUI runtime error: certificate file unreadable" in result.stdout
    assert "in use" not in result.stderr
