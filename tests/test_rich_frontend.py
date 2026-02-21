from __future__ import annotations

from rich.console import Console

from shandu.contracts import ResearchRequest
from shandu.ui.rich_frontend import ShanduUI


def test_result_panels_render() -> None:
    console = Console(record=True, width=140)
    ui = ShanduUI(console=console)
    request = ResearchRequest(query="q")
    snapshot = ui.new_snapshot(request, model="deepseek/deepseek-chat")
    console.print(ui.dashboard(snapshot))
    output = console.export_text()
    assert "Control Plane" in output
