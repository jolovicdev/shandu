from __future__ import annotations

from rich.console import Console

from shandu.ui.rich_frontend import ShanduUI


def test_inspect_panel_shows_run_id_and_summary() -> None:
    console = Console(record=True, width=200)
    ui = ShanduUI(console=console)
    payload = {
        "run_id": "run-abc-123",
        "status": "completed",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-02",
        "output_json": {
            "run_stats": {"iterations": 2, "evidence_count": 5, "citation_count": 3}
        },
        "events": [{"type": "llm.completed", "payload": {"x": "y" * 200}}] * 50,
    }

    console.print(ui.inspect_panel(payload))
    text = console.export_text()

    assert "run-abc-123" in text
    assert "iterations" in text
