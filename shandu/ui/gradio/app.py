from __future__ import annotations

from pathlib import Path

from ...config import config
from .layout import build_gui
from .theme import CSS, build_theme


def launch_gui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
    auth: tuple[str, str] | None = None,
) -> None:
    demo = build_gui()
    export_dir = str(
        Path(str(config.get("runtime", "storage_dir", ".blackgeorge"))) / "exports"
    )
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
        theme=build_theme(),
        css=CSS,
        footer_links=["gradio", "settings"],
        auth=auth,
        allowed_paths=[export_dir],
    )
