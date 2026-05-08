from __future__ import annotations

from .layout import build_gui
from .theme import CSS, build_theme


def launch_gui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
) -> None:
    demo = build_gui()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
        theme=build_theme(),
        css=CSS,
        footer_links=["gradio", "settings"],
    )
