from __future__ import annotations

from ...services.run_store import default_run_store
from .layout import build_gui
from .theme import CSS, HEAD, build_theme


def launch_gui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
    auth: tuple[str, str] | None = None,
) -> None:
    demo = build_gui()
    runs_dir = str(default_run_store().root)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
        theme=build_theme(),
        css=CSS,
        head=HEAD,
        # Gradio's own settings link would sit beside Shandu's Settings panel.
        footer_links=["gradio"],
        auth=auth,
        allowed_paths=[runs_dir],
    )
