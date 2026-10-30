from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.text import Text

from shandu.ui import intro


class _Screen(io.StringIO):
    """A terminal a person is watching, in the encoding it reports."""

    def __init__(self, encoding: str = "utf-8") -> None:
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding

    def isatty(self) -> bool:
        return True


def _terminal(**overrides) -> Console:
    settings = {"width": 120, "height": 40, "color_system": "truecolor", "file": _Screen()}
    settings.update(overrides)
    return Console(force_terminal=True, force_interactive=True, **settings)


def test_intro_plays_only_on_a_color_terminal_tall_enough_unless_turned_off(monkeypatch) -> None:
    monkeypatch.delenv("SHANDU_NO_INTRO", raising=False)

    assert intro.should_play(_terminal())
    assert not intro.should_play(Console(file=io.StringIO(), width=120, height=40))
    assert not intro.should_play(_terminal(no_color=True))
    assert not intro.should_play(_terminal(height=6))
    assert not intro.should_play(_terminal(file=_Screen("ascii")))

    monkeypatch.setenv("SHANDU_NO_INTRO", "0")
    assert intro.should_play(_terminal())
    monkeypatch.setenv("SHANDU_NO_INTRO", "1")
    assert not intro.should_play(_terminal())


@pytest.mark.parametrize("width", [30, 39, 40, 80])
def test_the_intro_plays_only_where_every_frame_fits(monkeypatch, width) -> None:
    monkeypatch.delenv("SHANDU_NO_INTRO", raising=False)
    widest = max(
        line.cell_len for t in (0.0, 0.5, 1.0, 1.5) for line in intro.frame(t).split("\n")
    )

    assert intro.should_play(_terminal(width=width)) == (widest <= width)


def test_forced_color_alone_does_not_play_the_intro_into_a_log(monkeypatch) -> None:
    monkeypatch.delenv("SHANDU_NO_INTRO", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    log = Console(file=io.StringIO(), width=120, height=40, color_system="truecolor")

    assert log.is_terminal
    assert not intro.should_play(log)


def _flag(frame: Text) -> list[tuple[str, str, str]]:
    """Each flag cell of a frame as its glyph, color and background."""
    cells = []
    for row in frame.split("\n"):
        styles = {span.start: span.style for span in row.spans}
        for column in range(1, 10):
            style = styles.get(column)
            colors = (style.color.name, style.bgcolor.name if style.bgcolor else "") if style else ("", "")
            cells.append((row.plain[column], *colors))
    return cells


def test_the_flag_waves_and_comes_to_rest_on_its_three_bands(monkeypatch) -> None:
    shown: list[Text] = []

    class Screen:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> Screen:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def update(self, renderable: Text, refresh: bool = False) -> None:
            shown.append(renderable)

    monkeypatch.setattr(intro, "Live", Screen)
    monkeypatch.setattr(intro.time, "sleep", lambda _: None)
    intro.play(_terminal())

    rest = _flag(shown[-1])
    assert rest[9:18] == [("▀", "#c6363c", "#c6363c")] * 9
    assert rest[18:27] == [("▀", "#0c4076", "#0c4076")] * 9
    # The white band draws its lower edge, which is all that shows of it on
    # a light terminal.
    assert rest[27:36] == [("▁", "#c4c9d1", "#ffffff")] * 9
    assert not any("▁" in frame.plain for frame in shown[:-1])
    assert any(_flag(frame) != rest for frame in shown)
    assert _flag(intro.still()) == rest
