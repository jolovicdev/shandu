"""Shandu's mark: the Serbian tricolor, still in headers and waving in the intro."""

from __future__ import annotations

import math
import os
import time

from rich.console import Console
from rich.live import Live
from rich.style import Style
from rich.text import Text

from .. import __version__

RGB = tuple[int, int, int]
Canvas = list[list[RGB | None]]

# The national flag of Serbia: red over blue over white in equal bands.
_RED: RGB = (198, 54, 60)
_BLUE: RGB = (12, 64, 118)
_WHITE: RGB = (255, 255, 255)
_BANDS = (_RED, _BLUE, _WHITE)
# A hairline under the white band: on a light terminal the band is the
# background's own color and only this edge shows where it ends.
_EDGE: RGB = (196, 201, 209)
_POLE: RGB = (138, 144, 154)
_FINIAL: RGB = (208, 69, 76)
# Three pixels to a row of text and two rows to a band, so each band sits
# beside one line of the name; 9 by 6 is the flag's own 3:2.
_FLAG_WIDTH = 9
_FLAG_HEIGHT = 6
# The crests rise and fall a whole pixel; the canvas leaves room for them
# above and below, which keeps the bands on the text rows.
_AMPLITUDE = 1
_MARGIN = 2

_WORDMARK = " ".join("SHANDU")
_TAGLINE = "Deep research with citations"
_GAP = "  "
# The CLI's muted and pending grays, so the mark reads the same wherever it is.
_TAGLINE_COLOR = "#80868f"
_VERSION_COLOR = "#6e747e"

_FPS = 24
_DURATION = 1.5
# The wave dies down over the last stretch, so the intro ends on the mark at rest.
_SETTLE = 0.4


def still() -> Text:
    """The mark at rest, as the intro leaves it: each band beside one line of the name."""
    return frame(_DURATION)


def lines() -> tuple[str, str, str]:
    """The three lines that sit beside the flag's bands."""
    return _WORDMARK, _TAGLINE, f"v{__version__}"


def can_draw(console: Console) -> bool:
    """The flag is colored half blocks: uncolored they print as bare blocks,
    and a stream in a legacy encoding cannot carry them at all."""
    return (
        not console.no_color
        and console.color_system is not None
        and console.encoding.startswith("utf")
    )


def is_watched(console: Console) -> bool:
    """Whether a person sees this console as it draws, which animation and
    live redraws need. Rich's is_terminal also says yes under FORCE_COLOR,
    which CI sets to color its logs."""
    # is_interactive is fixed when the console is made; isatty follows
    # wherever its output goes now.
    if not console.is_interactive:
        return False
    isatty = getattr(console.file, "isatty", None)
    try:
        return isatty is not None and bool(isatty())
    except ValueError:
        # The file is closed.
        return False


def should_play(console: Console) -> bool:
    """A flag that waves only makes sense on a color terminal a person is watching."""
    if not (is_watched(console) and can_draw(console)):
        return False
    if os.environ.get("SHANDU_NO_INTRO", "").strip().lower() not in ("", "0", "false", "no"):
        return False
    width, height = console.size
    return width >= _columns() and height >= _rows() + 2


def play(console: Console, keep: bool = False) -> None:
    """Wave the flag while the name types in, for a second and a half.

    ``keep`` leaves the last frame on screen; otherwise the area is cleared
    for the command's own output.
    """
    frames = int(_DURATION * _FPS)
    started = time.perf_counter()
    with Live(console=console, auto_refresh=False, transient=not keep) as live:
        for index in range(frames + 1):
            live.update(frame(index / _FPS), refresh=True)
            delay = started + (index + 1) / _FPS - time.perf_counter()
            if delay > 0:
                time.sleep(delay)


def frame(t: float) -> Text:
    """The intro ``t`` seconds in: the flag waving, the name typing in beside it."""
    shown = min(len(_WORDMARK), max(0, int((t - 0.1) * 30)))
    typed = _TAGLINE[: max(0, int((t - 0.45) * 48))]
    # Rows 1 to 3 hold the red, blue and white bands.
    side = {
        1: Text(_WORDMARK[:shown], style="bold"),
        2: Text(typed, style=_TAGLINE_COLOR),
        3: Text(f"v{__version__}" if t >= 1.05 else "", style=_VERSION_COLOR),
    }
    # The edge would break into dashes where the wave lifts some columns, so
    # it is drawn once the flag is at rest.
    rows = _half_blocks(_waving_flag(t), edge=t >= _DURATION)
    for index, text in side.items():
        rows[index].append(_GAP)
        rows[index].append_text(text)
    # On a terminal narrower than the tagline each line is cut short, so the
    # name never wraps through the flag.
    return Text("\n", no_wrap=True, overflow="ellipsis").join(rows)


def _columns() -> int:
    return _FLAG_WIDTH + 1 + len(_GAP) + len(_TAGLINE)


def _rows() -> int:
    return (_FLAG_HEIGHT + 2 * _MARGIN + 1) // 2


def _waving_flag(t: float) -> Canvas:
    height = _FLAG_HEIGHT + 2 * _MARGIN
    canvas: Canvas = [[None] * (_FLAG_WIDTH + 1) for _ in range(height)]
    for y in range(height):
        canvas[y][0] = _FINIAL if y == 0 else _POLE
    calm = _smoothstep((_DURATION - t) / _SETTLE)
    for x in range(_FLAG_WIDTH):
        # The hoist is tied to the pole, so the wave grows toward the fly.
        reach = (x / (_FLAG_WIDTH - 1)) ** 0.8
        phase = 2 * math.pi * x / 8 - 6 * t
        # The cloth moves by whole pixels: at this size a pixel shared by two
        # bands shows as a third color.
        shift = round(_AMPLITUDE * calm * reach * math.sin(phase))
        # Light catches the crests and the troughs fall in shadow.
        shade = 1 + 0.24 * calm * reach * math.cos(phase)
        for y in range(_FLAG_HEIGHT):
            band = _BANDS[y * len(_BANDS) // _FLAG_HEIGHT]
            canvas[_MARGIN + shift + y][x + 1] = _scaled(band, shade)
    return canvas


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _half_blocks(canvas: Canvas, edge: bool = False) -> list[Text]:
    """Two pixel rows per text row: the upper half block takes the top pixel
    as its color and the bottom pixel as its background. With ``edge``, the
    white band's lower edge is drawn too."""
    rows: list[Text] = []
    for y in range(0, len(canvas) - 1, 2):
        below = canvas[y + 2] if y + 2 < len(canvas) else [None] * len(canvas[y])
        line = Text()
        for top, bottom, under in zip(canvas[y], canvas[y + 1], below):
            if top is None and bottom is None:
                line.append(" ")
            elif top is None:
                line.append("▄", _style(bottom, None))
            elif edge and top == bottom == _WHITE and under != _WHITE:
                # White ends at the foot of this row, where a one-eighth
                # block can draw its edge.
                line.append("▁", _style(_EDGE, _WHITE))
            else:
                line.append("▀", _style(top, bottom))
        rows.append(line)
    return rows


_STYLES: dict[tuple[RGB | None, RGB | None], Style] = {}


def _style(color: RGB | None, background: RGB | None) -> Style:
    key = (color, background)
    if key not in _STYLES:
        _STYLES[key] = Style(
            color=_hex(color) if color else None,
            bgcolor=_hex(background) if background else None,
        )
    return _STYLES[key]


def _hex(color: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _scaled(color: RGB, factor: float) -> RGB:
    red, green, blue = (min(255, max(0, round(channel * factor))) for channel in color)
    return red, green, blue
