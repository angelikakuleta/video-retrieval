"""The picture of one variant's ranking: rows of three frames, nothing else.

What this module draws is a FIGURE, not a page. No margins, no query text, no
column captions, no page number -- whatever surrounds the figure is added where
it is used. The image carries only what has to be drawn: the frames, and beside
each row the position, the recording and the time interval.

The correct fragment is marked by one thing only: the word ``POPRAWNE`` in its
gutter, set bold and green. Every other line there is plain black and every frame
carries the same hairline border -- a coloured frame was tried and dropped,
because on a television still it competes with the picture instead of pointing at
it. The bold weight is what carries the mark into a grey print, where the green
reads only as a darker grey.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.evaluation import figures

# ---------------------------------------------------------- the figure, in mm
#: Width every figure is drawn at. One number for all of them, so no example
#: comes out at a different scale than another.
TEXT_W = 160.0

# ----------------------------------------------------------- the grid, in mm
#: Width of the gutter, and with it the width of the frames: the three pictures
#: take whatever the column has left.
#:
#: MEASURED, and measured AT THE OUTPUT RESOLUTION. The longest line the format
#: can produce is an interval of eighteen characters, which the font renders
#: 24.38 mm wide at :data:`GUTTER_PT` and 300 dpi. Of that, 23.9 mm is INKED:
#: the closing bracket carries half a millimetre of right side bearing that
#: nothing is drawn in. The gutter is the inked width plus one
#: :data:`COL_GAP`, so the white the eye sees between the label and the first
#: frame is the same as the white between two frames -- measuring from the
#: metric instead would make that gap half a millimetre too wide.
#:
#: Both halves of that sentence were learned the hard way. Guessing the width
#: from a character count left 4 mm of slack. Measuring it at 100 dpi gave
#: 22.86 mm -- font hinting compresses glyph advances at small sizes, so the
#: same 6.5 pt line is 1.5 mm narrower there than in the file actually saved --
#: and the label then ran into the picture. The test measures at
#: :data:`src.evaluation.figures.DPI` for that reason.
#:
#: The label is deliberately small -- the proportion a caption has beside the
#: picture it names. That is what lets the gutter be narrow and the frames
#: large.
GUTTER = 25.4
COL_GAP = 1.5
ROW_GAP = 1.2
FRAME_W = (TEXT_W - GUTTER - 2 * COL_GAP) / 3          # 43.87
FRAME_H = FRAME_W * 9 / 16                             # 24.68

#: line height inside the gutter, and its type size in points. Four lines of
#: 3.1 mm come to 12.4 mm, which sits inside the 24.4 mm height of a frame.
LINE_H = 3.1
GUTTER_PT = 6.5

# ------------------------------------------------------------------ colours
INK = "#15191B"
INK_SOFT = "#5C6A70"
FRAME_EDGE = "#AFBAB6"
#: Colour of the word that marks the correct fragment, and the only colour on
#: the figure. Everything else in the gutter is plain black, so the word is the
#: one thing the eye is led to; the frames all carry the same neutral hairline,
#: because a coloured border on a television still competes with the picture
#: instead of pointing at it.
#:
#: Since the colour carries the mark alone, the bold weight is its second
#: encoding for a grey print, where the green reads only as a darker grey.
VERDICT_COLOUR = "#2C7A4B"
PLACEHOLDER = "#D3DAD7"

MONO = ["DejaVu Sans Mono", "Consolas", "monospace"]

#: the word that carries correctness when the colour is gone
VERDICT = "POPRAWNE"

def output_dir() -> Path:
    """Where the figures are written. Outside version control, on purpose.

    ``results/figures/`` is versioned and this repository is public, but these
    are stills of the recordings, which are not redistributable. The
    subdirectory is in ``.gitignore``; it is reproducible from the run
    directories plus the recordings.
    """
    return figures.FIGURES_DIR / "examples"


def grid_height(rows: int) -> float:
    """Height in millimetres of a grid of ``rows`` rows."""
    return rows * (FRAME_H + ROW_GAP) - ROW_GAP


def geometry(rows: int = 4) -> dict:
    """The measured layout, for a notebook that wants to print it."""
    height = grid_height(rows)
    return {
        "text_column_mm": TEXT_W,
        "frame_mm": (round(FRAME_W, 2), round(FRAME_H, 2)),
        "frame_area_mm2": round(FRAME_W * FRAME_H),
        "grid_mm": (TEXT_W, round(height, 1)),
        "frames_per_grid": 3 * rows,
    }


def _style() -> None:
    """The rcParams hygiene of :mod:`src.evaluation.figures`, text renderer off.

    ``figures.style`` is called for one reason: it starts from matplotlib's own
    defaults instead of whatever the session holds, which is what stops an
    editor's dark theme from putting white text on this white figure. Its
    external text renderer stays off -- the only text here is monospaced
    identifiers, and the module must draw on a machine without that toolchain.
    """
    figures.style(usetex=False, log=lambda message: None)


def gutter_lines(hit, dataset: str) -> list[str]:
    """What stands beside one row: position, recording, interval, and the verdict.

    Exactly what the supervisor asked a row to carry, and nothing more. The
    verdict line appears only on the correct row -- an incorrect one gets no
    word at all, so a figure is not four repetitions of "wrong".
    """
    lines = [f"{hit.rank}.", f"{dataset}_{hit.episode}", hit.span]
    if hit.relevant:
        lines.append(VERDICT)
    return lines


def _draw_row(ax, hit, dataset: str, top: float, thumbnails) -> None:
    """One row: three frames and the gutter beside them."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lines = gutter_lines(hit, dataset)
    block = len(lines) * LINE_H
    first = top + (FRAME_H - block) / 2
    for index, line in enumerate(lines):
        # only the verdict is set apart: the position, the recording and the
        # interval are the same plain text on every row, correct or not
        verdict = line == VERDICT
        # Right-aligned, against the frames: every line then sits exactly one
        # COL_GAP from the picture it labels, where a left-aligned block gave a
        # gap that varied with the length of each line -- 0.2 mm on an interval
        # of four-digit seconds, 2.9 mm on a short one.
        #
        # The cost is a ragged left edge: an example whose longest interval is
        # short leaves white to the left of it. That is why GUTTER is kept tight
        # against the measured ink -- it bounds that strip instead of adding to
        # it, as a hand-guessed width did.
        ax.text(GUTTER - COL_GAP, first + LINE_H * index + 2.3, line,
                fontsize=GUTTER_PT, family=MONO,
                color=VERDICT_COLOUR if verdict else INK,
                fontweight="bold" if verdict else "normal",
                ha="right", va="baseline")

    group = (thumbnails or {}).get(hit.fragment) or []
    for column in range(3):
        x = GUTTER + column * (FRAME_W + COL_GAP)
        image = group[column] if column < len(group) else None
        if image is not None and Path(image).exists():
            ax.imshow(plt.imread(str(image)),
                      extent=(x, x + FRAME_W, top + FRAME_H, top),
                      aspect="auto", zorder=1)
        else:
            ax.add_patch(Rectangle((x, top), FRAME_W, FRAME_H, fc=PLACEHOLDER,
                                   ec="none", zorder=1))
        # the same hairline on every frame: it separates the picture from the
        # paper and says nothing about correctness
        ax.add_patch(Rectangle((x, top), FRAME_W, FRAME_H, fill=False,
                               ec=FRAME_EDGE, lw=0.4, zorder=2))


def _canvas(width: float, height: float):
    """A figure of exactly ``width`` x ``height`` millimetres, in mm coordinates."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(width / 25.4, height / 25.4))
    figure.patch.set_facecolor("white")
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)              # y grows downwards, as on paper
    ax.axis("off")
    return figure, ax


def grid(case, panel, thumbnails=None, rows: int | None = None):
    """One variant's grid as its own figure, at the designed width."""
    _style()
    hits = panel.hits[:rows] if rows else panel.hits
    figure, ax = _canvas(TEXT_W, grid_height(len(hits)))
    for index, hit in enumerate(hits):
        _draw_row(ax, hit, case.dataset, index * (FRAME_H + ROW_GAP),
                  thumbnails)
    return figure


def strip(case, hit, thumbnails=None):
    """A single row as its own figure, when one row has to stand alone."""
    _style()
    figure, ax = _canvas(TEXT_W, FRAME_H)
    _draw_row(ax, hit, case.dataset, 0.0, thumbnails)
    return figure


def save(case, thumbnails=None, rows: int | None = None,
         strips: bool = False, directory: Path | None = None,
         dpi: int | None = None,
         log: Callable[[str], None] = print) -> list[Path]:
    """Writes the two grids of one example; ``strips`` also writes every row alone.

    Not :func:`src.evaluation.figures.save` itself: that one trims the canvas to
    its content (``bbox_inches="tight"``), which would undo the millimetre
    geometry this module exists to hold. The resolution and the parent directory
    still come from there, so these figures and the charts of that module
    cannot drift apart on either.
    """
    import matplotlib.pyplot as plt

    directory = Path(directory) if directory is not None else output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    def write(figure, name):
        target = directory / f"{name}.png"
        figure.savefig(target, dpi=dpi or figures.DPI, facecolor="white")
        plt.close(figure)
        log(f"zapisano: {target}")
        written.append(target)

    for suffix, panel in (("baza", case.reference),
                          ("wariant", case.candidate)):
        write(grid(case, panel, thumbnails, rows), f"{case.spec.stem}_{suffix}")
        if strips:
            for hit in (panel.hits[:rows] if rows else panel.hits):
                write(strip(case, hit, thumbnails),
                      f"{case.spec.stem}_{suffix}_{hit.rank}")
    return written
