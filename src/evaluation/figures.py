"""Style shared by the notebook blocks that draw a figure for the thesis.

Two figures go into chapter 6 -- the distribution of face sizes and the weight
sensitivity -- and they are drawn in two different notebooks. What has to agree
between them is not the drawing but the look: the same colour-blind-safe
palette, the same marker as a SECOND encoding of the same signal (a reader
printing the page in grey has to keep the curves apart), the same serif set as
the thesis and the same place on disk. Kept in one module for the same reason as
:mod:`src.utils.experiments`: a fact that disagrees with the thesis has to be
wrong in exactly one place.

Nothing here computes and nothing here draws a chart. The panels stay in the
notebooks, next to the table they belong to.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.data.datasets import ROOT
from src.utils.experiments import SIGNAL_NAMES

#: where a figure of the thesis is written
FIGURES_DIR = ROOT / "results" / "figures"

#: resolution of the saved file; the thesis prints at 300 dpi
DPI = 300

#: Okabe-Ito, the eight-colour palette that stays distinguishable under the
#: three common forms of colour blindness. Black is left out: it is the colour
#: of the axes and of the reference lines here.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#CC79A7",   # reddish purple
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#F0E442",   # yellow
)

#: colour and marker of every signal, keyed by the name a run writes into
#: ``signals.npz`` -- the component names, not display names, so a curve cannot
#: be attached to the wrong row. The legend text comes from ``SIGNAL_NAMES``,
#: which is where the Polish names of the signals already live.
SIGNAL_STYLE: dict[str, dict[str, str]] = {
    "scene_embedding": {"color": OKABE_ITO[5], "marker": "*"},
    "caption": {"color": OKABE_ITO[0], "marker": "o"},
    "objects": {"color": OKABE_ITO[1], "marker": "s"},
    "motion": {"color": OKABE_ITO[2], "marker": "^"},
    "face_regions": {"color": OKABE_ITO[3], "marker": "D"},
    "identity": {"color": OKABE_ITO[4], "marker": "v"},
}


#: names for text DRAWN into a figure, which may carry diacritics. Console
#: output may not (see CLAUDE.md), and ``SIGNAL_NAMES`` feeds both, so the two
#: are kept apart; only the entries that actually differ are listed here.
FIGURE_SIGNAL_NAMES: dict[str, str] = {
    "identity": "tożsamość",
}


def signal_label(signal: str) -> str:
    """Legend text of one signal, as chapter 6 names it. ASCII, safe on console."""
    return SIGNAL_NAMES.get(signal, signal)


def figure_label(signal: str) -> str:
    """The same name for drawing, with diacritics where Polish has them."""
    return FIGURE_SIGNAL_NAMES.get(signal, signal_label(signal))


def latex_available() -> bool:
    """Whether the LaTeX toolchain matplotlib needs for ``text.usetex`` is here.

    ``usetex`` shells out to ``latex`` and ``dvipng``; without them matplotlib
    raises at DRAW time, not when the parameter is set, so the failure would
    land in the middle of a notebook run rather than at its top.
    """
    return all(shutil.which(program) for program in ("latex", "dvipng"))


def style(usetex: bool = True, log=print) -> bool:
    """Applies the look of the thesis and says whether LaTeX typesetting is on.

    ``usetex=True`` is a request, not a promise: on a machine without the LaTeX
    toolchain the figure is drawn with matplotlib's own fonts and the caller is
    told so. The figure is then not the one to paste into the thesis -- the
    glyphs differ -- and printing that is the point of the return value.
    """
    import matplotlib.pyplot as plt

    wanted = bool(usetex) and latex_available()
    if usetex and not wanted:
        log("no LaTeX toolchain (latex, dvipng) - drawing with matplotlib fonts. "
            "The figure is readable but its glyphs are NOT the ones of the thesis.")

    # Start from matplotlib's own defaults, not from whatever the session already
    # holds. rcParams are process-global and an editor may have put a dark theme
    # there: VS Code does it with `jupyter.themeMatplotlibPlots`, and JupyterLab
    # extensions do the same. Updating on top of that changed only the keys named
    # below and left the rest dark, which produced a figure with black axes, a
    # white canvas and WHITE text on it -- every label invisible, and invisible in
    # the saved PNG too, not only on screen. A figure that goes into the thesis
    # cannot depend on the theme of the editor that drew it.
    plt.style.use("default")
    plt.rcParams.update({
        "text.usetex": wanted,
        # T1 plus lmodern is what puts Polish glyphs in the figures: the diacritics
        # of the labels go through inputenc and come out of a font that has them.
        # lmodern carries a sans face as well, so the family below needs nothing
        # more than the switch.
        "text.latex.preamble": (r"\usepackage[utf8]{inputenc}"
                                r"\usepackage[T1]{fontenc}\usepackage{lmodern}"),
        # Sans-serif on purpose, and different from the serif body text of the
        # thesis: a label read at a glance beside a bar is not running text, and
        # the contrast tells the reader which is which.
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        # stated even where they match the default, so that a future style change
        # upstream cannot quietly darken the figures again
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })
    return wanted


def percent_sign() -> str:
    """``\\%`` while ``text.usetex`` is on, ``%`` while it is not.

    The escape is a LaTeX one: printed through matplotlib's own text engine it
    would appear on the axis label as a backslash.
    """
    import matplotlib as mpl

    return r"\%" if mpl.rcParams["text.usetex"] else "%"


def save(figure, name: str, extra_dir: Path | str | None = None,
         extra_name: str | None = None, log=print) -> list[Path]:
    """Writes one figure to ``results/figures/`` and optionally beside it.

    ``extra_dir`` is the figure directory of the thesis repository, when the
    author has one set. ``extra_name`` is what the copy is called there, because
    the two repositories name things in two languages and the ``.tex`` sources
    already refer to the Polish name; without it the copy keeps the name used
    here.
    """
    targets = [FIGURES_DIR / f"{name}.png"]
    if extra_dir is not None:
        targets.append(Path(extra_dir) / f"{extra_name or name}.png")
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=DPI, bbox_inches="tight", facecolor="white")
        log(f"zapisano: {target}")
    return targets
