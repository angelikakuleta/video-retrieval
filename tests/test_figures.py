"""The figure style two notebooks share.

Nothing here draws a chart -- the panels live in the notebooks. What a test can
pin is what would otherwise drift between them: that every signal a run can
write into ``signals.npz`` has a colour and a marker, that the marker really is
a SECOND encoding (a grey print has to stay readable), and that the LaTeX escape
of a percent sign follows whether LaTeX is actually being used.
"""

import matplotlib

matplotlib.use("Agg")

from src.evaluation import figures
from src.utils.config import Components
from src.utils.experiments import ADDED_SIGNALS, BASE_SIGNAL, SIGNAL_NAMES


def test_every_signal_a_run_can_write_has_a_colour_and_a_marker():
    """The names are the ones of ``signals.npz``, which are the component names.

    A curve keyed by a display name would be attached to nothing: the payload of
    the measurement carries `face_regions`, not "twarze".
    """
    wanted = set(Components.model_fields)
    assert wanted <= set(figures.SIGNAL_STYLE), wanted - set(figures.SIGNAL_STYLE)
    assert {BASE_SIGNAL, *ADDED_SIGNALS} <= set(figures.SIGNAL_STYLE)


def test_the_marker_is_a_second_encoding_and_not_a_repetition_of_the_colour():
    markers = [spec["marker"] for spec in figures.SIGNAL_STYLE.values()]
    colours = [spec["color"] for spec in figures.SIGNAL_STYLE.values()]
    assert len(set(markers)) == len(markers)      # printed in grey they still differ
    assert len(set(colours)) == len(colours)


def test_the_palette_is_the_colour_blind_safe_one():
    """Okabe-Ito, and every signal colour drawn from it."""
    assert figures.OKABE_ITO[0] == "#0072B2"
    assert len(set(figures.OKABE_ITO)) == len(figures.OKABE_ITO)
    for spec in figures.SIGNAL_STYLE.values():
        assert spec["color"] in figures.OKABE_ITO


def test_the_legend_text_comes_from_the_one_place_that_names_the_signals():
    for signal in figures.SIGNAL_STYLE:
        assert figures.signal_label(signal) == SIGNAL_NAMES[signal]


def test_the_style_says_whether_it_got_the_typesetting_it_asked_for():
    """A machine without LaTeX gets a readable figure and is told it is not the one.

    ``usetex`` shells out at DRAW time, so an unavailable toolchain would
    otherwise fail in the middle of a notebook run rather than at its top.
    """
    said = []
    got = figures.style(usetex=True, log=said.append)
    assert got is figures.latex_available()
    assert matplotlib.rcParams["text.usetex"] is got
    assert bool(said) is (not got)

    assert figures.style(usetex=False, log=said.append) is False
    assert matplotlib.rcParams["text.usetex"] is False


def test_the_percent_sign_is_escaped_only_while_latex_is_typesetting():
    figures.style(usetex=False)
    assert figures.percent_sign() == "%"
    if figures.latex_available():
        figures.style(usetex=True)
        assert figures.percent_sign() == r"\%"
        figures.style(usetex=False)


def test_a_figure_lands_in_the_results_directory_of_the_repository(tmp_path,
                                                                   monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path / "figures")
    figure = plt.figure()
    written = figures.save(figure, "probe", extra_dir=tmp_path / "thesis", log=lambda _: None)
    plt.close(figure)
    assert [path.name for path in written] == ["probe.png", "probe.png"]
    assert all(path.exists() for path in written)
