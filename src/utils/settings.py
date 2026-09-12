"""Parameters shared by more than one notebook, script or module.

One place for values that would otherwise be typed out again in every notebook
that needs them, and for the few that are tied to each other numerically and
must not drift apart. Everything else stays with the code that uses it: detector
thresholds, model and batch sizes, blackdetect levels, the standardisation
epsilon, the subtitle-anchoring parameters. Variant settings -- seed, the
segmentation strategy, the base model, the weights -- live in ``configs/*.yaml``,
because they describe an experiment and not the project.
"""

from __future__ import annotations

# -------------------------------------------------------------- the material

#: episodes selected per series, of which DEV_COUNT go to the development split
EPISODE_COUNT = 24
DEV_COUNT = 6

#: the two evaluated parts, in the order rows are written. All three datasets
#: have both: the VATEX development part settles the phrase-matching threshold
#: and checks the motion signal, and takes part in no configuration decision.
SPLITS = ("dev", "test")


def split_order(split: str) -> int:
    """Sort key of a split: development first, test after, anything else last.

    Every file that mixes both parts is written in this order, so an episode of
    the development set is always above a test one no matter how the seasons
    happen to be numbered.
    """
    return SPLITS.index(split) if split in SPLITS else len(SPLITS)


# ----------------------------------------------------------------- durations
#
# These four are tied to each other and drifting them apart silently breaks the
# experiments, so they sit together with the reason written down.

#: frame sampling grid [s]. Chosen so that the shortest admissible fragment
#: (MIN_LEN) still holds two frames.
FRAME_STEP = 1.25

#: shortest usable annotation [s]. Equal to FRAME_STEP: anything shorter cannot
#: be hit by a single sampled frame, so no model would ever see it.
MIN_DURATION = FRAME_STEP

#: longest usable annotation [s]. Above this an event spreads over several
#: fragments and the "correct answer" stops being one fragment.
MAX_DURATION = 15.0

#: shortest and longest fragment [s], after the length correction
MIN_LEN = 3.0
MAX_LEN = MAX_DURATION

#: shortest corpus range worth writing [s]. Equal to MIN_LEN, so a range is
#: never shorter than the shortest fragment it would have to hold.
MIN_RANGE = MIN_LEN

#: fixed-window length [s] of segmentation variant E1-A
WINDOW = 10.0

#: which of the ten VATEX descriptions of a clip becomes the query. It lives
#: here rather than beside the VATEX adapter because both parts have to use the
#: same one and the development notebook cannot import that adapter: the
#: notebook is stdlib-only by design and the adapter pulls in pandas.
VATEX_EXPERIMENT_DESC = 1

#: nominal length of a VATEX excerpt [s] and the band a downloaded clip has to
#: fall in. The threshold is one-sided in effect: a clip LONGER than nominal is
#: harmless (ffmpeg added a margin when cutting to a keyframe and the annotated
#: event is still inside), a SHORTER one means the source recording ended before
#: the annotation window closed. Both parts use the same band -- a development
#: part filtered differently from the test part would not be comparable with it.
VATEX_NOMINAL_S = 10.0
VATEX_MIN_CLIP_S = 9.5
VATEX_MAX_CLIP_S = 10.5

# ---------------------------------------------------------------- characters

#: the characters that get an identity profile (experiment E6), five per series,
#: by first name -- that is what a query says and what the identity signal
#: matches on (:func:`src.utils.vocabulary.resolve_characters`).
PROFILED_CHARACTERS = {
    "tbbt": ["Sheldon", "Leonard", "Penny", "Howard", "Raj"],
    "office": ["Michael", "Dwight", "Jim", "Pam", "Kevin"],
}


def profiled(series: str) -> list[str]:
    """Profiled characters of a series, by its directory name (``tbbt``/``office``)."""
    if series not in PROFILED_CHARACTERS:
        raise KeyError(f"no profiled characters for {series!r} - "
                       f"known: {sorted(PROFILED_CHARACTERS)}")
    return list(PROFILED_CHARACTERS[series])
