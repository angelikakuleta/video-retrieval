"""One table of facts about the experiments: labels, pairs, coverage, names.

No logic lives here, and that is the point. Which label is the baseline, which
signal a variant adds, which pair of labels a contribution is a difference of,
which datasets a signal exists on, how a configuration file is named -- these are
DECISIONS of chapter 6, and until now they were retyped in every notebook that
needed them. A number that disagrees with the thesis has to be wrong in exactly
one place.

Read by ``make_configs.py``, ``evaluation/runs.py``, ``scripts/compare_runs.py``
and the notebooks. Formatting stays in :mod:`src.evaluation.tables`, computation
in :mod:`src.evaluation.compare`.
"""

from __future__ import annotations

# ------------------------------------------------------------------ datasets
#: every dataset, in the order the tables of chapter 6 list them
DATASETS: tuple[str, ...] = ("tbbt", "office", "vatex")

#: the series; VATEX is a collection of independent clips and is treated apart
#: wherever the difference matters (pooling, identity, the judgment pool)
SERIES: tuple[str, ...] = ("tbbt", "office")

#: display names, ASCII only -- these reach the console (CLAUDE.md)
DATASET_NAMES: dict[str, str] = {
    "tbbt": "The Big Bang Theory",
    "office": "The Office",
    "vatex": "VATEX",
}

#: segmentation strategies of E1, as the thesis names them
STRATEGY_NAMES: dict[str, str] = {
    "fixed_window": "stale okna",
    "shots_histogram": "histogram",
    "shots_transnetv2": "TransNetV2",
}


# ------------------------------------------------------------------- signals
#: the base representation; present in every configuration, never "added"
BASE_SIGNAL = "scene_embedding"

#: the signals a variant ADDS to the base. The order is the one the thesis
#: reports them in, which is also the order of the contribution tables.
ADDED_SIGNALS: tuple[str, ...] = ("caption", "objects", "motion",
                                  "face_regions", "identity")

#: where a signal exists at all, and why it is absent where it is.
#:
#: Identity needs recurring characters with profiles, and VATEX clips have
#: neither, so its row is empty there -- not zero.
#:
#: Face regions are left out of VATEX for a different reason: the signal would
#: run, but there is nothing to read the result against. `vatex_query_tags.csv`
#: carries no `wymaga_mimiki` column at all -- a one-sentence clip description
#: does not settle what a face is doing (section 03) -- so the contrast column
#: of that row would be empty by construction, and about 3% of the VATEX
#: queries mention a facial expression in the first place. Decided before the
#: test runs, not after seeing them.
SIGNAL_DATASETS: dict[str, tuple[str, ...]] = {"identity": SERIES,
                                               "face_regions": SERIES}

#: the three signals whose variant is a CHOICE between two mechanisms, so that
#: swapping one for the other is a measurable effect (chapter 6, "efekt podmiany")
SWAPPED_SIGNALS: tuple[str, ...] = ("caption", "objects", "face_regions")

#: which field of a component names its mechanism
MECHANISM_FIELD: dict[str, str] = {"caption": "model", "objects": "detector",
                                   "face_regions": "mode"}

#: The mechanism each B/C variant carries. E3-B is BLIP and E3-C is LLaVA
#: whatever E3 decides: the PAIR is the comparison, so neither side may be read
#: off the winner -- doing that would give both halves the same mechanism and
#: the experiment would compare a thing with itself. Same for E4 and E5.
#: These values are the ones in the twenty-six hand-written development files
#: and a test holds the two in agreement.
LABEL_MECHANISM: dict[str, str] = {
    "E3-B": "blip", "E3-C": "llava_1_5_7b",
    "E4-B": "yolo11", "E4-C": "yoloe_promptfree",
    "E5-B": "clip_regions", "E5-C": "hsemotion",
}

#: The representation each E2 variant puts up, and whether it prompts. E2 is the
#: experiment that DECIDES the base, so none of its four variants may be read off
#: `frozen.scene_embedding` -- that key is the answer to the question these files
#: ask, and taking it would make every one of them the winner.
#:
#: The two pairs are comparisons in the same sense as E3, E4 and E5: E2-A against
#: E2-Ap is one representation at two sizes, E2-B against E2-Bp is X-CLIP with
#: and without video-specific prompting. Neither half may come from the other.
#:
#: Read off the existing e2*.yaml files, which are what these labels mean, and
#: held in agreement with them by a test.
LABEL_REPRESENTATION: dict[str, tuple[str, bool]] = {
    "E2-A": ("openclip_vit_h14", False),
    "E2-Ap": ("openclip_vit_b32", False),
    "E2-B": ("xclip_b32", True),
    "E2-Bp": ("xclip_b32", False),
}

SIGNAL_NAMES: dict[str, str] = {
    "scene_embedding": "reprezentacja bazowa",
    "caption": "opisy scen",
    "objects": "obiekty",
    "motion": "ruch",
    "face_regions": "twarze",
    "identity": "tozsamosc",
}


def datasets_of(signal: str) -> tuple[str, ...]:
    """The datasets a signal has a row on."""
    return SIGNAL_DATASETS.get(signal, DATASETS)


# -------------------------------------------------------------------- labels
#: the baseline of all three datasets. One label, so a table row means the same
#: thing everywhere; the display name is a separate matter.
BASE = "E2-A"

#: the full pipeline, and the two families derived from it
FULL = "full"
NO_PREFIX = "full_no_"
SWAP_PREFIX = "full_swap_"

#: which signal each variant adds to the base. This is what makes a label a
#: candidate for an individual contribution.
ADDS_SIGNAL: dict[str, str] = {
    "E2-C": "motion",
    "E3-B": "caption",
    "E3-C": "caption",
    "E4-B": "objects",
    "E4-C": "objects",
    "E5-B": "face_regions",
    "E5-C": "face_regions",
    "E6-B": "identity",
}

#: variants of each development experiment, simplest first, with the reference
#: each is compared against. A block of a notebook reads one entry and runs.
E1: tuple[str, ...] = ("E1-A", "E1-B", "E1-C")
E1_REFERENCE = "E1-A"

E2: tuple[str, ...] = ("E2-A", "E2-Ap", "E2-B", "E2-Bp")
#: the control variants only split the measured effect and do not decide
E2_DECISION: tuple[str, ...] = ("E2-A", "E2-B")
E2_REFERENCE = "E2-A"

#: the E3-E5 decisions, each a pair. The SIMPLER variant comes first and is the
#: reference, so the difference reads "complex minus simpler" and the fallback
#: of the decision rule lands on the simpler one -- the rule of chapter 4.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("E3-B", "E3-C", "BLIP wobec LLaVA"),
    ("E4-B", "E4-C", "YOLO11 wobec YOLOE"),
    ("E5-B", "E5-C", "regiony wobec HSEmotion"),
)

LABEL_NAMES: dict[str, str] = {
    "E1-A": "E1-A: stale okna",
    "E1-B": "E1-B: histogram",
    "E1-C": "E1-C: TransNetV2",
    "E2-A": "E2-A: OC (ViT-H/14)",
    "E2-Ap": "E2-Ap: OC (ViT-B/32), kontrola",
    "E2-B": "E2-B: XC z podpowiedziami",
    "E2-Bp": "E2-Bp: XC bez podpowiedzi, kontrola",
    "E2-C": "E2-C: BAZA + SlowFast",
    "E3-B": "E3-B: BAZA + BLIP",
    "E3-C": "E3-C: BAZA + LLaVA-1.5",
    "E4-B": "E4-B: BAZA + YOLO11",
    "E4-C": "E4-C: BAZA + YOLOE-11",
    "E4-D": "E4-D: YOLOE sterowany zapytaniem",
    "E5-B": "E5-B: BAZA + regiony twarzy",
    "E5-Bp": "E5-Bp: regiony na zdaniu, kontrola",
    "E5-C": "E5-C: BAZA + HSEmotion",
    "E6-B": "E6-B: BAZA + tozsamosc",
    FULL: "potok pelny",
}

#: the component each variant adds, named as the thesis names it. LABEL_NAMES
#: answers "which experiment"; this answers "which model", and a table that puts
#: two variants of one signal side by side needs the second -- a row reading
#: "obiektowy" does not say whether it is YOLO11 or YOLOE-11, and once every
#: variant has a test run both stand in the same table. Also the only way to
#: name the mechanism of a ``full_no_*`` or ``full_swap_*`` row, whose signal is
#: known from the label but whose model comes from the frozen verdict.
LABEL_COMPONENT: dict[str, str] = {
    "E2-C": "SlowFast",
    "E3-B": "BLIP",
    "E3-C": "LLaVA-1.5",
    "E4-B": "YOLO11",
    "E4-C": "YOLOE-11",
    "E4-D": "YOLOE sterowany zapytaniem",
    "E5-B": "regiony twarzy",
    "E5-Bp": "regiony twarzy, cale zdanie",
    "E5-C": "HSEmotion",
    "E6-B": "tozsamosc",
}

#: what the baseline is called once the development phase is over and the label
#: names the reference rather than a choice between representations
BASE_DISPLAY = "BAZA"


def display(label: str, base_as_name: bool = True) -> str:
    """Display name of a label; the label itself when none was written down.

    The baseline has two correct names and which one is correct depends on the
    table. In E2 it is one of the competing representations and its row has to
    say WHICH one, so it reads "E2-A: OC (ViT-H/14)". In E3 to E6 it is the
    reference every candidate is measured against and its row reads "BAZA";
    printing the representation there would name the same row two ways across
    one chapter.

    ``base_as_name=False`` asks for the second reading. It applies to the
    baseline alone -- every other label has one name.
    """
    if label == BASE and not base_as_name:
        return BASE_DISPLAY
    if label in LABEL_NAMES:
        return LABEL_NAMES[label]
    for prefix, shape in ((NO_PREFIX, "pelny bez: {}"), (SWAP_PREFIX, "podmiana: {}")):
        if label.startswith(prefix):
            signal = label[len(prefix):]
            return shape.format(SIGNAL_NAMES.get(signal, signal))
    return label


def signal_experiment(signal: str) -> str | None:
    """Which experiment decides a signal, from the labels that add it.

    ``E3-B`` and ``E3-C`` both add the caption signal, so the caption signal
    belongs to E3. Derived from :data:`ADDS_SIGNAL` rather than written out
    again, because a label added there is a label that already says this.
    """
    found = {label[:2] for label, added in ADDS_SIGNAL.items() if added == signal}
    if len(found) != 1:
        return None
    return found.pop()


def target_tags(signal: str) -> list[str]:
    """The requirement tags whose queries a signal is supposed to speak about.

    The "docelowe" column of the activation table is the activation share over
    exactly these queries. A tag points at an experiment
    (``vocabulary.TAG_EXPERIMENT``) and an experiment decides a signal, so the
    mapping is the composition of the two -- not a third list to keep in step.

    ``wymaga_mimiki`` is the hypothesis of two experiments at once (E2 and E5),
    so it targets the motion signal as well as the face one; that is a decision
    of chapter 4, visible here rather than hidden in a notebook.
    """
    from src.utils.vocabulary import TAG_EXPERIMENT

    experiment = signal_experiment(signal)
    if experiment is None:
        return []
    return [str(tag) for tag, wanted in TAG_EXPERIMENT.items() if experiment in wanted]


# ---------------------------- the three contributions, each a pair of labels
def individual_pair(label: str) -> tuple[str, str]:
    """Delta_ind: what one signal adds to the base on its own."""
    if label not in ADDS_SIGNAL:
        raise ValueError(f"{label!r} adds no single signal to the base; "
                         f"expected one of {sorted(ADDS_SIGNAL)}")
    return label, BASE


def marginal_pair(signal: str) -> tuple[str, str]:
    """Delta_kr: what one signal adds to the full pipeline, the rest present."""
    return FULL, no_label(signal)


def swap_pair(signal: str) -> tuple[str, str]:
    """Delta_pod: what changes when one signal's mechanism is swapped."""
    if signal not in SWAPPED_SIGNALS:
        raise ValueError(f"{signal!r} has no second mechanism to swap in; "
                         f"expected one of {list(SWAPPED_SIGNALS)}")
    return FULL, swap_label(signal)


def no_label(signal: str) -> str:
    """Label of the full pipeline with one signal removed."""
    _check_signal(signal)
    return f"{NO_PREFIX}{signal}"


def swap_label(signal: str) -> str:
    """Label of the full pipeline with one signal's mechanism swapped."""
    _check_signal(signal)
    return f"{SWAP_PREFIX}{signal}"


def _check_signal(signal: str) -> None:
    if signal not in ADDED_SIGNALS:
        raise ValueError(f"unknown added signal: {signal!r}; "
                         f"expected one of {list(ADDED_SIGNALS)}")


# ------------------------------------------------------- configuration files
def config_stem(label: str, dataset: str) -> str:
    """File name of a configuration, without the extension.

    ``E5-Bp`` + ``tbbt`` -> ``e5bp_tbbt``; ``full_no_objects`` + ``vatex`` ->
    ``full_no_objects_vatex``. The dashes of a development label disappear and
    the ``full_*`` labels are already file-shaped, which is why they were named
    that way.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r}; expected one of {list(DATASETS)}")
    return f"{label.replace('-', '').lower()}_{dataset}"
