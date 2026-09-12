"""Writes the configuration files that follow from the frozen decisions (§09).

    python scripts/make_configs.py                 # says what it would do
    python scripts/make_configs.py --execute       # writes it

Sixty-one files repeat the same handful of facts: the frame grid, the fragment
lengths, the seed, the whole scoring block, and whichever decisions of the
development phase they do not themselves probe. Writing that by hand was sixty-one
chances to disagree with ``configs/frozen.yaml`` and with ``src/utils/settings.py``,
and it disagreed twice in one day.

Three sources, no overlap. ``src/utils/experiments.py`` says what a variant IS --
which dimension it probes and what it puts up in that dimension.
``configs/frozen.yaml`` says what has been SETTLED -- the dimensions a variant
does not probe. ``src/utils/settings.py`` says what belongs to the project rather
than to any experiment. A variant never reads the frozen decision in the
dimension it probes: E2 puts up its own representation, E3 to E5 their own
mechanism, and only that keeps the two halves of a pair different files.

E1 is the exception and stays hand-written: it needs a base representation, and
E2 -- which runs after it -- is what decides one.

RELEASES PER FAMILY, not all or nothing. Every file declares which decisions it
cannot be written without, and the ones whose decisions are made get written now.
Three of the six stage-two keys wait for nothing -- segmentation and the base
representation are the verdicts of E1 and E2, already in the thesis, and motion
has no alternative to choose between -- so the thirteen individual contributions
come out long before the verdicts of E4 and E5, which only the full-pipeline
families read. A gate that held them all until every decision was made was
thicker than the actual dependency, and the auxiliary extraction file was a way
round that thickness.

Shows before it writes. The default run prints, family by family, what is new,
what would change, what is already right and what is waiting on which decision;
``--execute`` writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.utils import experiments as exp
from src.utils import frozen as frozen_module
from src.utils.config import ExperimentConfig, load_experiment

CONFIGS = ROOT / "configs"

#: The configurations still written by hand, by explicit list and never by a
#: glob over the directory: `vatex_base.yaml` is outside the ExperimentConfig
#: schema (§03) and a glob would hand it to the validator on every test run.
#:
#: Only E1 is left. It needs a base representation and E2 is what decides one, so
#: a generated E1 file would have to carry a provisional value that no verdict
#: contains -- see DEVELOPMENT below.
MANUAL_DEV: tuple[str, ...] = tuple(
    f"{label.replace('-', '').lower()}_{dataset}.yaml"
    for label in ("E1-A", "E1-B", "E1-C")
    for dataset in exp.SERIES)

#: E5-Bp asks the face regions with the whole query instead of its phrases. That
#: is a mode of one component rather than a variant of an experiment, so it has
#: no entry in the facts table and stays hand-written.
#:
#: `recommended_*` is the composition the test results point at -- the base, the
#: captions and identity, nothing else. It follows from numbers measured after
#: the freeze, not from any verdict in `frozen.yaml`, so the generator has
#: nothing to write it from and it stays hand-written as well.
MANUAL_EXTRA: tuple[str, ...] = ("e5bp_tbbt.yaml", "e5bp_office.yaml",
                                 "recommended_tbbt.yaml", "recommended_office.yaml")

#: outside the schema entirely
OUTSIDE_SCHEMA: tuple[str, ...] = ("vatex_base.yaml",)

#: The development variants E2-E5, one file per label and series. They were
#: hand-written until the generator could carry what defines them: which
#: dimension each one probes and what it puts up in that dimension. That is
#: `LABEL_REPRESENTATION` and `LABEL_MECHANISM` in the facts table -- everything
#: else in these files is the same four constants copied sixty times.
#:
#: E1 is NOT here, and that is a decision, not an omission: E1 needs a base
#: representation and E2 is what decides it. A generated E1 file would have to
#: carry a provisional value that no verdict contains, and `frozen.scene_embedding`
#: is by then the answer to the question E1 is still asking.
DEVELOPMENT: dict[str, tuple[str, ...]] = {
    "E2-A": exp.DATASETS,          # the base of all three collections
    "E2-Ap": exp.SERIES, "E2-B": exp.SERIES, "E2-Bp": exp.SERIES,
    "E3-B": exp.SERIES, "E3-C": exp.SERIES,
    "E4-B": exp.SERIES, "E4-C": exp.SERIES,
    "E5-B": exp.SERIES, "E5-C": exp.SERIES,
}

INDIVIDUAL: dict[str, tuple[str, ...]] = {
    "E2-C": exp.DATASETS,
    "E6-B": exp.SERIES,
    "E4-D": exp.DATASETS,
    "E3-B": ("vatex",), "E3-C": ("vatex",),
    "E4-B": ("vatex",), "E4-C": ("vatex",),
    "E5-B": ("vatex",), "E5-C": ("vatex",),
}

HEADER = """# GENERATED by scripts/make_configs.py from configs/frozen.yaml.
# Edit the decision there, not this file: the next run of the generator
# overwrites whatever is written here.
#
# {description}
"""


#: what each family of configurations is called in the report
FAMILIES: tuple[str, ...] = ("development", "individual", "full", "marginal", "swap")


def base_components(frozen) -> dict:
    """The components of the full pipeline, as the frozen decisions leave them."""
    return {
        "scene_embedding": {"enabled": True, "model": frozen.scene_embedding},
        "caption": {"enabled": True, "model": frozen.caption.chosen},
        "objects": {"enabled": True, "detector": frozen.objects.chosen},
        "face_regions": {"enabled": True, "mode": frozen.face_regions.chosen},
        "identity": {"enabled": True},
        "motion": {"enabled": True, "model": frozen.motion},
    }


def representation_of(label: str, frozen) -> dict:
    """The base component of one variant.

    A variant does not read the frozen decision in the dimension it PROBES. The
    four E2 labels probe the representation, so each puts up its own; everything
    else takes the winner. Reading E2 off `frozen.scene_embedding` would make all
    four of its variants the same file.
    """
    if label in exp.LABEL_REPRESENTATION:
        model, prompting = exp.LABEL_REPRESENTATION[label]
        entry = {"enabled": True, "model": model}
        if prompting:
            entry["prompting"] = True
        return entry
    return {"enabled": True, "model": frozen.scene_embedding}


def one_signal(signal: str | None, label: str, frozen) -> dict:
    """The base representation plus one signal, everything else off.

    Where the label names its own mechanism -- E3-B is BLIP, E3-C is LLaVA --
    that name is used and the frozen decision is NOT read. The pair IS the
    comparison of the two mechanisms, so reading either side off the winner
    would give both halves the same one and E3 would compare a thing with
    itself. (It did: until this was fixed, e3b_vatex and e3c_vatex both came out
    with the chosen captioner.)
    """
    components = {name: {"enabled": False} for name in
                  ("scene_embedding", "caption", "objects", "face_regions",
                   "identity", "motion")}
    components["scene_embedding"] = representation_of(label, frozen)
    if signal is None or signal == "scene_embedding":
        return components
    components[signal] = component_of(signal, label, frozen)
    return components


def component_of(signal: str, label: str, frozen) -> dict:
    """One component entry, reading ONLY the decision that component needs.

    Not a slice of `base_components`: that function dereferences every mechanism,
    so taking one entry out of it would make `e2c_tbbt.yaml` -- which needs the
    motion decision and nothing else -- wait for the verdicts of E3, E4 and E5.
    The requirement stated by `needs_of` has to be the requirement the builder
    actually has.
    """
    if label in exp.LABEL_MECHANISM:
        return {"enabled": True,
                exp.MECHANISM_FIELD[signal]: exp.LABEL_MECHANISM[label]}
    if signal == "motion":
        return {"enabled": True, "model": frozen.motion}
    if signal == "identity":
        return {"enabled": True}
    return {"enabled": True,
            exp.MECHANISM_FIELD[signal]: getattr(frozen, signal).chosen}


def compose(dataset: str, components: dict, frozen, *, label: str,
            description: str, detection: bool = False) -> dict:
    """One configuration, with the three composition rules of chapter 6.3.1.

    All three are coded here rather than left to the caller, because each one is
    a fact about the material and not about the variant: a signal absent from a
    dataset stays off there whatever the variant asks for (see
    ``SIGNAL_DATASETS`` for which and why); motion is a signal the base cannot
    see only while the base is an image representation, so it comes in with
    ``openclip`` and would be redundant with a video one; and a VATEX clip IS
    its fragment, so it has no segmentation to configure.
    """
    components = {name: dict(entry) for name, entry in components.items()}
    for signal in exp.ADDED_SIGNALS:
        if dataset not in exp.datasets_of(signal):
            components[signal] = {"enabled": False}
    # THIS file's representation, not the frozen one: an E2 variant puts up its
    # own, and the rule is about what the base can see, not about what won.
    if not str(components["scene_embedding"].get("model")).startswith("openclip"):
        components["motion"] = {"enabled": False}

    config: dict = {"experiment": label, "dataset": dataset, "seed": 1234}
    if dataset != "vatex":
        config["segmentation"] = frozen.segmentation.model_dump(exclude_none=True)
    config["frames"] = {"step_s": 1.25, "dedup_cosine": 0.9}
    config["components"] = components
    scoring: dict = {"scope": "full_collection", "normalization": "zscore",
                     "weights": {"mode": "uniform"}}
    if detection:
        scoring["query_time_detection"] = {"enabled": True,
                                           "detector": "yoloe_prompted",
                                           "candidates_top_n": 50}
    config["scoring"] = scoring
    config["evaluation"] = {
        "report_by": ["requirements", "complexity", "kinetics_vocab"]
        if dataset == "vatex" else ["requirements", "complexity"]}

    # the block goes to every generated file that scores against a closed
    # vocabulary, and never to E4-D: that variant asks the detector the phrase
    # itself and applies no threshold at all (section 08)
    temporary = ExperimentConfig.model_validate({**config, "matching": None})
    if temporary.needs_matching and not detection:
        config["matching"] = frozen.matching.model_dump(exclude_none=True)
    return {"_description": description, **config}


def needs_of(dataset: str, label: str, signal: str | None, *,
             whole_pipeline: bool, detection: bool = False) -> set[str]:
    """Which decisions of ``frozen.yaml`` one file cannot be written without.

    Derived from what the builders above actually read, not from a list kept
    beside them:

    * ``scene_embedding`` always -- every file names the base, and the rule that
      switches motion off under a video representation reads it too;
    * ``segmentation`` for a series file, because a VATEX clip has no section;
    * for the whole pipeline, every mechanism, including the loser of each pair
      (the swap family needs it by name);
    * for one signal alone, that signal's decision ONLY where the label does not
      name its own mechanism -- E2-C reads ``motion``, E3-B does not read
      ``caption``;
    * ``matching`` wherever the file scores against a closed vocabulary, which
      is decided by the components and not by the family.
    """
    # a variant that puts up its own representation does not wait for the verdict
    # that would have told it which one to use
    need = set() if label in exp.LABEL_REPRESENTATION else {"scene_embedding"}
    if dataset != "vatex":
        need.add("segmentation")

    if whole_pipeline:
        need |= {"caption", "objects", "face_regions", "motion"}
    elif signal and signal != "scene_embedding" and label not in exp.LABEL_MECHANISM:
        if signal in ("caption", "objects", "face_regions", "motion"):
            need.add(signal)

    # `needs_matching` is a property of the enabled components: objects,
    # hsemotion expressions or motion. E4-D never takes the block.
    if not detection:
        scores_vocabulary = whole_pipeline or signal in ("objects", "motion") or (
            signal == "face_regions"
            and exp.LABEL_MECHANISM.get(label) == "hsemotion")
        if scores_vocabulary:
            need.add("matching")
    return need


def specifications() -> list[dict]:
    """Every file the generator owns, with its family and its requirements.

    Built WITHOUT a frozen.yaml: the requirements are a property of the variant,
    so the generator can say what it could write before it knows any decision.
    """
    out: list[dict] = []

    for label, datasets in DEVELOPMENT.items():
        signal = exp.ADDS_SIGNAL.get(label)
        for dataset in datasets:
            out.append({
                "name": f"{exp.config_stem(label, dataset)}.yaml",
                "family": "development", "label": label, "dataset": dataset,
                "signal": signal, "detection": False, "whole_pipeline": False,
                "description": (f"{label}: the base representation alone." if signal is None
                                else f"{label}: the base plus the {signal} signal alone."),
            })

    for label, datasets in INDIVIDUAL.items():
        signal = exp.ADDS_SIGNAL.get(label)
        for dataset in datasets:
            if signal and dataset not in exp.datasets_of(signal):
                continue                    # identity has no row on VATEX
            detection = label == "E4-D"
            out.append({
                "name": f"{exp.config_stem(label, dataset)}.yaml",
                "family": "individual", "label": label, "dataset": dataset,
                "signal": None if detection else signal, "detection": detection,
                "whole_pipeline": False,
                "description": (f"{label}: the base re-ranked by a query-time detector."
                                if detection else
                                f"{label}: the base plus the {signal} signal alone."),
            })

    for dataset in exp.DATASETS:
        out.append({
            "name": f"{exp.config_stem(exp.FULL, dataset)}.yaml", "family": "full",
            "label": exp.FULL, "dataset": dataset, "signal": None, "detection": False,
            "whole_pipeline": True,
            "description": "The full pipeline: every decision of the development "
                           "phase applied at once.",
        })
        for signal in exp.ADDED_SIGNALS:
            if dataset not in exp.datasets_of(signal):
                continue
            label = exp.no_label(signal)
            out.append({
                "name": f"{exp.config_stem(label, dataset)}.yaml", "family": "marginal",
                "label": label, "dataset": dataset, "signal": signal,
                "detection": False, "whole_pipeline": True, "removed": signal,
                "description": f"The full pipeline without {signal}: the other half "
                               "of that signal's marginal contribution.",
            })
        for signal in exp.SWAPPED_SIGNALS:
            if dataset not in exp.datasets_of(signal):
                continue                    # nothing to swap where the signal is off
            label = exp.swap_label(signal)
            out.append({
                "name": f"{exp.config_stem(label, dataset)}.yaml", "family": "swap",
                "label": label, "dataset": dataset, "signal": signal,
                "detection": False, "whole_pipeline": True, "swapped": signal,
                "description": f"The full pipeline with the {signal} mechanism swapped "
                               "for the one that lost: the swap effect.",
            })

    for spec in out:
        spec["needs"] = needs_of(spec["dataset"], spec["label"], spec["signal"],
                                 whole_pipeline=spec["whole_pipeline"],
                                 detection=spec["detection"])
    return out


def build(spec: dict, frozen) -> dict:
    """One configuration from its specification and the decisions it needs."""
    if spec["whole_pipeline"]:
        components = base_components(frozen)
        if "removed" in spec:
            components[spec["removed"]] = {"enabled": False}
        if "swapped" in spec:
            signal = spec["swapped"]
            components[signal] = {
                "enabled": True,
                exp.MECHANISM_FIELD[signal]: getattr(frozen, signal).rejected}
    else:
        components = one_signal(spec["signal"], spec["label"], frozen)
    return compose(spec["dataset"], components, frozen, label=spec["label"],
                   description=spec["description"], detection=spec["detection"])


def releasable(frozen) -> tuple[dict[str, dict], list[dict]]:
    """``(what can be written now, what cannot and why)``.

    Per FILE, reported per family. The all-or-nothing gate this replaces was
    thicker than the actual dependency: three of the six stage-two decisions
    wait for nothing (segmentation and the base representation are the verdicts
    of E1 and E2, already in the thesis; motion has no alternative to choose
    between), and the individual contributions read no more than those.
    """
    ready: dict[str, dict] = {}
    blocked: list[dict] = []
    for spec in specifications():
        absent = sorted(key for key in spec["needs"] if getattr(frozen, key) is None)
        if absent:
            blocked.append({**spec, "missing": absent})
        else:
            ready[spec["name"]] = build(spec, frozen)
    return ready, blocked


def planned(frozen) -> dict[str, dict]:
    """Every file the generator owns. Requires every decision to be made."""
    return {spec["name"]: build(spec, frozen) for spec in specifications()}


def render(config: dict) -> str:
    """The YAML text of one generated file, header and all."""
    body = {key: value for key, value in config.items() if key != "_description"}
    return (HEADER.format(description=config["_description"])
            + "\n" + yaml.safe_dump(body, sort_keys=False, allow_unicode=True))


def verdict(path: Path, text: str) -> str:
    """``new``, ``changed`` or ``unchanged`` -- compared as CONFIGURATIONS.

    Not as text: a file differing only in the order of its keys or in a comment
    is the same configuration, and reporting it as changed would make the
    generator's own output impossible to read.
    """
    if not path.exists():
        return "new"
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(text)
        candidate = Path(handle.name)
    try:
        same = (load_experiment(path).model_dump()
                == load_experiment(candidate).model_dump())
    except Exception:
        same = False
    finally:
        candidate.unlink(missing_ok=True)
    return "unchanged" if same else "changed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="write the files; without it nothing is touched")
    parser.add_argument("--out", default=str(CONFIGS),
                        help="where to write (a temporary directory, for the tests)")
    args = parser.parse_args()

    # No all-or-nothing gate: every file says which decisions it needs, and the
    # ones whose decisions are made get written now. A family waiting on the
    # verdict of E4 must not hold up one that reads only the verdict of E2.
    frozen = frozen_module.load_frozen()
    out_dir = Path(args.out)
    ready, blocked = releasable(frozen)

    counts = {"new": 0, "changed": 0, "unchanged": 0}
    for family in FAMILIES:
        names = sorted(name for name in ready
                       if _family_of(name) == family)
        held = [spec for spec in blocked if spec["family"] == family]
        if not names and not held:
            continue
        print(f"\n{family}: {len(names)} ready, {len(held)} waiting")
        for name in names:
            text = render(ready[name])
            state = verdict(out_dir / name, text)
            counts[state] += 1
            print(f"  {state:10} {name}")
            if args.execute:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / name).write_text(text, encoding="utf-8")
        for missing in sorted({tuple(spec["missing"]) for spec in held}):
            waiting = sorted(spec["name"] for spec in held
                             if tuple(spec["missing"]) == missing)
            print(f"  waiting on {', '.join(missing)}: {len(waiting)} file(s)")
            for name in waiting:
                print(f"             {name}")

    total = len(ready) + len(blocked)
    print(f"\n{total} generated configurations: {len(ready)} writable "
          f"({', '.join(f'{n} {state}' for state, n in counts.items())}), "
          f"{len(blocked)} waiting on a decision")
    if blocked:
        absent = sorted({key for spec in blocked for key in spec["missing"]})
        print(f"decisions still missing from {frozen_module.FROZEN_FILE.name}: "
              f"{', '.join(absent)}")
    print(f"manual, untouched: {len(MANUAL_DEV)} development + {len(MANUAL_EXTRA)} "
          f"written by hand + {len(OUTSIDE_SCHEMA)} outside the schema")
    if not args.execute:
        print("nothing written - add --execute")


def _family_of(name: str) -> str:
    """Which family a file belongs to, by its name."""
    for spec in specifications():
        if spec["name"] == name:
            return spec["family"]
    return "individual"


if __name__ == "__main__":
    main()
