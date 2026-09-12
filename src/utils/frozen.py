"""What the development phase settled, in one file the generator reads (§09).

The thesis requires every variant to be its own configuration file, so the
configurations cannot inherit from one another. What they CAN do is stop
repeating the decisions: which segmentation won, which representation, which
caption generator, which detector, which face mechanism, and the threshold the
closed-vocabulary signals match against. ``configs/frozen.yaml`` is the only
carrier of that, and ``scripts/make_configs.py`` is the only writer of the files
derived from it.

**Written by hand, in two stages, and this module never writes it.** Stage one
carries the matching block and can be filled as soon as the threshold report
exists; stage two carries the winners and waits for the verdicts of E4 and E5.
The person filling it copies the winners out of the final table of
``dev_results.ipynb``. It has to be a person: a cell that wrote this file
automatically would write down "?" and "provisionally" as readily as a verdict,
and those are not values a run may be built on -- somebody has to look at an
interval covering zero and decide.

The file itself is written in the same plain layout as every configuration,
and what it carries is the values. A short marker beside an entry is fine;
the provenance of a verdict is not. The comparison that produced it is in
``results/reports/e<n>_dev.json`` and the tables of chapter 6, which say it
with the interval and the query counts a one-line comment could only
approximate.

The segmentation entry is the WHOLE block, not the name of the strategy. Of the
twenty-six development runs, twenty-four carry ``histogram_threshold: 0.5``; a
generator holding only the name would emit the schema default of ``None`` and
the test configurations would differ from the development ones in the parameter
that decides where fragments begin and end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import model_validator

from src.utils.config import (ROOT, Matching, SegmentationConfig, StrictModel,
                              load as load_yaml)

#: where the decisions live
FROZEN_FILE = ROOT / "configs" / "frozen.yaml"

#: filled after the threshold report (§07), before any run with the new scoring
STAGE_ONE: tuple[str, ...] = ("matching",)

#: filled after the verdicts of E4 and E5, and what the test gate checks for
STAGE_TWO: tuple[str, ...] = ("segmentation", "scene_embedding", "caption",
                              "objects", "face_regions", "motion")


class Choice(StrictModel):
    """A decision of the development phase: what won and what it beat.

    The loser is written down as well, because the "swap effect" of chapter 6 is
    the full pipeline with exactly that substitution -- so the generator needs
    both names, and neither may be guessed from the other.
    """

    @model_validator(mode="after")
    def check(self):
        if self.chosen == self.rejected:
            raise ValueError(f"chosen and rejected are the same: {self.chosen!r}; "
                             "a decision between one option is not a decision")
        return self


class CaptionChoice(Choice):
    chosen: Literal["blip", "llava_1_5_7b"]
    rejected: Literal["blip", "llava_1_5_7b"]


class ObjectsChoice(Choice):
    chosen: Literal["yolo11", "yoloe_promptfree"]
    rejected: Literal["yolo11", "yoloe_promptfree"]


class FaceRegionsChoice(Choice):
    chosen: Literal["clip_regions", "hsemotion"]
    rejected: Literal["clip_regions", "hsemotion"]


class Frozen(StrictModel):
    """The decisions, all optional -- the file is filled in two sittings.

    Optional in the SCHEMA and required by the GATE: the generator asks for what
    it is about to write and says what is missing, and
    ``scripts/run_experiment.py`` refuses the test split until stage two is
    complete. A schema that demanded everything at once would make the file
    unloadable between the two stages, which is most of the time it exists.
    """

    matching: Matching | None = None
    segmentation: SegmentationConfig | None = None
    scene_embedding: Literal["openclip_vit_h14", "openclip_vit_b32",
                             "xclip_b32"] | None = None
    caption: CaptionChoice | None = None
    objects: ObjectsChoice | None = None
    face_regions: FaceRegionsChoice | None = None
    motion: Literal["slowfast_k400"] | None = None

    def missing(self, stage: tuple[str, ...]) -> list[str]:
        """Which keys of a stage are still empty, in the order they are written."""
        return [key for key in stage if getattr(self, key) is None]

    def complete(self, stage: tuple[str, ...]) -> bool:
        return not self.missing(stage)


def load_frozen(file: Path | str = FROZEN_FILE) -> Frozen:
    """The decisions as they stand. A missing file is an empty one, not an error.

    Before the measurements there is nothing to freeze, and the generator has to
    be able to say so rather than fail to import.
    """
    path = Path(file)
    if not path.exists():
        return Frozen()
    data = load_yaml(path)
    if data is None:
        return Frozen()
    if not isinstance(data, dict):
        raise ValueError(f"{path}: frozen.yaml must be a YAML mapping")
    return Frozen.model_validate(data)


def require(stage: tuple[str, ...], what: str,
            file: Path | str = FROZEN_FILE) -> Frozen:
    """The decisions, or a refusal naming exactly what is not decided yet."""
    frozen = load_frozen(file)
    absent = frozen.missing(stage)
    if absent:
        raise SystemExit(
            f"{what} needs decisions that {Path(file)} does not carry yet: "
            f"{', '.join(absent)}. That file is written by hand, in two stages -- "
            "stage one after the threshold report, stage two after the verdicts of "
            "E4 and E5 (see src/utils/frozen.py).")
    return frozen
