"""Loading and validating the pipeline configuration.

One pipeline configuration = one YAML file, in the schema documented in the
thesis appendix. The schema below is that table expressed as Pydantic models:
unknown fields, wrong types, values from outside the documented set and
contradictory combinations abort the run before anything is computed.

The evaluated split (``dev``/``test``) is NOT part of the variant's identity:
it is passed at invocation (``--split``) and injected into the configuration,
so the resolved configuration written into the run directory describes the
run completely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]


def load(file: Path | str) -> dict:
    """Loads a YAML file into a dict (no validation -- helper scripts only)."""
    return yaml.safe_load(Path(file).read_text(encoding="utf-8"))


def path(relative: str) -> Path:
    """Expands a path from the configuration relative to the repository root."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else ROOT / candidate


# ----------- Schema (mirrors the configuration table in the thesis appendix)
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentationConfig(StrictModel):
    strategy: Literal["fixed_window", "shots_histogram", "shots_transnetv2"]
    min_len_s: float = Field(3, gt=0)
    max_len_s: float = Field(15, gt=0)
    window_s: float = Field(10, gt=0)
    histogram_threshold: float | None = Field(None, gt=0, lt=1)
    transnet_threshold: float | None = Field(None, gt=0, lt=1)

    @model_validator(mode="after")
    def check(self):
        if self.min_len_s >= self.max_len_s:
            raise ValueError("min_len_s must be smaller than max_len_s")
        if not self.min_len_s <= self.window_s <= self.max_len_s:
            raise ValueError("window_s must lie between min_len_s and max_len_s")
        if self.histogram_threshold is not None and self.strategy != "shots_histogram":
            raise ValueError("histogram_threshold applies only to shots_histogram")
        if self.transnet_threshold is not None and self.strategy != "shots_transnetv2":
            raise ValueError("transnet_threshold applies only to shots_transnetv2")
        return self


class FramesConfig(StrictModel):
    step_s: float = Field(1.25, gt=0)
    dedup_cosine: float = Field(0.9, ge=0, le=1)


class SceneEmbedding(StrictModel):
    enabled: bool
    model: Literal["openclip_vit_h14", "openclip_vit_b32", "xclip_b32"] = "openclip_vit_h14"
    #: X-CLIP only: video-specific prompting. Off, the fragment is matched against
    #: a stored vector like any contrastive representation; on, the query embedding
    #: is shifted by a term derived from the fragment being scored. Both read the
    #: same index and differ only in the query phase.
    prompting: bool = False

    @model_validator(mode="after")
    def check(self):
        if self.prompting and self.model != "xclip_b32":
            raise ValueError("prompting applies only to the xclip_b32 representation")
        return self


class Caption(StrictModel):
    enabled: bool
    model: Literal["blip", "llava_1_5_7b"] = "llava_1_5_7b"
    do_sample: bool = False


class Objects(StrictModel):
    enabled: bool
    detector: Literal["yolo11", "yoloe_promptfree"] = "yoloe_promptfree"


class FaceRegions(StrictModel):
    enabled: bool
    mode: Literal["clip_regions", "hsemotion"] = "clip_regions"
    #: what the face crops are compared against. `phrases` is the signal itself;
    #: `sentence` compares the whole query with the crop and exists only for the
    #: E5-Bp control row -- it shows how much of the open-vocabulary result comes
    #: from restricting the query to phrases at all.
    query: Literal["phrases", "sentence"] = "phrases"

    @model_validator(mode="after")
    def check(self):
        if self.query == "sentence" and self.mode != "clip_regions":
            raise ValueError("face_regions.query: sentence applies only to "
                             "mode: clip_regions")
        return self


class Identity(StrictModel):
    enabled: bool
    # no `profiles` path: the signal loads the dataset's profiles by convention
    # (`identity.load_profiles(dataset)`), so a field here would never be read


class Motion(StrictModel):
    enabled: bool
    model: Literal["slowfast_k400"] = "slowfast_k400"


class Components(StrictModel):
    scene_embedding: SceneEmbedding
    caption: Caption = Caption(enabled=False)
    objects: Objects = Objects(enabled=False)
    face_regions: FaceRegions = FaceRegions(enabled=False)
    identity: Identity = Identity(enabled=False)
    motion: Motion = Motion(enabled=False)

    def enabled_signals(self) -> list[str]:
        return [name for name in ("scene_embedding", "caption", "objects",
                                  "face_regions", "identity", "motion")
                if getattr(self, name).enabled]


class QueryTimeDetection(StrictModel):
    enabled: bool = False
    detector: Literal["yoloe_prompted"] = "yoloe_prompted"
    candidates_top_n: int = Field(50, gt=0)


class Weights(StrictModel):
    mode: Literal["uniform", "manual"] = "uniform"
    manual: dict[str, float] | None = None


class Scoring(StrictModel):
    scope: Literal["full_collection"] = "full_collection"
    normalization: Literal["zscore"] = "zscore"
    weights: Weights = Weights()
    query_time_detection: QueryTimeDetection | None = None


class Evaluation(StrictModel):
    # no `metrics` list: the runner always computes every measure in
    # metrics.PER_QUERY_KEYS, so declaring a subset here would not narrow it
    report_by: list[Literal["requirements", "complexity", "kinetics_vocab"]] = []


#: the vocabulary caches a per-vocabulary threshold may name. They are built in
#: :mod:`src.runners.components` from the templates that own them; repeated here
#: because that module imports this one, and pinned by a test so the two lists
#: cannot drift apart.
VOCABULARY_NAMES = ("objects_yolo11", "objects_yoloe_promptfree",
                    "expressions_hsemotion", "actions_kinetics400")


class Matching(StrictModel):
    """How a query phrase is matched against the names of a closed vocabulary.

    Neither field has a default: the threshold and the measure come out of one
    measurement (chapter 6) and are frozen in ``configs/frozen.yaml``. A default
    here would let a run quietly use a number nobody chose.
    """

    measure: Literal["top1", "z"]
    threshold: float
    #: a threshold of its own for a vocabulary that measured one. Higher or
    #: LOWER than `threshold`, either way: each vocabulary is calibrated on its
    #: own judged sample, and the four came out at 0,805 / 0,733 / 0,805 / 0,802
    #: -- two of them below the first.
    threshold_by_vocab: dict[str, float] = {}

    @model_validator(mode="after")
    def check(self):
        unknown = sorted(set(self.threshold_by_vocab) - set(VOCABULARY_NAMES))
        if unknown:
            raise ValueError(
                f"threshold_by_vocab keys must be vocabulary caches "
                f"{list(VOCABULARY_NAMES)}; got {unknown}")
        return self


class ExperimentConfig(StrictModel):
    experiment: str
    dataset: Literal["tbbt", "office", "vatex"]
    seed: int = 1234
    segmentation: SegmentationConfig | None = None
    frames: FramesConfig = FramesConfig()
    components: Components
    scoring: Scoring = Scoring()
    evaluation: Evaluation
    # optional in the SCHEMA on purpose: run_features.py and measure_cost.py read
    # these files too and neither produces a number that a threshold could bend.
    # Whether a run may go ahead without it is decided by run.execute, which is
    # where the numbers are made.
    matching: Matching | None = None
    split: Literal["dev", "test"] | None = None   # injected by --split at invocation

    @property
    def collection_strategy(self) -> str:
        """Name the collection is cached and indexed under.

        A VATEX configuration has no segmentation section -- there is nothing to
        configure when the recording is already the fragment -- but the caches
        still need a name, and every reader wants one field rather than a branch.
        Read instead of injecting a section: a section written here would be
        dumped into config.resolved.yaml, which the validator then refuses to
        read back.
        """
        from src.segmentation.build import WHOLE_CLIP     # avoids an import cycle

        return self.segmentation.strategy if self.segmentation else WHOLE_CLIP

    @property
    def needs_matching(self) -> bool:
        """Whether this run compares phrases against a closed vocabulary.

        The three signals that do: objects, expressions and motion. Face regions
        in ``clip_regions`` mode have no class names to reach a threshold
        against, and the query-time detector answers the phrase itself, so
        neither needs the block.
        """
        components = self.components
        return bool(components.objects.enabled
                    or (components.face_regions.enabled
                        and components.face_regions.mode == "hsemotion")
                    or components.motion.enabled)

    @model_validator(mode="after")
    def check(self):
        if self.dataset == "vatex":
            if self.segmentation is not None:
                raise ValueError("the segmentation section is omitted for VATEX")
        elif self.segmentation is None:
            raise ValueError(f"the {self.dataset} dataset requires a segmentation section")

        weights = self.scoring.weights
        enabled = set(self.components.enabled_signals())
        if not enabled:
            raise ValueError("at least one component must be enabled")

        # E4-D is a SCORING STAGE, not a signal of the full pipeline: it re-ranks
        # the top candidates of the base with a detector asked the query's own
        # phrases. Nothing else may be switched on beside it, or the re-ranking
        # would be measured against a mixture instead of against the base.
        detection = self.scoring.query_time_detection
        if detection is not None and detection.enabled:
            signals = self.components.enabled_signals()
            if signals != ["scene_embedding"]:
                raise ValueError(
                    "scoring.query_time_detection re-ranks the ranking of the BASE, "
                    "so scene_embedding must be the only enabled component; got "
                    f"{signals}")
            if weights.mode != "uniform":
                raise ValueError("scoring.query_time_detection requires "
                                 "weights.mode: uniform - the stage fuses its own "
                                 "two columns at 0.5/0.5, not by the pipeline weights")
        if weights.mode == "manual":
            if weights.manual is None:
                raise ValueError("weights.mode: manual requires weights.manual")
            if set(weights.manual) != enabled:
                raise ValueError(
                    "weights.manual keys must equal the enabled components: "
                    f"{sorted(enabled)}")
            if abs(sum(weights.manual.values()) - 1.0) > 1e-6:
                raise ValueError("weights.manual values must sum to 1.0")
        elif weights.manual is not None:
            raise ValueError("weights.manual must be null with weights.mode: uniform")
        return self


def load_experiment(file: Path | str, split: str | None = None) -> ExperimentConfig:
    """Loads and validates an experiment configuration; injects the split."""
    data = load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{file}: the configuration must be a YAML mapping")
    if split is not None:
        data["split"] = split
    return ExperimentConfig.model_validate(data)


def dump_resolved(config: ExperimentConfig, file: Path | str) -> Path:
    """Writes the resolved configuration of a run (defaults filled, split set)."""
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False,
                       allow_unicode=True),
        encoding="utf-8")
    return file
