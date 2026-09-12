"""The generator, the frozen decisions and the gate they guard (§09).

Everything here works on a frozen.yaml built in the test, never on the one in
``configs/``: that file is written by hand and is empty until the measurements
are done, so a test reading it would pass today for the wrong reason and fail in
a fortnight for another.

The regeneration test covers the GENERATED files only. Covering the development
ones would mean the generator had to know every variant that lost, which is
exactly what it must not know.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

from src.utils import experiments as exp
from src.utils import frozen as frozen_module
from src.utils.config import load_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_configs", ROOT / "scripts" / "make_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_configs = _generator()

DECIDED = {
    "matching": {"measure": "top1", "threshold": 0.72, "threshold_by_vocab": {}},
    "segmentation": {"strategy": "shots_histogram", "min_len_s": 3, "max_len_s": 15,
                     "window_s": 10, "histogram_threshold": 0.5},
    "scene_embedding": "openclip_vit_h14",
    "caption": {"chosen": "llava_1_5_7b", "rejected": "blip"},
    "objects": {"chosen": "yolo11", "rejected": "yoloe_promptfree"},
    "face_regions": {"chosen": "clip_regions", "rejected": "hsemotion"},
    "motion": "slowfast_k400",
}


@pytest.fixture
def frozen(tmp_path):
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump(DECIDED), encoding="utf-8")
    return frozen_module.load_frozen(path)


# -------------------------------------------------- what is frozen, and when
def test_the_file_is_written_in_two_stages_and_loads_between_them(tmp_path):
    """Most of the time this file exists it is half-filled; it has to load then."""
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump({"matching": DECIDED["matching"]}), encoding="utf-8")
    half = frozen_module.load_frozen(path)
    assert half.complete(frozen_module.STAGE_ONE)
    assert half.missing(frozen_module.STAGE_TWO) == list(frozen_module.STAGE_TWO)


def test_a_missing_file_is_an_empty_one_not_an_error(tmp_path):
    empty = frozen_module.load_frozen(tmp_path / "nothing.yaml")
    assert empty.missing(frozen_module.STAGE_ONE) == ["matching"]


def test_the_committed_decisions_load_and_validate():
    """`configs/frozen.yaml` is filled in by hand, so it is the one file here
    that can be wrong in a way no other test would see.

    What is asserted is the invariant, never the content: the file belongs to
    the author and fills up as the verdicts come in, so a test pinning what is
    in it would fail every time she decides something.
    """
    frozen = frozen_module.load_frozen()
    for key in frozen_module.STAGE_ONE + frozen_module.STAGE_TWO:
        assert hasattr(frozen, key), key
    for stage in (frozen_module.STAGE_ONE, frozen_module.STAGE_TWO):
        assert set(frozen.missing(stage)) <= set(stage)


def test_a_decision_between_one_option_is_refused(tmp_path):
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump({**DECIDED,
                                    "objects": {"chosen": "yolo11",
                                                "rejected": "yolo11"}}),
                    encoding="utf-8")
    with pytest.raises(Exception, match="chosen and rejected are the same"):
        frozen_module.load_frozen(path)


def test_the_segmentation_entry_is_the_whole_block(frozen):
    """The strategy name alone would emit a default None for the threshold.

    Twenty-four of the twenty-six development runs carry 0,5, and that parameter
    decides where fragments begin and end.
    """
    assert frozen.segmentation.histogram_threshold == 0.5
    generated = make_configs.planned(frozen)["full_tbbt.yaml"]
    assert generated["segmentation"]["histogram_threshold"] == 0.5


# ------------------------------------------------------------- the generator
def test_it_writes_fifty_seven_configurations(frozen):
    """The count follows from SIGNAL_DATASETS and moves when that table does.

    It was sixty-one while face regions still had a VATEX row. Dropping that row
    took four files -- the two halves of E5, the marginal one and the swap one --
    and the signal itself out of `full_vatex`.
    """
    files = make_configs.planned(frozen)
    assert len(files) == 57
    families = {}
    for spec in make_configs.specifications():
        families[spec["family"]] = families.get(spec["family"], 0) + 1
    assert families == {"development": 21, "individual": 12, "full": 3,
                        "marginal": 13, "swap": 8}
    # one per signal per dataset that signal exists on
    assert sum(1 for n in files if n.startswith("full_no_")) == sum(
        len(exp.datasets_of(signal)) for signal in exp.ADDED_SIGNALS)
    assert sum(1 for n in files if n.startswith("full_swap_")) == sum(
        len(exp.datasets_of(signal)) for signal in exp.SWAPPED_SIGNALS)
    assert sum(1 for n in files if n in
               {f"full_{d}.yaml" for d in exp.DATASETS}) == 3


def test_every_generated_file_validates(frozen, tmp_path):
    for name, config in make_configs.planned(frozen).items():
        path = tmp_path / name
        path.write_text(make_configs.render(config), encoding="utf-8")
        load_experiment(path, "test")


def test_regenerating_changes_nothing(frozen, tmp_path):
    """Compared as CONFIGURATIONS, not as text: key order is not a difference."""
    files = make_configs.planned(frozen)
    for name, config in files.items():
        (tmp_path / name).write_text(make_configs.render(config), encoding="utf-8")
    again = make_configs.planned(frozen)
    for name, config in again.items():
        assert make_configs.verdict(tmp_path / name,
                                    make_configs.render(config)) == "unchanged"


def test_identity_only_where_there_are_recurring_characters(frozen):
    files = make_configs.planned(frozen)
    assert files["full_vatex.yaml"]["components"]["identity"] == {"enabled": False}
    assert files["full_tbbt.yaml"]["components"]["identity"] == {"enabled": True}
    # and no marginal-contribution file for a signal that is not there
    assert "full_no_identity_vatex.yaml" not in files
    assert "full_no_identity_tbbt.yaml" in files


def test_motion_only_under_an_image_representation(tmp_path):
    """With a video base the motion signal is not what the base cannot see."""
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump({**DECIDED, "scene_embedding": "xclip_b32"}),
                    encoding="utf-8")
    files = make_configs.planned(frozen_module.load_frozen(path))
    assert files["full_tbbt.yaml"]["components"]["motion"] == {"enabled": False}


def test_vatex_has_no_segmentation_and_reports_by_kinetics(frozen):
    vatex = make_configs.planned(frozen)["full_vatex.yaml"]
    assert "segmentation" not in vatex
    assert vatex["evaluation"]["report_by"] == ["requirements", "complexity",
                                                "kinetics_vocab"]


def test_the_matching_block_goes_where_a_vocabulary_is_scored_against(frozen):
    files = make_configs.planned(frozen)
    assert "matching" in files["e2c_tbbt.yaml"]          # motion
    assert "matching" in files["full_tbbt.yaml"]          # objects and motion
    assert "matching" not in files["e6b_tbbt.yaml"]       # identity alone
    # and never to E4-D: it asks the detector the phrase itself, no threshold
    for dataset in exp.DATASETS:
        assert "matching" not in files[f"e4d_{dataset}.yaml"]


def test_the_swap_files_carry_the_mechanism_that_lost(frozen):
    files = make_configs.planned(frozen)
    assert files["full_swap_objects_tbbt.yaml"]["components"]["objects"] == {
        "enabled": True, "detector": "yoloe_promptfree"}
    assert files["full_swap_caption_tbbt.yaml"]["components"]["caption"] == {
        "enabled": True, "model": "blip"}


def test_it_refuses_to_work_on_decisions_that_are_not_made(tmp_path):
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump({"matching": DECIDED["matching"]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="segmentation"):
        frozen_module.require(frozen_module.STAGE_TWO, "the generator", path)


# ---------------------------------------- the manual files, by explicit list
def test_every_manual_configuration_loads():
    """By an explicit list, never a glob: vatex_base.yaml is outside the schema."""
    for name in make_configs.MANUAL_DEV + make_configs.MANUAL_EXTRA:
        path = CONFIGS / name
        assert path.exists(), name
        load_experiment(path, "dev")


def test_the_file_outside_the_schema_is_never_handed_to_the_validator():
    for name in make_configs.OUTSIDE_SCHEMA:
        assert (CONFIGS / name).exists()
        assert name not in make_configs.MANUAL_DEV + make_configs.MANUAL_EXTRA


# ----------------------------------------- round trip and the E4-D validator
def test_a_resolved_configuration_reads_back_the_same(tmp_path):
    from src.utils.config import dump_resolved

    for name in ("e2a_tbbt.yaml", "e5bp_office.yaml", "e2c_vatex.yaml"):
        config = load_experiment(CONFIGS / name, "dev")
        path = dump_resolved(config, tmp_path / f"resolved_{name}")
        assert load_experiment(path).model_dump() == config.model_dump()


def test_the_query_time_detector_may_not_share_the_ranking(frozen, tmp_path):
    """E4-D re-ranks the BASE, so nothing else may be switched on beside it."""
    config = dict(make_configs.planned(frozen)["e4d_tbbt.yaml"])
    config.pop("_description")
    config["components"]["objects"] = {"enabled": True, "detector": "yolo11"}
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(Exception, match="only enabled component"):
        load_experiment(path, "test")


def test_the_query_time_detector_requires_uniform_weights(frozen, tmp_path):
    config = dict(make_configs.planned(frozen)["e4d_tbbt.yaml"])
    config.pop("_description")
    config["scoring"]["weights"] = {"mode": "manual",
                                    "manual": {"scene_embedding": 1.0}}
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(Exception, match="uniform"):
        load_experiment(path, "test")


def test_sentence_queries_belong_to_the_region_mechanism_alone(tmp_path):
    config = yaml.safe_load((CONFIGS / "e5bp_tbbt.yaml").read_text(encoding="utf-8"))
    config["components"]["face_regions"]["mode"] = "hsemotion"
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(Exception, match="clip_regions"):
        load_experiment(path, "dev")


# --------------------------------- the pairs, and releasing family by family
def test_each_half_of_a_pair_carries_its_own_mechanism(frozen):
    """E3-B is BLIP and E3-C is LLaVA whatever E3 decides.

    The pair IS the comparison, so neither side may be read off the winner. It
    was: both halves came out with the chosen captioner, and the experiment
    compared a thing with itself.
    """
    files = make_configs.planned(frozen)
    assert files["e3b_vatex.yaml"]["components"]["caption"]["model"] == "blip"
    assert files["e3c_vatex.yaml"]["components"]["caption"]["model"] == "llava_1_5_7b"
    assert files["e4b_vatex.yaml"]["components"]["objects"]["detector"] == "yolo11"
    assert files["e4c_vatex.yaml"]["components"]["objects"]["detector"] \
        == "yoloe_promptfree"
    assert files["e5b_office.yaml"]["components"]["face_regions"]["mode"] \
        == "clip_regions"
    assert files["e5c_office.yaml"]["components"]["face_regions"]["mode"] == "hsemotion"


def test_the_mechanism_table_agrees_with_the_hand_written_files():
    """The generated VATEX half must say what the series half says."""
    for label, mechanism in exp.LABEL_MECHANISM.items():
        signal = exp.ADDS_SIGNAL[label]
        written = load_experiment(CONFIGS / f"{exp.config_stem(label, 'office')}.yaml",
                                  "dev")
        component = getattr(written.components, signal)
        assert getattr(component, exp.MECHANISM_FIELD[signal]) == mechanism, label


def test_an_individual_contribution_waits_only_for_what_it_reads(tmp_path):
    """Three of the six stage-two decisions wait for nothing.

    Segmentation and the base representation are the verdicts of E1 and E2,
    already in the thesis; motion has no alternative to choose between. Only the
    full-pipeline families read the verdicts of E4 and E5, so the all-or-nothing
    gate this replaces was thicker than the dependency.
    """
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump({"scene_embedding": "openclip_vit_h14"}),
                    encoding="utf-8")
    ready, blocked = make_configs.releasable(frozen_module.load_frozen(path))
    # every VATEX file whose label names its own mechanism or representation and
    # scores against no closed vocabulary, plus the query-time variant, which
    # reads no decision beyond the base
    assert set(ready) == {"e2a_vatex.yaml", "e3b_vatex.yaml", "e3c_vatex.yaml",
                          "e4d_vatex.yaml"}
    assert all(spec["family"] not in ("individual", "development")
               or spec["missing"] for spec in blocked)


def test_the_whole_individual_family_comes_out_without_the_e4_and_e5_verdicts(tmp_path):
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump({
        "scene_embedding": "openclip_vit_h14", "motion": "slowfast_k400",
        "segmentation": DECIDED["segmentation"], "matching": DECIDED["matching"]}),
        encoding="utf-8")
    ready, blocked = make_configs.releasable(frozen_module.load_frozen(path))
    assert len(ready) == 12 + 21          # the individual family and all of E2-E5
    assert {spec["family"] for spec in blocked} == {"full", "marginal", "swap"}
    # and what they wait for is exactly the two verdicts plus the caption one
    assert {key for spec in blocked for key in spec["missing"]} == {
        "caption", "objects", "face_regions"}


def test_a_blocked_file_names_the_decisions_it_lacks(tmp_path):
    path = tmp_path / "frozen.yaml"
    path.write_text(yaml.safe_dump({"scene_embedding": "openclip_vit_h14"}),
                    encoding="utf-8")
    _, blocked = make_configs.releasable(frozen_module.load_frozen(path))
    full_tbbt = next(spec for spec in blocked if spec["name"] == "full_tbbt.yaml")
    assert set(full_tbbt["missing"]) == {"caption", "objects", "face_regions",
                                         "matching", "motion", "segmentation"}


def test_the_requirements_are_the_ones_the_builder_actually_has(tmp_path):
    """Stated requirement met means the file builds -- for every file.

    A builder reading more than `needs_of` declares would make a file wait for a
    decision it does not use; reading less would let it be written from a
    decision nobody made. This walks every file with exactly its own stated
    requirements filled in and nothing else.
    """
    path = tmp_path / "frozen.yaml"
    for spec in make_configs.specifications():
        path.write_text(yaml.safe_dump({key: DECIDED[key] for key in spec["needs"]}),
                        encoding="utf-8")
        built = make_configs.build(spec, frozen_module.load_frozen(path))
        load_experiment_from(built, tmp_path / spec["name"])


def load_experiment_from(config: dict, path: Path):
    path.write_text(make_configs.render(config), encoding="utf-8")
    return load_experiment(path, "test")


# ---- the threshold block travels with the file, never written in afterwards
def test_every_file_that_scores_a_vocabulary_is_composed_with_the_threshold(frozen):
    """Nothing writes the block in afterwards, so nothing can forget to.

    There was once a separate mode for writing this block into the hand-written
    files, because some of them scored a closed vocabulary. It went away with
    them: every configuration that needs a threshold is generated today, and
    `compose` puts the block in at composition time.
    """
    from src.utils.config import ExperimentConfig

    wanted = frozen.matching.model_dump(exclude_none=True)
    checked = 0
    for spec in make_configs.specifications():
        config = make_configs.build(spec, frozen)
        temporary = ExperimentConfig.model_validate(
            {**{k: v for k, v in config.items() if k != "_description"},
             "matching": None})
        if temporary.needs_matching and not spec["detection"]:
            assert config.get("matching") == wanted, spec["name"]
            checked += 1
        else:
            assert "matching" not in config, spec["name"]
    assert checked > 30


def test_no_hand_written_configuration_needs_a_threshold(frozen):
    """Which is what let the mode that wrote them one be removed.

    E1 is the base representation alone and E5-Bp is face regions in an open
    vocabulary; neither scores against a closed one.
    """
    from src.utils.config import load_experiment

    for name in make_configs.MANUAL_DEV + make_configs.MANUAL_EXTRA:
        config = load_experiment(CONFIGS / name, "dev")
        assert not config.needs_matching, name
        assert config.matching is None, name
