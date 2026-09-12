"""Configuration contract validation."""

import pytest

from src.utils.config import (VOCABULARY_NAMES, ExperimentConfig, Matching,
                              load_experiment)

#: every HAND-WRITTEN configuration file, named explicitly. Everything else in
#: configs/ is either generated -- and the generator's to remember, not a list's
#: -- or outside this schema, like vatex_base.yaml, which a glob would sweep in.
#:
#: Ten are left. E1 needs a base representation and E2 is what decides one, so
#: a generated E1 file would carry a provisional value no verdict contains; E5-Bp
#: is a mode of one component rather than a variant of an experiment; and the
#: recommended composition is read off the test results rather than written from
#: a verdict, so no decision of frozen.yaml could generate it.
CONFIG_FILES = [f"{variant}_{dataset}"
                for variant in ("e1a", "e1b", "e1c", "e5bp", "recommended")
                for dataset in ("office", "tbbt")]

#: outside ExperimentConfig on purpose. `vatex_base` drives the implementation
#: correctness check, a different collection with its own shape; `frozen` is not
#: a variant at all but the decisions the generator writes variants FROM.
NOT_AN_EXPERIMENT = {"vatex_base", "frozen"}

VALID = {
    "experiment": "E1-A", "dataset": "tbbt", "seed": 1,
    "segmentation": {"strategy": "fixed_window"},
    "components": {"scene_embedding": {"enabled": True}},
    "evaluation": {"report_by": []},
}


def test_defaults_are_filled():
    config = ExperimentConfig.model_validate(VALID)
    assert config.segmentation.min_len_s == 3
    assert config.frames.step_s == 1.25
    assert config.components.caption.enabled is False


def test_unknown_key_is_rejected():
    with pytest.raises(Exception):
        ExperimentConfig.model_validate({**VALID, "strateg": "x"})


def test_threshold_on_wrong_strategy_is_rejected():
    bad = {**VALID, "segmentation": {"strategy": "fixed_window",
                                     "histogram_threshold": 0.5}}
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(bad)


def test_vatex_has_no_segmentation_section():
    """Nothing to configure when the recording is already the fragment.

    The development part is a part like any other: it settles the matching
    threshold and checks the motion signal, so the split is not restricted here.
    """
    with pytest.raises(Exception):
        ExperimentConfig.model_validate({**VALID, "dataset": "vatex"})
    ok = {k: v for k, v in VALID.items() if k != "segmentation"}
    for split in ("dev", "test"):
        config = ExperimentConfig.model_validate({**ok, "dataset": "vatex",
                                                  "split": split})
        assert config.segmentation is None


def test_manual_weights_must_match_enabled_components():
    bad = {**VALID, "scoring": {"weights": {"mode": "manual",
                                            "manual": {"caption": 1.0}}}}
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(bad)
    ok = {**VALID, "scoring": {"weights": {"mode": "manual",
                                           "manual": {"scene_embedding": 1.0}}}}
    assert ExperimentConfig.model_validate(ok)


def test_real_config_files_load(tmp_path):
    for name in ("e1a_tbbt", "e1b_office", "e1c_tbbt", "full_office", "full_tbbt"):
        config = load_experiment(f"configs/{name}.yaml", split="dev")
        assert config.split == "dev"


def test_the_full_pipeline_is_labelled_full():
    """The run directory and the thesis appendix both name it `full`, lowercase."""
    for name in ("full_office", "full_tbbt"):
        assert load_experiment(f"configs/{name}.yaml").experiment == "full"


# -------------------------------------------------------- the matching block
def test_every_hand_written_configuration_loads():
    """Six E1 files, the two E5-Bp controls and the two recommended ones."""
    assert len(CONFIG_FILES) == 10
    for name in CONFIG_FILES:
        load_experiment(f"configs/{name}.yaml", split="dev")


def test_a_configuration_carries_the_threshold_exactly_where_it_needs_one():
    """The block is optional in the SCHEMA -- run_features.py and measure_cost.py
    read these files and neither produces a number a threshold could bend -- but
    a file that scores against a closed vocabulary has to have it, and it has to
    be the frozen one.

    Both halves matter. A missing block stops a run at `_require_matching`; a
    block in a file that needs none would be a threshold nobody applies, quietly
    disagreeing with frozen.yaml.
    """
    from src.utils.frozen import load_frozen

    frozen = load_frozen()
    for name in CONFIG_FILES:
        config = load_experiment(f"configs/{name}.yaml", split="dev")
        if config.needs_matching and frozen.matching is not None:
            assert config.matching is not None, f"{name} scores against a vocabulary"
            assert config.matching.model_dump() == frozen.matching.model_dump(), name
        elif not config.needs_matching:
            assert config.matching is None, f"{name} needs no threshold"


def test_matching_requires_both_a_measure_and_a_threshold():
    """Neither has a default: both come out of one measurement."""
    for incomplete in ({"measure": "top1"}, {"threshold": 0.25}):
        with pytest.raises(Exception):
            Matching.model_validate(incomplete)
    assert Matching(measure="top1", threshold=0.25).threshold_by_vocab == {}


def test_threshold_by_vocab_keys_must_name_a_vocabulary_cache():
    with pytest.raises(Exception, match="threshold_by_vocab keys"):
        Matching(measure="top1", threshold=0.25,
                 threshold_by_vocab={"objects_yolo12": 0.4})


def test_a_vocabulary_threshold_may_be_lower_than_the_shared_one():
    """Each vocabulary is calibrated on its own judged sample, so it lands where
    it lands. The measured four are 0,805 / 0,733 / 0,805 / 0,802 -- two of them
    below the first, which an earlier design (one threshold, raised per
    vocabulary where an unrelated phrase landed too easily) would have refused.
    """
    lower = Matching(measure="top1", threshold=0.8054,
                     threshold_by_vocab={"objects_yoloe_promptfree": 0.7330,
                                         "actions_kinetics400": 0.8017})
    assert lower.threshold_by_vocab["objects_yoloe_promptfree"] == 0.7330
    # the keys are still checked against the four vocabulary caches
    with pytest.raises(Exception, match="vocabulary caches"):
        Matching(measure="top1", threshold=0.25, threshold_by_vocab={"nonsense": 0.9})


def test_the_vocabulary_names_are_the_caches_components_builds():
    """The schema repeats them because components.py imports this module."""
    from src.runners import components

    built = {components.OBJECT_VOCABULARY.format(detector=detector)
             for detector in ("yolo11", "yoloe_promptfree")}
    built |= {components.EXPRESSION_VOCABULARY, components.MOTION_VOCABULARY}
    assert built == set(VOCABULARY_NAMES)


# -------------------------------------------------------------- who needs it
def _with(components=None, scoring=None):
    data = {**VALID, "components": {"scene_embedding": {"enabled": True},
                                    **(components or {})}}
    if scoring is not None:
        data["scoring"] = scoring
    return ExperimentConfig.model_validate(data)


def test_which_configurations_need_a_matching_block():
    assert not _with().needs_matching                       # the base alone
    assert not _with({"caption": {"enabled": True}}).needs_matching
    assert not _with({"identity": {"enabled": True}}).needs_matching
    assert _with({"objects": {"enabled": True}}).needs_matching
    assert _with({"motion": {"enabled": True}}).needs_matching

    # face regions only in the closed-vocabulary mode; clip_regions has no class
    # names for a threshold to act on
    assert _with({"face_regions": {"enabled": True,
                                   "mode": "hsemotion"}}).needs_matching
    assert not _with({"face_regions": {"enabled": True,
                                       "mode": "clip_regions"}}).needs_matching

    # the query-time detector answers the phrase itself, without a vocabulary
    assert not _with(scoring={"query_time_detection": {"enabled": True}}).needs_matching


def test_a_run_is_refused_when_the_block_it_needs_is_missing():
    """The schema lets it through; the runner, where the numbers are made, does not."""
    from src.runners import run

    needs_it = _with({"objects": {"enabled": True}})
    with pytest.raises(ValueError, match="matching block"):
        run._require_matching(needs_it, "configs/e4b_office.yaml")

    needs_it.matching = Matching(measure="top1", threshold=0.25)
    run._require_matching(needs_it, "configs/e4b_office.yaml")     # accepted with it
    run._require_matching(_with(), "configs/e2a_office.yaml")      # never needed it


def test_a_resolved_configuration_still_round_trips(tmp_path):
    """The dump gained a `matching` key; it has to read back as it was written."""
    from src.utils.config import dump_resolved

    for name in ("e4b_office", "full_tbbt"):
        config = load_experiment(f"configs/{name}.yaml", split="dev")
        written = dump_resolved(config, tmp_path / f"{name}.resolved.yaml")
        assert load_experiment(written).model_dump() == config.model_dump()


def test_the_explicit_list_covers_every_hand_written_configuration():
    """A glob would sweep in vatex_base.yaml, which is not this schema.

    So the list is written out, and this is what notices when a HAND-WRITTEN
    file is added to configs/ and nobody adds it here. The generated ones are
    subtracted: they arrive and leave with `scripts/make_configs.py --execute`
    and registering each of the thirty-nine by hand would make the generator
    pointless.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "make_configs", Path("scripts") / "make_configs.py")
    make_configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_configs)
    generated = {Path(entry["name"]).stem for entry in make_configs.specifications()}

    on_disk = {path.stem for path in Path("configs").glob("*.yaml")}
    assert on_disk - NOT_AN_EXPERIMENT - generated == set(CONFIG_FILES)

