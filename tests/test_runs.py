"""The run loader and the table of experiment facts.

The loader is the only place that decides which directory a number came from, so
what it does with a repeated run and with a label that covers fewer datasets is
worth pinning down.
"""

import json

import pytest

from src.evaluation import runs as runs_module
from src.utils import experiments as exp


def write_run(root, label, dataset, split="dev", stamp="20260101_000000",
              per_query=None, finished=True):
    """A run directory with just enough in it for `compare.load_run`."""
    run_dir = root / f"{label}_{dataset}_{split}_{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(
        f"experiment: {label}\ndataset: {dataset}\nsplit: {split}\n", encoding="utf-8")
    records = per_query if per_query is not None else [
        {"desc_id": 1, "episode": "s01e01", "recall@10": 1.0, "rank": 1}]
    (run_dir / "per_query.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8")
    if finished:
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    return run_dir


@pytest.fixture
def results(tmp_path, monkeypatch):
    """``results/runs`` under tmp_path, so load_runs finds only what a test wrote."""
    from src.data import datasets

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    folder = tmp_path / "results" / "runs"
    folder.mkdir(parents=True)
    return folder


# ----------------------------------------------------------------- load_runs
def test_runs_are_grouped_by_label_and_dataset(results):
    write_run(results, "E1-A", "tbbt")
    write_run(results, "E1-A", "office")
    write_run(results, "E1-B", "tbbt")

    found = runs_module.load_runs("dev", log=lambda *_: None)
    assert sorted(found) == ["E1-A", "E1-B"]
    assert sorted(found["E1-A"]) == ["office", "tbbt"]


def test_the_newest_directory_of_a_repeated_run_wins(results):
    write_run(results, "E1-A", "tbbt", stamp="20260101_000000")
    newer = write_run(results, "E1-A", "tbbt", stamp="20260307_120000")

    messages = []
    found = runs_module.load_runs("dev", log=messages.append)
    assert found["E1-A"]["tbbt"]["dir"].name == newer.name
    assert any("skipped older run" in message for message in messages)


def test_a_run_of_another_split_is_not_loaded(results):
    write_run(results, "E1-A", "tbbt", split="dev")
    write_run(results, "E1-A", "office", split="test")

    found = runs_module.load_runs("dev", log=lambda *_: None)
    assert sorted(found["E1-A"]) == ["tbbt"]


def test_an_interrupted_run_is_named_and_skipped(results):
    write_run(results, "E1-A", "tbbt")
    write_run(results, "E1-A", "office", finished=False)

    messages = []
    found = runs_module.load_runs("dev", log=messages.append)
    assert sorted(found["E1-A"]) == ["tbbt"]
    assert any("no metrics.json" in message for message in messages)


def test_named_directories_are_loaded_instead_of_the_whole_folder(results):
    write_run(results, "E1-A", "tbbt")
    only = write_run(results, "E1-B", "tbbt")

    found = runs_module.load_runs("dev", [only], log=lambda *_: None)
    assert sorted(found) == ["E1-B"]


# ----------------------------------------------------------------- available
def test_available_narrows_to_the_shared_datasets(results):
    write_run(results, "E1-A", "tbbt")
    write_run(results, "E1-A", "office")
    write_run(results, "E1-B", "tbbt")
    runs = runs_module.load_runs("dev", log=lambda *_: None)

    labels, datasets, skipped = runs_module.available(runs, ["E1-A", "E1-B"])
    assert labels == ["E1-A", "E1-B"]
    assert datasets == ["tbbt"]                 # office is not shared
    assert ("E1-A", ) not in skipped            # E1-A takes part, minus one dataset
    assert any(label == "E1-A" and "office" in reason for label, reason in skipped)


def test_a_label_without_any_run_is_reported_not_raised(results):
    write_run(results, "E1-A", "tbbt")
    runs = runs_module.load_runs("dev", log=lambda *_: None)

    labels, datasets, skipped = runs_module.available(runs, ["E1-A", "E1-C"])
    assert labels == ["E1-A"] and datasets == ["tbbt"]
    assert skipped == [("E1-C", "no run for this split")]


def test_labels_sharing_no_dataset_leave_nothing_to_compare(results):
    write_run(results, "E1-A", "tbbt")
    write_run(results, "E1-B", "office")
    runs = runs_module.load_runs("dev", log=lambda *_: None)

    labels, datasets, skipped = runs_module.available(runs, ["E1-A", "E1-B"])
    assert labels == [] and datasets == []
    assert len(skipped) == 2


def test_the_datasets_come_back_in_the_order_the_thesis_lists_them(results):
    for dataset in ("vatex", "office", "tbbt"):
        write_run(results, "E2-A", dataset)
    runs = runs_module.load_runs("dev", log=lambda *_: None)

    _, datasets, _ = runs_module.available(runs, ["E2-A"])
    assert datasets == ["tbbt", "office", "vatex"]


def test_select_narrows_the_mapping_to_what_was_agreed(results):
    write_run(results, "E1-A", "tbbt")
    write_run(results, "E1-A", "office")
    runs = runs_module.load_runs("dev", log=lambda *_: None)

    chosen = runs_module.select(runs, ["E1-A"], ["tbbt"])
    assert sorted(chosen["E1-A"]) == ["tbbt"]


# -------------------------------------------------------- the table of facts
def test_the_baseline_is_one_label_for_all_three_datasets():
    assert exp.BASE == "E2-A"
    assert exp.DATASETS == ("tbbt", "office", "vatex")


def test_identity_exists_only_where_there_are_characters_to_profile():
    assert exp.datasets_of("identity") == exp.SERIES
    assert exp.datasets_of("objects") == exp.DATASETS
    assert "vatex" not in exp.datasets_of("identity")


def test_the_three_contributions_are_three_pairs_of_labels():
    assert exp.individual_pair("E4-B") == ("E4-B", "E2-A")
    assert exp.marginal_pair("objects") == ("full", "full_no_objects")
    assert exp.swap_pair("objects") == ("full", "full_swap_objects")


def test_a_signal_with_no_second_mechanism_cannot_be_swapped():
    """Motion and identity have one mechanism each; there is nothing to swap in."""
    assert exp.swap_pair("caption") == ("full", "full_swap_caption")
    with pytest.raises(ValueError, match="no second mechanism"):
        exp.swap_pair("motion")
    with pytest.raises(ValueError, match="adds no single signal"):
        exp.individual_pair("full")


def test_configuration_file_names_follow_one_convention():
    assert exp.config_stem("E2-A", "vatex") == "e2a_vatex"
    assert exp.config_stem("E5-Bp", "tbbt") == "e5bp_tbbt"
    assert exp.config_stem(exp.no_label("identity"), "office") == "full_no_identity_office"
    assert exp.config_stem(exp.swap_label("caption"), "vatex") == "full_swap_caption_vatex"
    with pytest.raises(ValueError, match="unknown dataset"):
        exp.config_stem("E2-A", "tbbt2")


def test_display_names_are_plain_ascii():
    """They reach the console, which on Windows mangles anything else."""
    names = (list(exp.DATASET_NAMES.values()) + list(exp.STRATEGY_NAMES.values())
             + list(exp.LABEL_NAMES.values()) + list(exp.SIGNAL_NAMES.values())
             + [exp.display(exp.no_label("caption")), exp.display("E9-Z")])
    assert all(name.isascii() for name in names)
    assert exp.display("E9-Z") == "E9-Z"                 # unknown label speaks for itself
    assert exp.display(exp.no_label("caption")) == "pelny bez: opisy scen"


def test_every_variant_of_a_signal_names_its_component():
    """A table putting two variants of one signal side by side has to say which."""
    named = set(exp.LABEL_COMPONENT)
    assert set(exp.ADDS_SIGNAL) <= named, "a variant with no component name"
    assert all(name.isascii() for name in exp.LABEL_COMPONENT.values())
    # the two variants of a swapped signal may not carry the same name, or the
    # comparison between them would read as a thing against itself
    for signal in exp.SWAPPED_SIGNALS:
        variants = [label for label, added in exp.ADDS_SIGNAL.items() if added == signal]
        assert len({exp.LABEL_COMPONENT[label] for label in variants}) == len(variants)


def test_every_label_that_adds_a_signal_names_a_signal_that_exists():
    assert set(exp.ADDS_SIGNAL.values()) <= set(exp.ADDED_SIGNALS)
    assert set(exp.SWAPPED_SIGNALS) <= set(exp.ADDED_SIGNALS)
    assert exp.BASE_SIGNAL not in exp.ADDED_SIGNALS
