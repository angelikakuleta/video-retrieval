"""What the evaluation layer gained: intersection, confirmation, transitions.

Kept apart from test_compare.py, which pins the paired interval and the decision
rule the development phase already runs on.
"""

import json
import math

import numpy as np
import pytest

from src.evaluation.compare import (
    activation_share,
    active_ids,
    compare_variant,
    confirmed,
    contrast_variant,
    subset_ids,
    transitions,
    two_sample_interval,
)
from src.evaluation.metrics import recall_at


def run_stub(dataset, values_by_episode, metric="recall@10"):
    per_query, desc_id = [], 0
    for episode, values in values_by_episode.items():
        for value in values:
            per_query.append({"desc_id": desc_id, "episode": episode, metric: value})
            desc_id += 1
    return {"per_query": per_query, "dataset": dataset}


def flat(dataset, values, metric="recall@10"):
    """One episode per value, so the episode unit and the query unit agree."""
    return run_stub(dataset, {f"e{i}": [v] for i, v in enumerate(values)}, metric)


# ----------------------------------------------------- comparing what exists
def test_a_candidate_missing_a_dataset_is_compared_on_the_rest():
    """It used to raise KeyError, which turned "VATEX is downloading" into a crash."""
    reference = {"tbbt": flat("tbbt", [0.5] * 6), "office": flat("office", [0.5] * 6)}
    candidate = {"tbbt": flat("tbbt", [0.7] * 6)}

    result = compare_variant(candidate, reference, "recall@10")
    assert sorted(result["per_dataset"]) == ["tbbt"]
    assert result["pooled_datasets"] == ["tbbt"]
    assert math.isclose(result["per_dataset"]["tbbt"]["mean"], 0.2)


def test_the_pool_names_only_the_datasets_it_actually_pooled():
    reference = {"tbbt": flat("tbbt", [0.5] * 6), "vatex": flat("vatex", [0.5] * 6)}
    candidate = {"tbbt": flat("tbbt", [0.7] * 6), "vatex": flat("vatex", [0.7] * 6)}

    result = compare_variant(candidate, reference, "recall@10")
    assert result["pooled_datasets"] == ["tbbt"]        # clips never join the pool


# ------------------------------------------------------ a confirmed contrast
def tagged(dataset, tagged_values, rest_values):
    """Two queries per episode: the first tagged, the second not."""
    per_query, desc_id = [], 0
    for i, (a, b) in enumerate(zip(tagged_values, rest_values)):
        per_query.append({"desc_id": desc_id, "episode": f"e{i}", "recall@10": a})
        per_query.append({"desc_id": desc_id + 1, "episode": f"e{i}", "recall@10": b})
        desc_id += 2
    return {"per_query": per_query, "dataset": dataset}


def contrast_of(tbbt_gain, office_gain):
    """A contrast where the tagged queries gain `gain` more than the rest."""
    reference, candidate, matching, rest = {}, {}, {}, {}
    for dataset, gain in (("tbbt", tbbt_gain), ("office", office_gain)):
        n = 6
        reference[dataset] = tagged(dataset, [0.5] * n, [0.5] * n)
        candidate[dataset] = tagged(dataset, [0.5 + gain] * n, [0.5] * n)
        ids = [r["desc_id"] for r in candidate[dataset]["per_query"]]
        matching[dataset] = set(ids[0::2])
        rest[dataset] = set(ids[1::2])
    return contrast_variant(candidate, reference, "recall@10", matching, rest)


def test_a_contrast_holds_when_both_series_agree_and_the_pool_excludes_zero():
    assert confirmed(contrast_of(0.2, 0.2))


def test_a_contrast_fails_when_one_series_points_the_other_way():
    """The pooled interval can still clear zero while a series disagrees."""
    contrast = contrast_of(0.4, -0.05)
    assert contrast["per_dataset"]["office"]["mean"] < 0
    assert not confirmed(contrast)


def test_a_contrast_fails_when_the_pooled_interval_covers_zero():
    assert not confirmed(contrast_of(0.0, 0.0))


def test_a_confirmed_contrast_reads_series_only(monkeypatch):
    """VATEX carries neither wymaga_osoby nor wymaga_mimiki; it cannot confirm."""
    contrast = contrast_of(0.2, 0.2)
    contrast["per_dataset"]["vatex"] = {"unit": "clip", "mean": -9.0,
                                        "low": -9.0, "high": -9.0}
    assert contrast["pooled_datasets"] == ["office", "tbbt"]
    assert confirmed(contrast)                # the clip row takes no part


# -------------------------------------------- the two-sample interval of PB3
def test_two_sample_interval_against_a_hand_computed_value():
    """a = 1..5, b = 2..6: both variances 2.5, so the pooled one is 2.5 too.

    se = sqrt(2.5 * (1/5 + 1/5)) = 1, df = 8, t(0.975, 8) = 2.3060.
    """
    result = two_sample_interval([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert result["df"] == 8 and result["n"] == 10
    assert math.isclose(result["mean"], -1.0)
    assert math.isclose(result["high"] - result["mean"], 2.3060, abs_tol=1e-3)


def test_the_pooled_variance_is_not_welch():
    """Unequal group sizes and variances: Welch would give a different df.

    The thesis states one pooled figure, so the df here is na + nb - 2 whatever
    the variances do.
    """
    tight = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    wide = [0.0, 20.0, 0.0, 20.0]
    result = two_sample_interval(tight, wide)
    assert result["df"] == 8 + 4 - 2
    assert result["method"] == "t2"


def test_a_group_of_one_gets_no_interval():
    result = two_sample_interval([1.0], [2.0, 3.0])
    assert result["mean"] is None and result["low"] is None


# ------------------------------------------ who was fixed and who was broken
def transition_runs():
    reference = {"per_query": [{"desc_id": i, "episode": "e0", "recall@10": v}
                               for i, v in enumerate([1.0, 1.0, 0.0, 0.0, 0.0])]}
    candidate = {"per_query": [{"desc_id": i, "episode": "e0", "recall@10": v}
                               for i, v in enumerate([1.0, 0.0, 1.0, 1.0, 0.0])]}
    return candidate, reference


def test_transitions_split_the_queries_into_four_disjoint_groups():
    candidate, reference = transition_runs()
    counts = transitions(candidate, reference)

    assert counts == {"1->1": 1, "1->0": 1, "0->1": 2, "0->0": 1}
    assert sum(counts.values()) == 5
    # the mean difference is what the four groups add up to
    assert math.isclose((counts["0->1"] - counts["1->0"]) / 5, (3 - 2) / 5)


def test_transitions_can_be_taken_on_a_subset():
    candidate, reference = transition_runs()
    counts = transitions(candidate, reference, desc_ids={2, 3})
    assert counts == {"0->1": 2, "1->0": 0, "1->1": 0, "0->0": 0}


def test_a_query_only_one_run_answered_is_refused_not_dropped():
    """The four counts are the n of the appendix table; a short sum would lie."""
    candidate, reference = transition_runs()
    reference["per_query"] = reference["per_query"][:3]
    with pytest.raises(ValueError, match="different queries"):
        transitions(candidate, reference)

    # narrowing the scope on purpose is how a partial comparison is asked for
    counts = transitions(candidate, reference, desc_ids={0, 1, 2})
    assert sum(counts.values()) == 3


def test_a_metric_that_is_not_binary_is_refused():
    """The four groups are only exhaustive because R@10 of one query is 0 or 1."""
    candidate, reference = transition_runs()
    candidate["per_query"][0]["recall@10"] = 0.5
    with pytest.raises(ValueError, match="not binary"):
        transitions(candidate, reference)


# -------------------------------------------------- how often a signal spoke
def signals_run(tmp_path, names, active, desc_ids):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.savez_compressed(run_dir / "signals.npz",
                        names=np.array(names),
                        desc_ids=np.array(desc_ids, dtype=np.int64),
                        active=np.asarray(active, dtype=bool),
                        values=np.zeros((len(names), len(desc_ids), 1), np.float16))
    return {"dir": run_dir}


def test_activation_share_is_the_fraction_of_queries_a_signal_spoke_about(tmp_path):
    run = signals_run(tmp_path, ["scene_embedding", "objects"],
                      [[True, True, True, True], [True, False, False, False]],
                      [10, 11, 12, 13])
    shares = activation_share(run)
    assert shares["scene_embedding"] == 1.0
    assert shares["objects"] == 0.25


def test_activation_share_on_the_targeted_queries_only(tmp_path):
    run = signals_run(tmp_path, ["objects"], [[True, True, False, False]],
                      [10, 11, 12, 13])
    assert activation_share(run, desc_ids={10, 11}) == {"objects": 1.0}
    assert activation_share(run, desc_ids={12, 13}) == {"objects": 0.0}


def test_an_empty_subset_gives_no_share_rather_than_a_zero(tmp_path):
    """Nobody asked is not the same answer as nobody answered."""
    run = signals_run(tmp_path, ["objects"], [[True, True]], [10, 11])
    assert math.isnan(activation_share(run, desc_ids=set())["objects"])


def test_active_ids_names_the_queries_rather_than_counting_them(tmp_path):
    """The share answers "how often"; a row of chapter 6 asks "which"."""
    run = signals_run(tmp_path, ["face_regions"], [[True, False, True, False]],
                      [10, 11, 12, 13])
    assert active_ids(run, "face_regions") == {10, 12}


def test_the_two_face_mechanisms_intersect_on_the_queries_both_spoke_about(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    regions = signals_run(tmp_path / "b", ["face_regions"],
                          [[True, True, False, False]], [10, 11, 12, 13])
    hsemotion = signals_run(tmp_path / "c", ["face_regions"],
                            [[True, False, True, False]], [10, 11, 12, 13])
    both = active_ids(regions, "face_regions") & active_ids(hsemotion, "face_regions")
    assert both == {10}


def test_a_signal_the_run_does_not_carry_has_an_empty_set_not_an_error(tmp_path):
    """The block that asks is the one that reports the absence."""
    run = signals_run(tmp_path, ["scene_embedding"], [[True, True]], [10, 11])
    assert active_ids(run, "face_regions") == set()


# ------------------------------------------------------------------- subsets
def queries_file(tmp_path, monkeypatch, dataset, records):
    from src.data import datasets

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    folder = tmp_path / "data" / "annotations" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{dataset}_queries_dev.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def query_record(desc_id, **fields):
    return {"desc_id": desc_id, "desc": "x", "vid_name": f"clip{desc_id}",
            "ts": [0.0, 10.0], "source": "vatex", "requirements": [],
            "complexity": None, "identities": [], "event_id": "", **fields}


def test_identities_selects_the_queries_that_call_up_a_character(tmp_path, monkeypatch):
    queries_file(tmp_path, monkeypatch, "tbbt", [
        query_record(1, source="tbbt", identities=[]),
        query_record(2, source="tbbt", identities=["Sheldon"]),
        query_record(3, source="tbbt", identities=["Sheldon", "Penny"]),
    ])
    one, _ = subset_ids("tbbt", "dev", "identities:>=1")
    two, rest = subset_ids("tbbt", "dev", "identities:>=2")
    assert one == {2, 3} and two == {3} and rest == {1, 2}


def test_kinetics_vocab_splits_the_clips_by_what_slowfast_knows(tmp_path, monkeypatch):
    from src.utils import vatex

    queries_file(tmp_path, monkeypatch, "vatex",
                 [query_record(1), query_record(2), query_record(3)])
    monkeypatch.setattr(vatex, "k400_membership",
                        lambda part: {"clip1": True, "clip2": False})
    known, other = subset_ids("vatex", "dev", "kinetics_vocab:yes")
    assert known == {1}
    assert other == {2, 3}                 # clip3 is absent from the split file


def test_kinetics_vocab_is_refused_for_a_series(tmp_path, monkeypatch):
    queries_file(tmp_path, monkeypatch, "tbbt", [query_record(1, source="tbbt")])
    with pytest.raises(ValueError, match="VATEX only"):
        subset_ids("tbbt", "dev", "kinetics_vocab:yes")


def test_vatex_carries_neither_person_nor_expression_tags(tmp_path, monkeypatch):
    """A regression test: those two columns do not exist in the VATEX tag file.

    An empty subset is the right answer, not an error and not a crash -- a
    one-sentence clip caption settles neither of them.
    """
    queries_file(tmp_path, monkeypatch, "vatex", [
        query_record(1, requirements=["wymaga_obiektu"]),
        query_record(2, requirements=["wymaga_ruchu"]),
    ])
    for tag in ("wymaga_osoby", "wymaga_mimiki"):
        matching, rest = subset_ids("vatex", "dev", f"requirements:{tag}")
        assert matching == set()
        assert rest == {1, 2}


def test_complexity_still_works_and_a_misspelling_still_does_not(tmp_path, monkeypatch):
    queries_file(tmp_path, monkeypatch, "vatex", [
        query_record(1, complexity="P"), query_record(2, complexity="Z")])
    simple, complex_ = subset_ids("vatex", "dev", "complexity:P")
    assert simple == {1} and complex_ == {2}
    with pytest.raises(ValueError, match="unknown complexity value"):
        subset_ids("vatex", "dev", "complexity:trudne")


# ----------------------------------------------------------- recall at any k
def test_recall_at_reads_the_rank_that_is_already_stored():
    per_query = [{"rank": 1}, {"rank": 12}, {"rank": 48}, {"rank": None}]
    assert recall_at(per_query, 10) == 0.25
    assert recall_at(per_query, 50) == 0.75            # the E4-D ceiling
    assert recall_at(per_query, 1) == 0.25


def test_recall_at_of_no_queries_is_not_zero():
    assert recall_at([], 10) is None


def test_recall_at_refuses_a_meaningless_cut_off():
    with pytest.raises(ValueError, match="at least 1"):
        recall_at([{"rank": 1}], 0)


# ------ the script's own decision: which dataset a contrast can still run on
def compare_runs_module():
    """scripts/ is not a package, so the file is loaded by path."""
    import importlib.util
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
    spec = importlib.util.spec_from_file_location("compare_runs_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_empty_subset_drops_one_dataset_not_the_whole_contrast(tmp_path,
                                                                  monkeypatch):
    """VATEX carries no wymaga_mimiki; the two series still have a contrast.

    Dropping it for everyone would throw away the material the claim is about.
    """
    script = compare_runs_module()
    queries_file(tmp_path, monkeypatch, "tbbt", [
        query_record(1, source="tbbt", requirements=["wymaga_mimiki"]),
        query_record(2, source="tbbt", requirements=[])])
    queries_file(tmp_path, monkeypatch, "vatex", [query_record(3)])

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    matching, rest, usable = script.contrast_subsets(
        "requirements:wymaga_mimiki", ["tbbt", "vatex"])

    assert usable == ["tbbt"]
    assert matching == {"tbbt": {1}} and rest == {"tbbt": {2}}
    assert any("vatex" in message for message in printed)

