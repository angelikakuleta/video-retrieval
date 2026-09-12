"""Paired episode differences, t interval and the decision rule."""

import math

from src.evaluation.compare import (
    bootstrap_interval,
    compare_variant,
    contrast_variant,
    decide,
    episode_means,
    inference_unit,
    t_interval,
    wins,
)


def run_stub(dataset, values_by_episode):
    per_query = [{"desc_id": i, "episode": ep, "recall@10": v}
                 for i, (ep, v) in enumerate(
                     (ep, v) for ep, vs in values_by_episode.items() for v in vs)]
    return {"per_query": per_query, "dataset": dataset}


def test_episode_means():
    run = run_stub("tbbt", {"s01e01": [1.0, 0.0], "s01e02": [1.0]})
    assert episode_means(run["per_query"], "recall@10") == {
        "s01e01": 0.5, "s01e02": 1.0}


def test_t_interval_known_value():
    # diffs 1, 2, 3: mean 2, sd 1, t(0.975, df=2) = 4.3027 -> half-width 2.4841
    ci = t_interval([1.0, 2.0, 3.0])
    assert ci["n"] == 3 and ci["mean"] == 2.0
    assert math.isclose(ci["high"] - ci["mean"], 2.4841, abs_tol=1e-3)


def test_decision_rule():
    reference = {"tbbt": run_stub("tbbt", {f"e{i}": [0.5] for i in range(6)}),
                 "office": run_stub("office", {f"e{i}": [0.5] for i in range(6)})}
    better = {"tbbt": run_stub("tbbt", {f"e{i}": [0.7 + 0.01 * i] for i in range(6)}),
              "office": run_stub("office", {f"e{i}": [0.7] for i in range(6)})}
    mixed = {"tbbt": run_stub("tbbt", {f"e{i}": [0.7] for i in range(6)}),
             "office": run_stub("office", {f"e{i}": [0.4] for i in range(6)})}

    result_better = compare_variant(better, reference, "recall@10")
    result_mixed = compare_variant(mixed, reference, "recall@10")
    assert wins(result_better)
    assert not wins(result_mixed)          # direction disagrees between serials

    verdict = decide({"B": result_mixed, "C": result_better}, ["A", "B", "C"])
    assert verdict["winner"] == "C"
    verdict = decide({"B": result_mixed}, ["A", "B"])
    assert verdict["winner"] == "A"        # nobody beat the reference -> simplest


# ----------------------------------------------------- confirmatory contrast

def tagged_run(dataset, per_episode):
    """``per_episode`` maps an episode to ``(tagged values, untagged values)``.

    Query identifiers are laid out so that the tagged ones are even and the
    untagged odd, which keeps the subsets easy to name in the tests.
    """
    per_query, next_id = [], 0
    for episode, (tagged, untagged) in per_episode.items():
        for value in tagged:
            per_query.append({"desc_id": next_id * 2, "episode": episode,
                              "recall@10": value})
            next_id += 1
        for value in untagged:
            per_query.append({"desc_id": next_id * 2 + 1, "episode": episode,
                              "recall@10": value})
            next_id += 1
    return {"per_query": per_query, "dataset": dataset}


def ids_of(run, even):
    return {r["desc_id"] for r in run["per_query"] if (r["desc_id"] % 2 == 0) == even}


def test_contrast_is_the_difference_of_the_two_differences():
    episodes = {f"e{i}": ([0.0, 0.0], [0.0, 0.0]) for i in range(5)}
    reference = {"tbbt": tagged_run("tbbt", episodes)}
    # the candidate gains 0.5 on the tagged queries and nothing on the rest
    candidate = {"tbbt": tagged_run("tbbt", {f"e{i}": ([0.5, 0.5], [0.0, 0.0])
                                             for i in range(5)})}
    matching = {"tbbt": ids_of(reference["tbbt"], True)}
    rest = {"tbbt": ids_of(reference["tbbt"], False)}

    result = contrast_variant(candidate, reference, "recall@10", matching, rest)
    entry = result["per_dataset"]["tbbt"]
    assert math.isclose(entry["matching_delta"], 0.5)
    assert math.isclose(entry["rest_delta"], 0.0)
    assert math.isclose(entry["mean"], 0.5)
    assert entry["low"] is not None and entry["low"] > 0     # the contrast holds


def test_contrast_is_zero_when_the_effect_is_spread_evenly():
    reference = {"tbbt": tagged_run("tbbt", {f"e{i}": ([0.0], [0.0]) for i in range(5)})}
    candidate = {"tbbt": tagged_run("tbbt", {f"e{i}": ([0.4], [0.4]) for i in range(5)})}
    result = contrast_variant(candidate, reference, "recall@10",
                              {"tbbt": ids_of(reference["tbbt"], True)},
                              {"tbbt": ids_of(reference["tbbt"], False)})
    entry = result["per_dataset"]["tbbt"]
    assert math.isclose(entry["mean"], 0.0, abs_tol=1e-12)


def test_an_episode_without_tagged_queries_drops_out_of_the_contrast():
    episodes = {f"e{i}": ([0.0], [0.0]) for i in range(4)}
    episodes["e_no_tag"] = ([], [0.0, 0.0])
    reference = {"tbbt": tagged_run("tbbt", episodes)}
    candidate = {"tbbt": tagged_run("tbbt", episodes)}
    result = contrast_variant(candidate, reference, "recall@10",
                              {"tbbt": ids_of(reference["tbbt"], True)},
                              {"tbbt": ids_of(reference["tbbt"], False)})
    assert result["per_dataset"]["tbbt"]["n"] == 4


# --------------------------------------------------------- unit of inference

def test_vatex_uses_a_bootstrap_over_clips_and_stays_out_of_the_pool():
    assert inference_unit("vatex") == "clip"
    assert inference_unit("tbbt") == "episode"

    reference = {"vatex": run_stub("vatex", {"c": [0.0] * 40}),
                 "tbbt": run_stub("tbbt", {f"e{i}": [0.5] for i in range(6)})}
    candidate = {"vatex": run_stub("vatex", {"c": [1.0] * 40}),
                 "tbbt": run_stub("tbbt", {f"e{i}": [0.7] for i in range(6)})}
    result = compare_variant(candidate, reference, "recall@10")
    assert result["per_dataset"]["vatex"]["method"] == "bootstrap"
    assert result["per_dataset"]["tbbt"]["method"] == "t"
    assert result["pooled_datasets"] == ["tbbt"]
    assert result["pooled"]["n"] == 6          # only the episodes of the serial


def test_bootstrap_interval_brackets_the_mean():
    ci = bootstrap_interval([1.0] * 30 + [0.0] * 30)
    assert math.isclose(ci["mean"], 0.5)
    assert ci["low"] < 0.5 < ci["high"]
    assert ci["method"] == "bootstrap"


# ------------------------------------------------------------- verdict scope

def test_a_verdict_from_one_serial_is_marked_provisional():
    reference = {"office": run_stub("office", {f"e{i}": [0.5] for i in range(6)})}
    better = {"office": run_stub("office", {f"e{i}": [0.7] for i in range(6)})}
    verdict = decide({"B": compare_variant(better, reference, "recall@10")}, ["A", "B"])
    assert verdict["series"] == 1 and verdict["provisional"]


def test_a_verdict_from_both_serials_is_not_provisional():
    reference = {"tbbt": run_stub("tbbt", {f"e{i}": [0.5] for i in range(6)}),
                 "office": run_stub("office", {f"e{i}": [0.5] for i in range(6)})}
    better = {"tbbt": run_stub("tbbt", {f"e{i}": [0.7] for i in range(6)}),
              "office": run_stub("office", {f"e{i}": [0.7] for i in range(6)})}
    verdict = decide({"B": compare_variant(better, reference, "recall@10")}, ["A", "B"])
    assert verdict["series"] == 2 and not verdict["provisional"]
