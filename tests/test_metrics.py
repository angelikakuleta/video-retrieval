"""Ranking metrics: the event as the unit of evaluation."""

import numpy as np

from src.evaluation.metrics import (
    PER_QUERY_KEYS,
    evaluate_matrix,
    query_metrics,
    summarize,
)

FRAGMENTS = [f"s01e01_{i:03d}" for i in range(1, 21)]


def test_recall_is_binary_no_matter_how_many_fragments_the_event_covers():
    # the event covers three fragments; one of them is 2nd in the ranking
    metrics = query_metrics(FRAGMENTS, {"s01e01_002", "s01e01_015", "s01e01_018"})
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0


def test_the_fragment_level_reading_is_kept_alongside():
    metrics = query_metrics(FRAGMENTS, {"s01e01_002", "s01e01_015", "s01e01_018"})
    assert metrics["recall_full@5"] == 1 / 3
    assert metrics["recall_full@10"] == 1 / 3


def test_a_single_correct_fragment_makes_both_readings_agree():
    metrics = query_metrics(FRAGMENTS, {"s01e01_004"})
    for k in (1, 5, 10):
        assert metrics[f"recall@{k}"] == metrics[f"recall_full@{k}"]


def test_a_query_with_nothing_found_scores_zero_everywhere():
    metrics = query_metrics(FRAGMENTS[:3], {"s01e01_019"})
    assert metrics["recall@10"] == 0.0
    assert metrics["rr"] == 0.0 and metrics["ap"] == 0.0
    assert metrics["rank"] is None


def test_event_level_average_precision_is_the_reciprocal_rank():
    metrics = query_metrics(FRAGMENTS, {"s01e01_003", "s01e01_009"})
    assert metrics["ap"] == metrics["rr"] == 1 / 3
    # the fragment-level variant does look at both hits, so it differs
    assert metrics["ap_full"] != metrics["ap"]


def test_summary_reports_map_equal_to_mrr():
    per_query = [query_metrics(FRAGMENTS, {"s01e01_002", "s01e01_015"}),
                 query_metrics(FRAGMENTS, {"s01e01_007"})]
    summary = summarize(per_query)
    assert summary["mAP"] == summary["MRR"]
    assert summary["mAP_full"] != summary["mAP"]
    assert summary["query_count"] == 2
    assert summary["MedR"] == 4.5 and summary["MnR"] == 4.5


def test_per_query_difference_of_two_configurations_is_minus_one_zero_or_one():
    a = query_metrics(FRAGMENTS, {"s01e01_002", "s01e01_015"})
    b = query_metrics(FRAGMENTS[::-1], {"s01e01_002", "s01e01_015"})
    assert a["recall@10"] - b["recall@10"] in (-1.0, 0.0, 1.0)


def test_every_declared_key_is_produced():
    metrics = query_metrics(FRAGMENTS, {"s01e01_002"})
    assert set(PER_QUERY_KEYS) <= set(metrics)


def test_matrix_mode_matches_the_list_mode():
    scores = np.zeros((1, len(FRAGMENTS)), dtype=np.float32)
    scores[0] = np.arange(len(FRAGMENTS))[::-1]      # first fragment scores highest
    relevant = {"s01e01_002", "s01e01_015"}
    from_matrix = evaluate_matrix(scores, FRAGMENTS, [relevant])
    from_list = summarize([query_metrics(FRAGMENTS, relevant)])
    assert from_matrix["recall@10"] == from_list["recall@10"] == 1.0
    assert from_matrix["mAP"] == from_list["mAP"]


def test_a_query_with_no_correct_fragment_in_the_collection_is_skipped():
    scores = np.zeros((2, len(FRAGMENTS)), dtype=np.float32)
    summary = evaluate_matrix(scores, FRAGMENTS, [{"s01e01_002"}, {"elsewhere_001"}])
    assert summary["query_count"] == 1


def test_empty_summary_reports_no_queries():
    assert summarize([]) == {"query_count": 0}
