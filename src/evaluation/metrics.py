"""Ranking metrics, computed per event (chapter 4).

One query describes one event, so Recall@K is binary: 1 when at least one
correct fragment is among the first K. Average precision reduces the same way,
which makes the event-level mAP equal to MRR.

The fragment-level readings are kept alongside as auxiliary measures --
``recall_full@k`` and ``ap_full``. They differ from the event-level ones exactly
when an event spans more than one fragment.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, median

import numpy as np

KS = (1, 5, 10)

#: keys of one query's metrics, in reporting order
PER_QUERY_KEYS = tuple(
    [f"recall@{k}" for k in KS] + [f"recall_full@{k}" for k in KS]
    + ["ap", "ap_full", "rr", "rank"])


def _metrics_from_ranks(ranks: Sequence[int], n_relevant: int,
                        ks: Sequence[int] = KS) -> dict:
    """Metrics for one query from the ranks of its correct fragments.

    ``n_relevant`` is how many the collection holds -- the denominator of the
    fragment-level measures.
    """
    ranks = sorted(ranks)
    first = ranks[0] if ranks else None
    metrics: dict[str, float | None] = {}
    for k in ks:
        found = sum(r <= k for r in ranks)
        metrics[f"recall@{k}"] = 1.0 if found else 0.0
        metrics[f"recall_full@{k}"] = found / n_relevant if n_relevant else 0.0
    metrics["rr"] = 1.0 / first if first else 0.0
    # event-level average precision: one event per query, so AP is the
    # reciprocal rank of the first correct fragment
    metrics["ap"] = metrics["rr"]
    metrics["ap_full"] = (sum((j + 1) / r for j, r in enumerate(ranks)) / n_relevant
                          if ranks and n_relevant else 0.0)
    metrics["rank"] = first
    return metrics


def query_metrics(ranking: Sequence[str], relevant: set[str],
                  ks: Sequence[int] = KS) -> dict:
    """Metrics for a single query given the ranked list of identifiers."""
    ranks = [i + 1 for i, vid in enumerate(ranking) if vid in relevant]
    return _metrics_from_ranks(ranks, len(relevant), ks)


def _ranks_from_matrix(row: np.ndarray, relevant_cols: Sequence[int]) -> list[int]:
    """Ranks without a full sort: how many fragments score higher, plus one."""
    relevant_scores = row[list(relevant_cols)]
    return [int((row > s).sum()) + 1 for s in relevant_scores]


def evaluate_matrix(
    scores: np.ndarray,
    vids: Sequence[str],
    relevant: Sequence[set[str]],
    ks: Sequence[int] = KS,
) -> dict:
    """Averaged metrics for a query set, from a (n_queries, n_fragments) score matrix.

    A query with no correct fragment in the collection is skipped.
    """
    column = {vid: i for i, vid in enumerate(vids)}
    per_query = []
    for row, rel in zip(scores, relevant):
        relevant_cols = [column[v] for v in rel if v in column]
        if not relevant_cols:
            continue
        ranks = _ranks_from_matrix(row, relevant_cols)
        per_query.append(_metrics_from_ranks(ranks, len(relevant_cols), ks))
    return summarize(per_query, ks)


def recall_at(per_query: Sequence[dict], k: int) -> float | None:
    """Recall@k of a query set for ANY k, read from the stored ``rank``.

    ``KS`` fixes which cut-offs a run averages and writes down, and it stays as
    it is -- the thesis reports those three. But a rank is a rank: E4-D needs
    Recall@50 of the baseline as its ceiling, and re-running a collection to
    learn it would be absurd when every record already says where its first
    correct fragment landed.

    ``None`` for an empty query set, like the other averages here: no query is
    not the same as no hit.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not per_query:
        return None
    hits = sum(1 for m in per_query
               if m.get("rank") is not None and m["rank"] <= k)
    return hits / len(per_query)


def summarize(per_query: Sequence[dict], ks: Sequence[int] = KS) -> dict:
    """Averages the per-query metrics.

    ``mAP`` is event-level and therefore equals ``MRR``; ``mAP_full`` is its
    fragment-level counterpart. ``MedR``/``MnR``: median and mean rank of the first
    hit, over the queries that had one.
    """
    if not per_query:
        return {"query_count": 0}
    first_ranks = [m["rank"] for m in per_query if m["rank"] is not None]
    summary: dict[str, float | None] = {"query_count": len(per_query)}
    for k in ks:
        summary[f"recall@{k}"] = mean(m[f"recall@{k}"] for m in per_query)
    for k in ks:
        summary[f"recall_full@{k}"] = mean(m[f"recall_full@{k}"] for m in per_query)
    summary["mAP"] = mean(m["ap"] for m in per_query)
    summary["mAP_full"] = mean(m["ap_full"] for m in per_query)
    summary["MRR"] = mean(m["rr"] for m in per_query)
    summary["MedR"] = median(first_ranks) if first_ranks else None
    summary["MnR"] = mean(first_ranks) if first_ranks else None
    return summary
