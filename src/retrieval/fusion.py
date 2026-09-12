"""Standardization and weighted fusion of the component signals (chapter 4).

Three rules live here and nowhere else:

* every signal is standardized per query over the whole collection (Z-score),
* a signal whose deviation falls below EPSILON separates nothing for that query
  and is zeroed instead of amplified,
* a fragment with no value (no face, no object, no caption) gets the collection
  mean, which after standardization is 0 -- the neutral value, not the minimum.

A signal inactive for a whole query is left out of that query's weighting and
the uniform weights are shared among the rest.
"""

from __future__ import annotations

import numpy as np

#: below this standard deviation a signal is treated as non-discriminating
EPSILON = 1e-6


def standardize(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Z-score of a (n_queries, n_fragments) matrix, row by row.

    NaN marks a fragment the component had no value for: excluded from the mean and
    the deviation, returned as 0. A row below ``epsilon``, or with no values at all,
    comes back all zeros.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a (n_queries, n_fragments) matrix, got {values.shape}")

    known = np.isfinite(values)
    counts = known.sum(axis=1, keepdims=True)
    present = np.where(known, values, 0.0)
    mean = np.divide(present.sum(axis=1, keepdims=True), counts,
                     out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    centered = np.where(known, values - mean, 0.0)
    variance = np.divide((centered ** 2).sum(axis=1, keepdims=True), counts,
                         out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    deviation = np.sqrt(variance)

    usable = (counts > 0) & (deviation >= epsilon)
    scale = np.where(usable, deviation, 1.0)
    return np.where(usable & known, centered / scale, 0.0).astype(np.float32)


def query_weights(active: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """Weight of every signal for every query, summing to one per query.

    ``active`` is (n_signals, n_queries); ``base`` are the configured per-signal
    weights (None = uniform). An inactive signal gets zero and its share goes to the
    active ones, so the uniform case is exactly 1/|A(q)|.
    """
    active = np.asarray(active, dtype=bool)
    if active.ndim != 2:
        raise ValueError(f"expected a (n_signals, n_queries) matrix, got {active.shape}")
    base = (np.ones(active.shape[0], dtype=np.float64) if base is None
            else np.asarray(base, dtype=np.float64))
    if base.shape != (active.shape[0],):
        raise ValueError("number of weights differs from the number of signals")

    weights = base[:, None] * active
    total = weights.sum(axis=0, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)


def combine(standardized: list[np.ndarray], active: np.ndarray,
            base: np.ndarray | None = None) -> np.ndarray:
    """Weighted sum of the standardized signals: (n_queries, n_fragments)."""
    if not standardized:
        raise ValueError("at least one signal is required")
    weights = query_weights(active, base)
    total = np.zeros(standardized[0].shape, dtype=np.float32)
    for matrix, row in zip(standardized, weights):
        total += (row[:, None] * matrix).astype(np.float32)
    return total
