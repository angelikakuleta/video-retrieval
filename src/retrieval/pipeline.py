"""Combines the component signals into one ranking.

The score of a fragment is the weighted sum of the signals active for a given
query (:mod:`src.retrieval.fusion`). Which are active is decided per query, not
per configuration: a signal that cannot speak about a query is left out of its
weighting and the rest share the weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.indexing.faiss_index import Collection
from src.retrieval.fusion import combine, standardize
from src.retrieval.signals import Signal
from src.utils.queries import Query


@dataclass
class Result:
    """A ranking entry returned for a query."""

    vid_name: str
    score: float
    position: int


def as_queries(items: list) -> list[Query]:
    """Accepts ready query records or plain texts, returns query records.

    The plain-text form is what the VATEX implementation check uses.
    """
    return [item if isinstance(item, Query)
            else Query(desc_id=i, desc=str(item), vid_name="", ts=(0.0, 0.0),
                       source="")
            for i, item in enumerate(items)]


class Pipeline:
    """Combines the component signals into a ranking of the collection."""

    def __init__(
        self,
        collection: Collection,
        signals: list[Signal],
        weights: dict[str, float] | None = None,
    ) -> None:
        if not signals:
            raise ValueError("the pipeline requires at least one signal")
        names = [signal.name for signal in signals]
        if len(set(names)) != len(names):
            raise ValueError(f"signal names must be unique, got {names}")
        if weights is not None and set(weights) != set(names):
            raise ValueError("weights must name exactly the active signals: "
                             f"{sorted(names)}")
        self.collection = collection
        self.signals = signals
        self.weights = weights
        self.base = (None if weights is None
                     else np.array([weights[name] for name in names], dtype=np.float64))

    def signal_matrices(self, queries: list,
                        timings: list[float] | None = None
                        ) -> tuple[list[np.ndarray], np.ndarray]:
        """Standardized values and the activity mask of every signal.

        ``(matrices, active)`` with ``matrices[j]`` of shape (n_queries, n_fragments) and
        ``active`` of (n_signals, n_queries). Given a ``timings`` list the queries are
        processed one at a time and each handling time is appended, in milliseconds --
        the cost reported per query. Standardization is per query either way, so the two
        paths differ only in how the encoders are batched.
        """
        records = as_queries(queries)
        extract = self._phrase_extractor()
        if timings is None:
            found = [extract(record) for record in records] if extract else None
            matrices = [standardize(signal.raw_values(records, found))
                        for signal in self.signals]
            active = [[signal.active(record, found[i] if found else None)
                       for i, record in enumerate(records)]
                      for signal in self.signals]
            return matrices, np.array(active, dtype=bool)

        import time

        shape = (len(records), self.collection.size)
        matrices = [np.zeros(shape, dtype=np.float32) for _ in self.signals]
        active = np.zeros((len(self.signals), len(records)), dtype=bool)
        if records:                       # warm-up: the first call of a model
            warm = [extract(records[0])] if extract else None
            for signal in self.signals:   # loads kernels the rest reuses
                signal.raw_values(records[:1], warm)
        for i, record in enumerate(records):
            started = time.perf_counter()
            # inside the timing: reading the phrases out of the query is part of
            # handling it, and the closed-vocabulary signals cannot start without
            one = [extract(record)] if extract else None
            for j, signal in enumerate(self.signals):
                matrices[j][i] = standardize(signal.raw_values([record], one))[0]
                active[j, i] = signal.active(record, one[0] if one else None)
            timings.append((time.perf_counter() - started) * 1000.0)
        return matrices, active

    def _phrase_extractor(self):
        """The phrase step of a query, or ``None`` when no signal reads phrases.

        Extracted ONCE per query and handed to every signal: it saves two spaCy
        parses, gives one object to log, and makes it structural that the two
        face mechanisms see the same list of expression phrases.
        """
        if not any(getattr(signal, "needs_phrases", False) for signal in self.signals):
            return None
        from src.retrieval.phrases import phrases_for

        return phrases_for

    def scores(self, queries: list) -> np.ndarray:
        """Combined score: a matrix (n_queries, n_fragments)."""
        matrices, active = self.signal_matrices(queries)
        return combine(matrices, active, self.base)

    def ranking(self, queries: list, k: int = 10) -> list[list[Result]]:
        """For every query, the ranking of the top ``k`` fragments."""
        scores = self.scores(queries)
        return [self.top(row, k) for row in scores]

    def top(self, scores: np.ndarray, k: int = 10) -> list[Result]:
        """Top ``k`` entries of one already computed score row."""
        k = min(k, self.collection.size)
        # partial sort of the top k, then an exact ordering
        indices = np.argpartition(-scores, k - 1)[:k]
        order = indices[np.argsort(-scores[indices])]
        return [Result(self.collection.vids[j], float(scores[j]), position + 1)
                for position, j in enumerate(order)]

    def search(self, query, k: int = 10) -> list[Result]:
        """Ranking of the top ``k`` fragments for a single query."""
        return self.ranking([query], k)[0]
