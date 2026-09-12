"""The pipeline: per-query activity, weighting and the top-k ordering."""

import numpy as np
import pytest

from src.indexing.faiss_index import build_index
from src.retrieval.pipeline import Pipeline, as_queries
from src.utils.queries import Query


class StubSignal:
    """A signal reading its values from a table keyed by the query text."""

    needs_phrases = False

    def __init__(self, name, values, active_for=None):
        self.name = name
        self.values = np.asarray(values, dtype=np.float32)
        self.active_for = active_for

    def raw_values(self, queries, phrases=None):
        return np.stack([self.values[hash(q.desc) % len(self.values)]
                         if len(self.values) > 1 else self.values[0]
                         for q in queries])

    def active(self, query, phrases=None):
        return self.active_for is None or query.desc in self.active_for


def collection_of(n_fragments=4, dim=8):
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((n_fragments, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return build_index(matrix, [f"s01e01_{i:03d}" for i in range(1, n_fragments + 1)])


def query(text):
    return Query(desc_id=abs(hash(text)) % 10_000, desc=text,
                 vid_name="tbbt_s01e01", ts=(0.0, 5.0), source="tbbt")


def test_plain_texts_are_accepted_as_queries():
    records = as_queries(["a man cooks", "a dog runs"])
    assert [r.desc for r in records] == ["a man cooks", "a dog runs"]
    assert all(isinstance(r, Query) for r in records)


def test_a_ready_record_passes_through_untouched():
    record = query("a man cooks")
    assert as_queries([record])[0] is record


def test_signal_names_must_be_unique():
    collection = collection_of()
    with pytest.raises(ValueError):
        Pipeline(collection, [StubSignal("scene", [[1, 2, 3, 4]]),
                              StubSignal("scene", [[4, 3, 2, 1]])])


def test_manual_weights_must_name_the_active_signals():
    collection = collection_of()
    with pytest.raises(ValueError):
        Pipeline(collection, [StubSignal("scene", [[1, 2, 3, 4]])],
                 weights={"caption": 1.0})


def test_an_inactive_signal_leaves_the_ranking_to_the_others():
    collection = collection_of()
    scene = StubSignal("scene", [[4.0, 3.0, 2.0, 1.0]])
    # active only for a query that names a profiled character
    identity = StubSignal("identity", [[1.0, 2.0, 3.0, 40.0]],
                          active_for={"Sheldon eats"})

    both = Pipeline(collection, [scene, identity])
    scene_only = Pipeline(collection, [scene])

    named = both.scores([query("Sheldon eats")])
    unnamed = both.scores([query("a man eats")])
    # with nobody named the identity signal is left out entirely, and the score
    # is the scene signal alone rather than the scene halved
    assert np.allclose(unnamed, scene_only.scores([query("a man eats")]))
    assert not np.allclose(named, scene_only.scores([query("Sheldon eats")]))


def test_timings_are_collected_per_query_and_match_the_batched_path():
    collection = collection_of()
    signals = [StubSignal("scene", [[4.0, 3.0, 2.0, 1.0]]),
               StubSignal("caption", [[1.0, 1.0, 5.0, 1.0]])]
    queries = [query("a"), query("b"), query("c")]
    pipeline = Pipeline(collection, signals)

    batched, active_batched = pipeline.signal_matrices(queries)
    timings: list[float] = []
    looped, active_looped = pipeline.signal_matrices(queries, timings)

    assert len(timings) == len(queries)
    assert all(t >= 0 for t in timings)
    assert np.array_equal(active_batched, active_looped)
    for a, b in zip(batched, looped):
        assert np.allclose(a, b, atol=1e-6)


def test_top_returns_positions_in_descending_order():
    collection = collection_of()
    pipeline = Pipeline(collection, [StubSignal("scene", [[1.0, 9.0, 5.0, 3.0]])])
    results = pipeline.search(query("anything"), k=3)
    assert [r.vid_name for r in results] == ["s01e01_002", "s01e01_003", "s01e01_004"]
    assert [r.position for r in results] == [1, 2, 3]
    assert results[0].score > results[1].score > results[2].score


def test_k_larger_than_the_collection_is_clamped():
    collection = collection_of(n_fragments=3)
    pipeline = Pipeline(collection, [StubSignal("scene", [[1.0, 2.0, 3.0]])])
    assert len(pipeline.search(query("anything"), k=10)) == 3
