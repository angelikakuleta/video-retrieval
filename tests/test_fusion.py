"""Standardization, the neutral value and the per-query weighting."""

import numpy as np
import pytest

from src.retrieval.fusion import EPSILON, combine, query_weights, standardize


def test_zscore_centres_and_scales_each_query_separately():
    values = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    z = standardize(values)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-6)
    assert np.allclose(z[0], z[1])       # scale does not survive standardization


def test_ordering_inside_one_signal_is_preserved():
    z = standardize(np.array([[0.31, 0.28, 0.35, 0.30]]))
    assert list(np.argsort(-z[0])) == [2, 0, 3, 1]


def test_a_signal_below_epsilon_is_dropped_instead_of_amplified():
    barely = np.array([[0.5, 0.5 + 1e-9, 0.5 - 1e-9, 0.5]])
    assert np.all(standardize(barely) == 0.0)


def test_a_deviation_above_epsilon_still_standardizes():
    values = np.array([[0.0, 4 * EPSILON, 8 * EPSILON, 12 * EPSILON]])
    assert np.any(standardize(values) != 0.0)


def test_a_constant_signal_is_zero_everywhere():
    assert np.all(standardize(np.full((2, 5), 0.7)) == 0.0)


def test_a_missing_value_becomes_neutral_not_minimal():
    values = np.array([[1.0, 2.0, np.nan, 3.0]])
    z = standardize(values)
    assert z[0, 2] == 0.0                       # the collection mean
    assert z[0, 0] < 0.0 < z[0, 3]              # and it sits between the others


def test_missing_values_do_not_enter_the_mean_or_the_deviation():
    with_gap = standardize(np.array([[1.0, 2.0, 3.0, np.nan]]))
    without = standardize(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(with_gap[0, :3], without[0])


def test_a_signal_with_no_data_at_all_is_zero_everywhere():
    assert np.all(standardize(np.full((1, 4), np.nan)) == 0.0)


def test_a_matrix_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        standardize(np.array([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------- weighting

def test_uniform_weights_are_one_over_the_active_count():
    active = np.array([[True, True], [True, False], [False, False]])
    weights = query_weights(active)
    assert np.allclose(weights[:, 0], [0.5, 0.5, 0.0])   # two active
    assert np.allclose(weights[:, 1], [1.0, 0.0, 0.0])   # one active


def test_weights_sum_to_one_per_query():
    active = np.array([[True, True, True], [True, True, False], [False, True, False]])
    weights = query_weights(active)
    assert np.allclose(weights.sum(axis=0), 1.0)


def test_manual_weights_are_renormalized_when_a_signal_is_inactive():
    active = np.array([[True], [False]])
    weights = query_weights(active, base=np.array([0.7, 0.3]))
    assert np.allclose(weights[:, 0], [1.0, 0.0])


def test_manual_weights_keep_their_proportions_when_all_are_active():
    active = np.array([[True], [True]])
    weights = query_weights(active, base=np.array([0.75, 0.25]))
    assert np.allclose(weights[:, 0], [0.75, 0.25])


def test_a_query_with_no_active_signal_gets_zero_weights():
    weights = query_weights(np.array([[False], [False]]))
    assert np.all(weights == 0.0)


# -------------------------------------------------------------------- fusion

def test_combine_is_the_mean_of_the_standardized_signals():
    a = np.array([[1.0, -1.0]], dtype=np.float32)
    b = np.array([[3.0, 1.0]], dtype=np.float32)
    scores = combine([a, b], np.array([[True], [True]]))
    assert np.allclose(scores, [[2.0, 0.0]])


def test_an_inactive_signal_does_not_enter_the_score():
    a = np.array([[1.0, -1.0]], dtype=np.float32)
    b = np.array([[9.0, 9.0]], dtype=np.float32)
    scores = combine([a, b], np.array([[True], [False]]))
    assert np.allclose(scores, a)


def test_combine_needs_at_least_one_signal():
    with pytest.raises(ValueError):
        combine([], np.zeros((0, 1), dtype=bool))
