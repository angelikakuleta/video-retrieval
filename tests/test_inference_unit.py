"""The unit a DIFFERENCE is read at: the episode for a series, the query never.

Chapter 4 fixes two readings that are easy to confuse and impossible to tell
apart on balanced data: a LEVEL of Recall is a mean over queries, a DIFFERENCE
between two configurations is the mean of the per-episode differences. Every
example here is deliberately UNBALANCED -- one episode carries a single query and
another carries three -- because that is the only shape in which the two readings
disagree, and a test built on balanced episodes would pass either way.
"""

import numpy as np
import pytest

from src.evaluation.compare import (
    effect_sizes,
    inference_unit,
    mean_of_differences,
    paired_differences,
)
from src.evaluation.sensitivity import marginal_contributions

#: one query in the first episode, three in the second
UNITS = ["s01e01", "s01e02", "s01e02", "s01e02"]
#: only the query of the first episode improves
DIFFERENCES = [1.0, 0.0, 0.0, 0.0]
#: 1/4 if the query is the unit, (1 + 0)/2 if the episode is
BY_QUERY, BY_EPISODE = 0.25, 0.5


def run_stub(values_by_episode, metric="recall@10"):
    per_query = [{"desc_id": i, "episode": episode, metric: value}
                 for i, (episode, value) in enumerate(
                     (episode, value)
                     for episode, values in values_by_episode.items()
                     for value in values)]
    return {"per_query": per_query}


# ------------------------------------------------------------- the primitive
def test_mean_of_differences_reads_the_unit_it_is_given():
    assert mean_of_differences(DIFFERENCES) == BY_QUERY
    assert mean_of_differences(DIFFERENCES, UNITS) == BY_EPISODE


def test_mean_of_differences_refuses_a_partial_unit_list():
    with pytest.raises(ValueError, match="unit of every difference"):
        mean_of_differences(DIFFERENCES, UNITS[:2])


def test_mean_of_differences_empty_is_none():
    assert mean_of_differences([]) is None
    assert mean_of_differences([], []) is None


def test_mean_of_differences_agrees_with_paired_differences():
    """The two roads to the same quantity have to meet.

    ``paired_differences`` groups the per_query records itself; this one is
    handed differences that were computed some other way. A drift between them
    would put the sensitivity analysis and the contribution tables on two
    different definitions of the same number.
    """
    candidate = run_stub({"s01e01": [1.0], "s01e02": [1.0, 0.0, 1.0]})
    reference = run_stub({"s01e01": [0.0], "s01e02": [1.0, 0.0, 1.0]})
    by_hand = mean_of_differences(DIFFERENCES, UNITS)
    episode = paired_differences(candidate, reference, "recall@10", None, "episode")
    assert sum(episode) / len(episode) == pytest.approx(by_hand)

    clip = paired_differences(candidate, reference, "recall@10", None, "clip")
    assert sum(clip) / len(clip) == pytest.approx(mean_of_differences(DIFFERENCES))


# ---------------------------------------------------------- the weight sweep
def marginal_example():
    """Four queries, two signals, two fragments; ``f0`` is the correct one.

    Dropping the caption signal flips the single query of the first episode from
    a hit to a miss and leaves the three queries of the second alone -- so the
    contribution is 1/4 per query and 1/2 per episode.
    """
    names = ["scene_embedding", "caption"]
    fragments = ["f0", "f1"]
    scene = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    caption = np.array([[4.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    values = np.stack([scene, caption])
    active = np.ones((2, 4), dtype=bool)
    relevant = [{"f0"}] * 4
    return values, active, fragments, relevant, names


def test_marginal_contributions_read_the_unit():
    values, active, fragments, relevant, names = marginal_example()
    by_query = marginal_contributions(values, active, None, fragments, relevant,
                                      names, k=1)
    by_episode = marginal_contributions(values, active, None, fragments, relevant,
                                        names, k=1, units=UNITS)
    assert by_query["caption"] == pytest.approx(BY_QUERY)
    assert by_episode["caption"] == pytest.approx(BY_EPISODE)
    assert "scene_embedding" not in by_query      # the base has no contribution


# ---------------------------------------------------- the secondary measures
def test_effect_sizes_delta_is_not_the_difference_of_the_levels():
    """For a series the delta is the per-episode mean, the levels are per query.

    So ``delta`` is NOT ``value - reference`` there, and a table printing all
    three has to say which is which.
    """
    candidate = run_stub({"s01e01": [1.0], "s01e02": [1.0, 0.0, 1.0]})
    reference = run_stub({"s01e01": [0.0], "s01e02": [1.0, 0.0, 1.0]})
    entry = effect_sizes({"tbbt": candidate}, {"tbbt": reference},
                         metrics=("recall@10",))["tbbt"]["recall@10"]
    assert entry["unit"] == "episode"
    assert entry["value"] - entry["reference"] == pytest.approx(BY_QUERY)
    assert entry["delta"] == pytest.approx(BY_EPISODE)


def test_effect_sizes_delta_matches_the_levels_for_clips():
    """VATEX has no episodes, so there the two readings are one number."""
    assert inference_unit("vatex") == "clip"
    candidate = run_stub({"c1": [1.0], "c2": [1.0, 0.0, 1.0]})
    reference = run_stub({"c1": [0.0], "c2": [1.0, 0.0, 1.0]})
    entry = effect_sizes({"vatex": candidate}, {"vatex": reference},
                         metrics=("recall@10",))["vatex"]["recall@10"]
    assert entry["unit"] == "clip"
    assert entry["delta"] == pytest.approx(entry["value"] - entry["reference"])
    assert entry["delta"] == pytest.approx(BY_QUERY)
