"""Binding the items of the frame grid to the fragments of a collection."""

import numpy as np

from src.runners.components import fragment_of_times
from src.segmentation.timeline import Timeline


def segments(episode="s01e01", count=3, length=10.0):
    return [{"episode": episode, "segment_id": i + 1,
             "start": i * length, "end": (i + 1) * length} for i in range(count)]


def row_index(rows):
    return {f"{r['episode']}_{r['segment_id']:03d}": i for i, r in enumerate(rows)}


def test_a_moment_lands_in_the_fragment_that_contains_it():
    rows = segments()
    times = np.array([0.0, 5.0, 12.5, 25.0])
    assigned = fragment_of_times(rows, "s01e01", times, row_index(rows), None)
    assert list(assigned) == [0, 0, 1, 2]


def test_a_moment_outside_every_fragment_belongs_to_none():
    rows = segments(count=2)
    times = np.array([-1.0, 5.0, 45.0])
    assigned = fragment_of_times(rows, "s01e01", times, row_index(rows), None)
    assert list(assigned) == [-1, 0, -1]


def test_fragments_are_half_open_so_a_boundary_belongs_to_the_later_one():
    rows = segments()
    assigned = fragment_of_times(rows, "s01e01", np.array([10.0]), row_index(rows), None)
    assert assigned[0] == 1


def test_an_episode_with_no_fragments_yields_nothing():
    rows = segments()
    assigned = fragment_of_times(rows, "s09e99", np.array([1.0, 2.0]),
                                 row_index(rows), None)
    assert list(assigned) == [-1, -1]


def test_the_row_of_the_collection_is_used_not_the_order_of_the_segments():
    rows = segments()
    shuffled = {"s01e01_001": 2, "s01e01_002": 0, "s01e01_003": 1}
    assigned = fragment_of_times(rows, "s01e01", np.array([5.0, 15.0, 25.0]),
                                 shuffled, None)
    assert list(assigned) == [2, 0, 1]


def test_a_fragment_missing_from_the_collection_takes_no_items():
    rows = segments()
    partial = {"s01e01_001": 0, "s01e01_003": 1}      # the middle one is absent
    assigned = fragment_of_times(rows, "s01e01", np.array([5.0, 15.0, 25.0]),
                                 partial, None)
    assert list(assigned) == [0, -1, 1]


def test_a_masked_moment_reaches_no_fragment():
    rows = segments()
    # the corpus is the whole 30 s, with 12-14 s cut out as a transition
    timeline = Timeline([(0.0, 30.0)], [(12.0, 14.0)], [])
    times = np.array([5.0, 13.0, 25.0])
    assigned = fragment_of_times(rows, "s01e01", times, row_index(rows), timeline)
    assert list(assigned) == [0, -1, 2]


def test_a_black_moment_reaches_no_fragment_either():
    rows = segments()
    timeline = Timeline([(0.0, 30.0)], [], [(20.0, 22.0)])
    times = np.array([5.0, 21.0, 25.0])
    assigned = fragment_of_times(rows, "s01e01", times, row_index(rows), timeline)
    assert list(assigned) == [0, -1, 2]


def test_no_times_means_no_assignments():
    rows = segments()
    assigned = fragment_of_times(rows, "s01e01", np.array([]), row_index(rows), None)
    assert len(assigned) == 0
