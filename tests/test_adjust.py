"""Rule W-22: how an imported annotation is fitted to the masks of its episode."""

import pytest

from src.annotation.adjust import (FLAG_SHIFTED, FLAG_SHIFT_FAILED, FLAG_SPANS_MASK,
                                   FLAG_TRIMMED, fit, fit_episode)

RANGES = [(0.0, 1200.0)]
#: a transition at 100 s, a structural mask (credits) from 1100 s
MASKS = [(100.0, 101.35), (1100.0, 1200.0)]


def test_untouched_when_it_meets_no_mask():
    assert fit(200.0, 204.0, MASKS, RANGES) == (200.0, 204.0, [], 0.0)


def test_start_inside_a_mask_is_shifted_and_keeps_its_length():
    # the measured shape of tbbt_s03e02_d88919: most of it inside the animation
    start, end, flags, offset = fit(100.42, 101.85, MASKS, RANGES)
    assert (start, end) == (101.35, 102.78)
    assert round(end - start, 2) == 1.43        # length preserved, not cut to 0.5 s
    assert flags == [FLAG_SHIFTED]
    assert offset == 0.93


def test_end_inside_a_mask_is_trimmed_however_much_it_costs():
    start, end, flags, _ = fit(1090.0, 1150.0, MASKS, RANGES)
    assert (start, end) == (1090.0, 1100.0)
    assert flags == [FLAG_TRIMMED]


def test_entirely_inside_a_mask_cannot_be_rescued():
    assert fit(100.5, 101.0, MASKS, RANGES) is None


def test_a_mask_inside_the_annotation_is_only_flagged():
    start, end, flags, _ = fit(99.0, 103.0, MASKS, RANGES)
    assert (start, end) == (99.0, 103.0)        # times untouched
    assert flags == [FLAG_SPANS_MASK]


def test_shift_with_nowhere_to_land_keeps_the_original_times():
    # two masks 1 s apart; a 4 s annotation starting in the first one cannot fit
    masks = [(100.0, 101.0), (102.0, 103.0)]
    start, end, flags, offset = fit(100.5, 104.5, masks, RANGES)
    assert (start, end) == (100.5, 104.5)
    assert flags == [FLAG_SHIFT_FAILED] and offset == 0.0


def test_shift_is_refused_when_it_would_leave_the_corpus():
    masks = [(1090.0, 1100.0)]
    ranges = [(0.0, 1100.0)]
    _, _, flags, _ = fit(1095.0, 1105.0, masks, ranges)
    assert flags == [FLAG_SHIFT_FAILED]


# --------------------------------------------------------------- propagation

def items():
    return [("a", 100.42, 101.85), ("b", 200.0, 204.0), ("c", 300.0, 303.0)]


def test_propagation_off_leaves_the_others_alone():
    out = fit_episode(items(), MASKS, RANGES, "off")
    assert out["a"][:2] == (101.35, 102.78)
    assert out["b"][:2] == (200.0, 204.0)
    assert out["c"][:2] == (300.0, 303.0)


def test_propagation_forward_moves_every_later_annotation():
    out = fit_episode(items(), MASKS, RANGES, "forward")
    assert out["b"][:2] == (200.93, 204.93)
    assert out["c"][:2] == (300.93, 303.93)


def test_propagation_does_nothing_when_no_shift_was_measured():
    plain = [("b", 200.0, 204.0), ("c", 300.0, 303.0)]
    for mode in ("off", "forward"):
        out = fit_episode(plain, MASKS, RANGES, mode)
        assert out["b"][:2] == (200.0, 204.0) and out["c"][:2] == (300.0, 303.0)


def test_only_two_propagation_modes_remain():
    """episode_median is gone: the choice is made per episode, by hand."""
    from src.annotation.adjust import PROPAGATION_MODES

    assert PROPAGATION_MODES == ("off", "forward")
    with pytest.raises(ValueError):
        fit_episode(items(), MASKS, RANGES, "episode_median")
