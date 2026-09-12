"""Length correction and the segmentation strategies, on the content axis."""

from src.segmentation.segments import correct_lengths, fixed_windows, from_boundaries, rows
from src.segmentation.timeline import Timeline


def lengths(spans, timeline):
    return [round(timeline.content_length(a, b), 3) for a, b in spans]


def plain(*ranges):
    """A timeline without transition masks -- content time equals file time."""
    return Timeline.from_ranges(list(ranges))


# ---------------------------------------- behaviour without transition masks
# These are the regression tests: with no masks the content axis must reproduce
# the results of the implementation that predated it.

def test_merge_short_with_shorter_neighbour():
    # 2 s span sits between a 4 s and a 10 s span -> merges into the 4 s one
    tl = plain((0, 16))
    out = correct_lengths([(0, 4), (4, 6), (6, 16)], tl, min_len=3, max_len=15)
    assert lengths(out, tl) == [6.0, 10.0]


def test_single_short_span_is_kept():
    tl = plain((0, 2))
    assert correct_lengths([(0, 2)], tl, min_len=3, max_len=15) == [(0.0, 2.0)]


def test_split_long_into_equal_parts():
    tl = plain((0, 31))
    out = correct_lengths([(0, 31)], tl, min_len=3, max_len=15)
    assert len(out) == 3
    assert all(abs(l - 31 / 3) < 1e-3 for l in lengths(out, tl))
    assert out[0][0] == 0.0 and out[-1][1] == 31.0


def test_fixed_windows_restart_per_range_and_merge_tail():
    # range 1: 25 s -> 10 + 10 + 5; range 2: 12 s -> 10 + 2 -> merged into 12
    tl = plain((0, 25), (100, 112))
    out = fixed_windows(tl, window=10, min_len=3, max_len=15)
    assert lengths(out, tl) == [10.0, 10.0, 5.0, 12.0]
    assert out[3] == (100.0, 112.0)


def test_boundaries_are_cut_to_ranges():
    # a boundary inside a structural mask (50) must not create a fragment
    tl = plain((0, 20), (60, 70))
    out = from_boundaries([10.0, 50.0], tl, min_len=3, max_len=15)
    assert (0.0, 10.0) in out and (10.0, 20.0) in out
    assert all(not (20 < a < 60) for a, _ in out)


# ------------------------------------------- behaviour with transition masks

def test_every_fixed_window_holds_exactly_ten_seconds_of_content():
    tl = Timeline([(0.0, 100.0)], [(35.0, 36.5)])
    out = fixed_windows(tl, window=10, min_len=3, max_len=15)
    assert lengths(out, tl)[:-1] == [10.0] * (len(out) - 1)
    # the window holding the animation is longer in the file than in content
    wide = [(a, b) for a, b in out if b - a > 10.0 + 1e-6]
    assert wide == [(30.0, 41.5)]
    assert tl.content_length(30.0, 41.5) == 10.0


def test_a_shot_shrunk_by_a_mask_is_merged_away():
    # 4 s of file holding a 1.5 s animation is 2.5 s of content -> below MIN_LEN
    tl = Timeline([(0.0, 100.0)], [(10.0, 11.5)])
    out = from_boundaries([8.0, 12.0], tl, min_len=3, max_len=15)
    assert out[0] == (0.0, 12.0)
    assert tl.content_length(*out[0]) == 10.5
    assert all(tl.content_length(a, b) >= 3.0 for a, b in out)


def test_boundary_inside_a_mask_moves_behind_it():
    tl = Timeline([(0.0, 100.0)], [(10.0, 11.5)])
    out = from_boundaries([10.7], tl, min_len=3, max_len=15)
    assert out[0] == (0.0, 11.5)
    assert all(not tl.in_hole(a) for a, _ in out)


def test_black_stretches_change_no_fragment():
    # the entire difference between a skip and a transition mask
    holes = [(35.0, 36.5)]
    with_black = Timeline([(0.0, 100.0)], holes, [(70.0, 72.0)])
    without = Timeline([(0.0, 100.0)], holes)
    assert (fixed_windows(with_black, window=10, min_len=3, max_len=15)
            == fixed_windows(without, window=10, min_len=3, max_len=15))


def test_rows_report_both_durations():
    tl = Timeline([(0.0, 100.0)], [(35.0, 36.5)])
    out = rows("s03e02", "dev", "tbbt_s03e02.mp4", [(30.0, 41.5), (0.0, 10.0)], tl)
    assert out[0]["segment_id"] == 1 and out[0]["start"] == 0.0
    wide = out[1]
    assert wide["duration"] == 10.0 and wide["file_duration"] == 11.5
