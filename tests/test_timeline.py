"""The content time axis: the two clocks, fragment pieces and frame sampling."""

from src.segmentation.timeline import Timeline, subtract, total

#: two corpus ranges, two transition masks in the first one, one black stretch
RANGES = [(0.0, 100.0), (200.0, 300.0)]
HOLES = [(50.0, 51.5), (80.0, 81.5)]
SKIPS = [(30.0, 32.0)]


def timeline() -> Timeline:
    return Timeline(RANGES, HOLES, SKIPS)


# ---------------------------------------------------------------- arithmetic

def test_subtract_cuts_every_shape_of_hole():
    spans = [(0.0, 10.0)]
    assert subtract(spans, [(4.0, 6.0)]) == [(0.0, 4.0), (6.0, 10.0)]   # inside
    assert subtract(spans, [(0.0, 3.0)]) == [(3.0, 10.0)]               # leading
    assert subtract(spans, [(8.0, 10.0)]) == [(0.0, 8.0)]               # trailing
    assert subtract(spans, [(0.0, 10.0)]) == []                         # whole
    assert subtract(spans, [(20.0, 30.0)]) == [(0.0, 10.0)]             # disjoint


# ------------------------------------------------------------ the two clocks

def test_content_time_skips_the_holes():
    tl = timeline()
    assert tl.to_content(0.0) == 0.0
    assert tl.to_content(49.0) == 49.0
    assert tl.to_content(51.5) == 50.0          # the 1.5 s hole does not count
    assert tl.to_content(85.0) == 82.0          # both holes gone
    assert tl.to_content(200.0) == 0.0          # content time restarts per range


def test_content_time_undefined_inside_a_hole_and_outside_the_corpus():
    tl = timeline()
    assert tl.to_content(50.7) is None
    assert tl.to_content(150.0) is None
    assert tl.to_content(400.0) is None


def test_round_trip_file_content_file():
    tl = timeline()
    for t in (0.0, 12.5, 49.999, 51.5, 60.0, 81.5, 99.0, 200.0, 250.0):
        assert abs(tl.to_file(tl.range_index(t), tl.to_content(t)) - t) < 1e-6


def test_content_time_is_monotonic():
    tl = timeline()
    times = [t / 10 for t in range(0, 1000) if tl.is_content(t / 10)]
    values = [tl.to_content(t) for t in times]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_position_on_a_hole_edge_lands_after_the_hole():
    # 50 s of content is exactly where the first hole starts; a fragment must
    # not begin with material that is not there
    assert timeline().to_file(0, 50.0) == 51.5


def test_black_stretches_do_not_change_content_time():
    with_skips = Timeline(RANGES, HOLES, SKIPS)
    without = Timeline(RANGES, HOLES)
    assert with_skips.content_length(0.0, 100.0) == without.content_length(0.0, 100.0)
    assert with_skips.to_content(85.0) == without.to_content(85.0)


# ----------------------------------------------------------------- fragments

def test_pieces_of_a_fragment():
    tl = timeline()
    assert tl.content_pieces(40.0, 60.0) == [(40.0, 50.0), (51.5, 60.0)]
    assert tl.content_pieces(10.0, 20.0) == [(10.0, 20.0)]
    assert tl.content_pieces(45.0, 85.0) == [(45.0, 50.0), (51.5, 80.0), (81.5, 85.0)]
    assert tl.content_pieces(50.2, 51.0) == []


def test_content_length_matches_the_pieces():
    tl = timeline()
    for start, end in ((40.0, 60.0), (0.0, 100.0), (10.0, 20.0), (45.0, 85.0)):
        assert abs(tl.content_length(start, end)
                   - total(tl.content_pieces(start, end))) < 1e-9
    assert tl.content_length(0.0, 100.0) == 97.0          # 100 minus two 1.5 s holes


def test_usable_pieces_drop_black_but_keep_the_fragment_alive():
    tl = timeline()
    assert tl.usable_pieces(25.0, 35.0) == [(25.0, 30.0), (32.0, 35.0)]
    # a fragment lying entirely inside a black stretch keeps its frames:
    # an empty representation would be worse than a black one
    assert tl.usable_pieces(30.5, 31.5) == [(30.5, 31.5)]


def test_a_timeline_without_masks_is_the_identity():
    plain = Timeline.from_ranges([(0.0, 100.0)])
    assert plain.to_content(42.0) == 42.0
    assert plain.content_length(10.0, 30.0) == 20.0
    assert plain.content_pieces(10.0, 30.0) == [(10.0, 30.0)]


# ------------------------------------------------------------------ sampling

def test_select_rejects_holes_and_black_frames():
    tl = timeline()
    times = [25.0, 31.0, 49.0, 50.7, 85.0, 150.0]
    assert tl.select(times) == [True, False, True, False, True, False]


def test_sampling_without_masks_is_evenly_spaced():
    plain = Timeline.from_ranges([(0.0, 100.0)])
    assert plain.sample_content(0.0, 10.0, 8) == [1.25 * k for k in range(8)]


def test_sampling_steps_over_a_hole():
    tl = Timeline([(0.0, 100.0)], [(5.0, 6.5)])
    out = tl.sample_content(0.0, 11.5, 8)          # 10 s of content, step 1.25
    assert len(out) == 8
    assert all(not tl.in_hole(t) for t in out)
    content = [tl.to_content(t) for t in out]
    assert all(abs(b - a - 1.25) < 1e-6 for a, b in zip(content, content[1:]))


def test_frame_indices_are_sorted_and_clamped():
    tl = timeline()
    indices = tl.frame_indices(40.0, 60.0, 8, fps=24.0, last_frame=1000)
    assert indices == sorted(indices)
    assert max(indices) <= 1000
    assert all(not tl.in_hole(i / 24.0) for i in indices)


# ------------------------------------------------ X-CLIP windows on the axis

def test_xclip_windows_match_the_old_arithmetic_without_masks():
    from src.features.xclip import _content_windows, window_frame_indices

    tl = Timeline.from_ranges([(0.0, 100.0)])
    fps = 24.0
    start, end = 0.0, 10.0
    old = window_frame_indices(0, round((end - start) * fps), 10_000)
    new = _content_windows(start, end, tl, fps, 10_000)
    assert len(new) == len(old)
    assert all(abs(a - b) <= 1 for wa, wb in zip(new, old) for a, b in zip(wa, wb))


def test_xclip_windows_step_over_a_mask():
    from src.features.xclip import _content_windows

    tl = Timeline([(0.0, 100.0)], [(4.0, 5.5)], [(8.0, 8.5)])
    windows = _content_windows(0.0, 11.5, tl, fps=24.0, last_frame=10_000)
    assert windows and all(len(w) == 8 for w in windows)
    for window in windows:
        assert window == sorted(window)
        for index in window:
            assert tl.is_usable(index / 24.0), index / 24.0


def test_xclip_windows_cover_the_whole_fragment():
    """The tail of a fragment used to fall outside every window (integer division)."""
    from src.features.xclip import WINDOW_FRAMES, window_offsets

    for length in (50, 64, 100, 128, 240, 250, 300, 1000):
        offsets = window_offsets(length, WINDOW_FRAMES, 1.0)
        assert offsets[0] == 0
        assert offsets == sorted(offsets)
        assert min(offsets[-1] + WINDOW_FRAMES, length) == length
        # windows stay whole and none of them degenerates into a sliver
        assert all(b - a >= WINDOW_FRAMES / 2
                   for a, b in zip(offsets, offsets[1:]))


def test_xclip_and_slowfast_tile_a_fragment_the_same_way():
    """Chapter 4 ties both models to one window length, so the stretches match."""
    from src.features import motion
    from src.features.xclip import WINDOW_FRAMES, window_offsets

    fps, span = 24.0, WINDOW_FRAMES / 24.0
    tl = Timeline([(0.0, 100.0)], [(4.0, 5.5)], [])
    starts = motion.window_starts(0.0, 11.5, tl, fps)
    offsets = window_offsets(tl.content_length(0.0, 11.5), span, fps)
    assert len(starts) == len(offsets)
    assert starts[-1][1] >= 11.5 - 1.0 / fps


def test_last_window_reaches_the_end_of_the_fragment():
    from src.features.xclip import _content_windows

    fps = 23.976024
    tl = Timeline.from_ranges([(0.0, 100.0)])
    windows = _content_windows(0.0, 10.0, tl, fps, 10_000)
    last_frame_of_fragment = round(10.0 * fps) - 1
    # the final window opens late enough to still hold the last frame
    assert windows[-1][0] + 63 >= last_frame_of_fragment


# ------------------------------------------------------ the two mask classes

def test_masks_split_by_tag_not_by_length():
    from src.annotation import intervals as iv
    from src.annotation.ranges import split_masks

    items = [
        iv.Interval("tbbt_s03e02_m001", iv.MASK, 0.0, 3.4),                       # logo
        iv.Interval("tbbt_s03e02_m002", iv.MASK, 138.0, 163.0),                   # titles
        iv.Interval("tbbt_s03e02_m003", iv.MASK, 733.9, 735.2, tags=["przejscie"]),
        iv.Interval("tbbt_s03e02_d1", iv.EVENT, 10.0, 14.0, "x" * 10),
    ]
    structural, transitions = split_masks(items)
    assert structural == [(0.0, 3.4), (138.0, 163.0)]
    assert transitions == [(733.9, 735.2)]
    # an untagged short mask is structural, a tagged long one is a transition
    odd = [iv.Interval("tbbt_s03e02_m004", iv.MASK, 500.0, 501.2),
           iv.Interval("tbbt_s03e02_m005", iv.MASK, 600.0, 640.0, tags=["przejscie"])]
    assert split_masks(odd) == ([(500.0, 501.2)], [(600.0, 640.0)])
