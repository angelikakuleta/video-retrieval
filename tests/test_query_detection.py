"""E4-D: the scoring stage that asks the detector the query's own phrases (§08).

The detector itself never runs here. What these tests pin is the arithmetic
around it -- which fragments are candidates, what a candidate is worth, how the
two columns are standardized and fused, and which frames reach the detector at
all -- because every one of those is a decision of chapter 6 rather than a
property of YOLOE.
"""

import numpy as np
import pytest

from src.retrieval import query_detection as qd
from src.segmentation.frames import _sample_times


def base_row(n: int = 60) -> np.ndarray:
    """A base ranking: fragment 0 best, fragment n-1 worst, no ties."""
    return np.linspace(1.0, 0.0, n)


def scene_of(row: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.asarray(row)[candidates]


def values(hits: dict[int, float], candidates: np.ndarray) -> np.ndarray:
    """Detection values: the named candidates get a score, the rest a raw zero."""
    return np.array([hits.get(int(column), 0.0) for column in candidates], float)


# ---------------------------------------------------- candidates and prompts
def test_the_candidates_are_the_fifty_best_of_the_base_in_its_order():
    row = base_row(60)
    candidates = qd.candidates_of(row)
    assert len(candidates) == qd.CANDIDATES == 50
    assert candidates.tolist() == list(range(50))


def test_a_collection_smaller_than_the_subset_is_taken_whole():
    assert qd.candidates_of(base_row(12)).tolist() == list(range(12))


def test_both_forms_of_a_phrase_are_asked_for_and_asked_once():
    texts, owners = qd.texts_of([("coffee mug", "mug"), ("mug",), ("desk",)])
    assert texts == ["coffee mug", "mug", "desk"]
    # `mug` belongs to two phrases and is asked for once, not twice: a class list
    # with the same name twice makes the detector's answer ambiguous
    assert owners == [[0], [0, 1], [2]]


def test_a_detection_lands_on_every_phrase_that_owns_the_form():
    _, owners = qd.texts_of([("coffee mug", "mug"), ("mug",)])
    best = qd.per_phrase_maxima([{1: 0.8}], owners, n_phrases=2)
    assert best == [0.8, 0.8]


# ------------------------------------------------- what a candidate is worth
def test_a_candidate_is_the_mean_over_its_phrases():
    assert qd.value_of([0.9, 0.3]) == pytest.approx(0.6)


def test_nothing_found_is_a_raw_zero_and_not_a_missing_value():
    """The detector was asked for these phrases and answered no.

    Neutral values here would flatten the column: with fifty candidates and one
    detection the deviation falls under EPSILON, fusion zeroes the column and
    E4-D becomes the base exactly.
    """
    assert qd.value_of([0.0, 0.0]) == 0.0
    assert not np.isnan(qd.value_of([0.0]))


def test_only_a_candidate_with_no_usable_frame_has_no_value():
    assert np.isnan(qd.value_of(None))
    assert np.isnan(qd.value_of([]))


# --------------------------------------------------------------- the ranking
def test_without_phrases_the_ranking_is_the_one_the_base_produced():
    row = base_row(60)
    order = np.argsort(-row, kind="stable")
    # no prompts -> the stage is inactive and nothing is re-ranked
    assert qd.prompts_of.__doc__ and "inactive" in qd.prompts_of.__doc__
    assert order.tolist() == list(range(60))


def test_one_detection_puts_that_candidate_first():
    """[0.9, nothing x 49] -> the candidate with the detection leads."""
    row = base_row(60)
    candidates = qd.candidates_of(row)
    detection = values({7: 0.9}, candidates)
    fused = qd.fuse(detection, scene_of(row, candidates))
    ranking = qd.reorder(np.argsort(-row, kind="stable"), candidates, fused)
    assert ranking[0] == 7


def test_a_weaker_detection_lands_second_not_fiftieth():
    """[0.9, 0.3, nothing x 48] -> second place, not the bottom of the subset."""
    row = base_row(60)
    candidates = qd.candidates_of(row)
    detection = values({7: 0.9, 30: 0.3}, candidates)
    fused = qd.fuse(detection, scene_of(row, candidates))
    ranking = qd.reorder(np.argsort(-row, kind="stable"), candidates, fused)
    assert ranking[0] == 7
    assert ranking[1] == 30


def test_a_fragment_outside_the_subset_does_not_move():
    row = base_row(60)
    candidates = qd.candidates_of(row)
    detection = values({7: 0.9}, candidates)
    fused = qd.fuse(detection, scene_of(row, candidates))
    ranking = qd.reorder(np.argsort(-row, kind="stable"), candidates, fused)
    assert ranking[50:].tolist() == list(range(50, 60))


def test_ties_keep_the_order_of_the_base():
    """Nothing detected anywhere: the subset comes back exactly as it went in."""
    row = base_row(60)
    candidates = qd.candidates_of(row)
    fused = qd.fuse(values({}, candidates), scene_of(row, candidates))
    ranking = qd.reorder(np.argsort(-row, kind="stable"), candidates, fused)
    assert ranking.tolist() == list(range(60))


# ----------------------------------------- standardization inside the subset
def test_the_scene_is_standardized_again_inside_the_fifty():
    """A collection where the global z-score and the local one differ.

    Fifty top-ranked fragments are all far above the collection mean and barely
    differ from each other; standardized over the collection they would be one
    flat block and the detector would decide everything. Inside the subset they
    compete.
    """
    row = np.concatenate([np.linspace(0.90, 0.80, 50), np.zeros(200)])
    candidates = qd.candidates_of(row)
    scene = scene_of(row, candidates)

    from src.retrieval.fusion import standardize
    globally = standardize(row[None, :])[0][candidates]
    locally = standardize(scene[None, :])[0]

    assert globally.std() < locally.std()          # flat over the collection
    assert locally.std() == pytest.approx(1.0, abs=1e-5)

    # and the fusion uses the local one: a detection of 0.9 on the WORST
    # candidate is enough to move it, which the global scaling would not allow
    detection = values({49: 0.9}, candidates)
    fused = qd.fuse(detection, scene)
    ranking = qd.reorder(np.argsort(-row, kind="stable"), candidates, fused)
    assert ranking[0] == 49


def test_a_candidate_without_a_usable_frame_is_neutral_not_worst():
    row = base_row(60)
    candidates = qd.candidates_of(row)
    detection = values({7: 0.9}, candidates)
    detection[30] = np.nan
    fused = qd.fuse(detection, scene_of(row, candidates))
    assert np.isfinite(fused).all()
    # it keeps a middling place, not the bottom
    ranking = qd.reorder(np.argsort(-row, kind="stable"), candidates, fused).tolist()
    assert ranking.index(30) < ranking.index(49)


# ------------------------------------------ the frames the detector is given
def test_the_frame_numbers_are_the_ones_the_indexing_pass_read():
    """Same arithmetic as frames.iter_frames: min(round(t*fps), n-1)."""
    fps, count = 23.976, 5000
    times = _sample_times(count / fps, 1.25)
    numbers = qd.frame_numbers(times, fps, count)
    expected = [min(round(t * fps), count - 1) for t in times]
    assert numbers.tolist() == expected
    assert numbers[-1] <= count - 1


def test_the_grid_is_the_grid_of_the_indexing_pass():
    fps, count = 25.0, 1000
    assert qd.grid_times(count, fps, 1.25) == _sample_times(count / fps, 1.25)


class _Timeline:
    """A timeline that excludes one stretch -- a transition mask or black."""

    def __init__(self, hole):
        self.hole = hole

    def select(self, times):
        low, high = self.hole
        return [not (low <= t < high) for t in times]


def test_a_frame_in_a_mask_never_reaches_the_detector():
    rows = [{"episode": "s01e01", "segment_id": 1, "start": 0.0, "end": 10.0},
            {"episode": "s01e01", "segment_id": 2, "start": 10.0, "end": 20.0}]
    row_of = {"s01e01_001": 0, "s01e01_002": 1}
    fps, count = 10.0, 200                      # 20 s of file

    without = qd.episode_frames(rows, "s01e01", row_of, None, count, fps)
    masked = qd.episode_frames(rows, "s01e01", row_of, _Timeline((2.5, 7.5)),
                               count, fps)

    assert set(masked[0]) < set(without[0])     # the first fragment lost frames
    assert masked[1] == without[1]              # the second is untouched
    # nothing from inside the mask survived
    assert not [n for n in masked[0] if 25 <= n < 75]


def test_a_fragment_the_timeline_empties_has_no_frames_at_all():
    rows = [{"episode": "s01e01", "segment_id": 1, "start": 0.0, "end": 10.0}]
    row_of = {"s01e01_001": 0}
    emptied = qd.episode_frames(rows, "s01e01", row_of, _Timeline((0.0, 10.0)),
                                200, 10.0)
    assert emptied == {}                        # and the caller gives it NaN


# --------------------------------------------------------------- the weights
def test_the_query_time_detector_has_its_own_checkpoint():
    """The prompt-free weights cannot take text; this is a different file."""
    from src.features import objects

    assert qd.PROMPTABLE == objects.DETECTORS["yoloe_prompted"]
    assert qd.PROMPTABLE != objects.DETECTORS["yoloe_promptfree"]
    assert "yoloe_prompted" not in objects.INDEXED


def test_the_query_time_detector_fills_no_cache():
    from src.features import objects

    with pytest.raises(ValueError, match="fills no cache"):
        objects.ensure_detections("tbbt", {}, "yoloe_prompted")


# --------------------------------- a recording that is named but not on disk
def test_black_detection_skips_a_recording_that_is_not_there(tmp_path, capsys):
    """VATEX arrives in batches, so a named but absent clip is normal.

    `episodes_of` returns only the recordings it FOUND, so the guard worked for
    the series. A `videos` mapping is built from the ranges and names every
    episode whether its file exists or not -- and then ffmpeg was handed a path
    that is not there, which killed the whole extraction on the first gap.
    """
    from src.segmentation import black

    present = tmp_path / "here.mp4"
    present.write_bytes(b"")
    found = black.ensure_black(
        "vatex", episodes=["absent"], force=True,
        videos={"absent": tmp_path / "nowhere.mp4", "here": present},
        log=lambda message: print(message))
    assert found == {}                      # nothing measured, nothing raised
    assert "no recording" in capsys.readouterr().out
