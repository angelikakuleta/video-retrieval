"""Window layout and boundary decoding of the learned shot detector."""

import numpy as np
import pytest

from src.segmentation.transnet import PAD, STRIDE, WINDOW, boundaries_from_predictions


def layout(n):
    """The padded length and window starts the predictor derives for n frames."""
    tail = PAD + STRIDE - (n % STRIDE or STRIDE)
    length = PAD + n + tail
    return length, list(range(0, length - WINDOW + 1, STRIDE))


@pytest.mark.parametrize("n", [1, 49, 50, 51, 99, 100, 101, 149, 150,
                               30000, 30040, 30049, 30050, 30051, 31914])
def test_every_window_fits_inside_the_padding(n):
    # the crash this guards against: a window running past the end of the
    # padded sequence, which numpy reports only as "shapes differ"
    length, starts = layout(n)
    assert starts, "at least one window is always produced"
    assert starts[-1] + WINDOW <= length


@pytest.mark.parametrize("n", [1, 49, 50, 51, 99, 100, 101, 149, 150, 31914])
def test_the_windows_cover_every_frame(n):
    _, starts = layout(n)
    assert len(starts) * STRIDE >= n


def test_boundaries_are_the_first_frame_after_a_transition():
    fps = 25.0
    predictions = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.8, 0.2])
    # runs at 2-3 and 6 -> boundaries at frames 4 and 7
    assert boundaries_from_predictions(predictions, fps) == [round(4 / fps, 3),
                                                            round(7 / fps, 3)]


def test_a_sequence_without_transitions_has_no_boundaries():
    assert boundaries_from_predictions(np.zeros(50), 25.0) == []


def test_a_transition_running_to_the_end_yields_no_boundary():
    # nothing follows it, so there is no first frame of a new shot
    assert boundaries_from_predictions(np.array([0.1, 0.9, 0.9]), 25.0) == []
