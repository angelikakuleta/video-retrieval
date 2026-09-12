"""Unlearned shot-boundary detection: histogram difference of neighbouring frames.

Variant E1-B. Each decoded frame (downscaled, BGR) is described by a 2D
histogram over hue and saturation in HSV space; the difference between
neighbouring frames is one minus the correlation of their histograms
(``cv2.HISTCMP_CORREL``). A boundary is declared where the difference exceeds
a fixed global threshold -- the method stays a deliberately simple, unlearned
reference point for TransNetV2, with a single parameter set in advance.

The boundary time is the time of the first frame of the new shot, consistent
with the half-open ``[start, end)`` convention of the annotations.
"""

from __future__ import annotations

import cv2
import numpy as np

THRESHOLD = 0.5      # boundary when 1 - correlation > THRESHOLD
BINS = (32, 32)      # hue x saturation


def frame_histogram(small_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(BINS), [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


class HistogramDetector:
    """Consumes frames one by one, collects boundary times."""

    def __init__(self, fps: float, threshold: float = THRESHOLD):
        self.fps = fps
        self.threshold = threshold
        self.boundaries: list[float] = []
        self._previous: np.ndarray | None = None

    def feed(self, index: int, small_bgr: np.ndarray) -> None:
        hist = frame_histogram(small_bgr)
        if self._previous is not None:
            difference = 1.0 - cv2.compareHist(self._previous, hist, cv2.HISTCMP_CORREL)
            if difference > self.threshold:
                self.boundaries.append(round(index / self.fps, 3))
        self._previous = hist
