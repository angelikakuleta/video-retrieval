"""Sampling frames from a video on a fixed time grid.

The step of 1.25 s was chosen so that the shortest admissible segment (3 s)
contains at least two frames. The grid is the contract every single-frame
component shares -- scene embeddings, captions, object and face detections are
computed for the SAME moments -- so a difference between two configurations can
never come from a difference in what they were shown.

Frames are yielded one at a time: a full episode is around a thousand
full-resolution images, several gigabytes if held at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.utils.settings import FRAME_STEP

if TYPE_CHECKING:
    from PIL import Image

#: sampling grid, defined once in :mod:`src.utils.settings`
STEP = FRAME_STEP


def _sample_times(duration: float, step: float) -> list[float]:
    """Sampling moments on the grid; always at least one frame."""
    if duration <= 0:
        return [0.0]
    times = [float(t) for t in np.arange(0.0, duration, step)]
    return times or [0.0]


def iter_frames(path: Path | str, step: float = STEP) -> Iterator[tuple[float, "Image.Image"]]:
    """Yields ``(time, frame)`` for every grid point of the file.

    Reading is by frame number derived from the grid, so the same moment is sampled
    regardless of the frame rate.
    """
    import cv2
    from PIL import Image

    reader = cv2.VideoCapture(str(path))
    try:
        fps = reader.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"could not read the video: {path}")

        seen = 0
        for t in _sample_times(frame_count / fps, step):
            number = min(round(t * fps), frame_count - 1)
            reader.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, frame = reader.read()
            if not ok:
                continue
            seen += 1
            yield round(number / fps, 3), Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not seen:
            raise ValueError(f"no frames were read from: {path}")
    finally:
        reader.release()


def sample_frames(path: Path | str, step: float = STEP) -> tuple[list, list[float]]:
    """The whole grid at once: ``(frames, times)``; prefer :func:`iter_frames`."""
    frames, times = [], []
    for time, frame in iter_frames(path, step):
        frames.append(frame)
        times.append(time)
    return frames, times


def iter_batches(path: Path | str, step: float = STEP, batch: int = 32
                 ) -> Iterator[tuple[list[float], list]]:
    """Grid frames in batches of at most ``batch``: ``(times, frames)``."""
    times: list[float] = []
    frames: list = []
    for time, frame in iter_frames(path, step):
        times.append(time)
        frames.append(frame)
        if len(frames) == batch:
            yield times, frames
            times, frames = [], []
    if frames:
        yield times, frames
