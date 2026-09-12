"""Sequential frame decoding for the shot-boundary detectors.

One pass over the file feeds both detectors at once: every frame is decoded
once and downscaled to the two sizes they need -- a small BGR frame for the
histogram method and a 48x27 RGB frame for TransNetV2. Sequential reading
(no seeking) keeps the pass at the decoder's native speed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SMALL_WIDTH = 320          # histogram frames
TINY_SIZE = (48, 27)       # TransNetV2 input (width, height), forced by the net


class FrameStream:
    """Iterates ``(index, small_bgr, tiny_rgb)`` over all frames of a file.

    ``small_bgr`` and ``tiny_rgb`` are produced only when the corresponding
    flag is set, otherwise ``None`` -- so a run that needs one detector does
    not pay for the other's resizing.
    """

    def __init__(self, path: Path | str, small: bool = True, tiny: bool = True):
        self.reader = cv2.VideoCapture(str(path))
        self.fps = self.reader.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_count = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.fps <= 0 or self.frame_count <= 0:
            self.reader.release()
            raise ValueError(f"could not read the video: {path}")
        height = self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        width = self.reader.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        self.small_size = (SMALL_WIDTH, max(1, round(height * SMALL_WIDTH / max(width, 1))))
        self.small = small
        self.tiny = tiny

    def __iter__(self):
        index = 0
        while True:
            ok, frame = self.reader.read()
            if not ok:
                break
            small = cv2.resize(frame, self.small_size, interpolation=cv2.INTER_AREA) \
                if self.small else None
            tiny = cv2.cvtColor(
                cv2.resize(frame, TINY_SIZE, interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB) if self.tiny else None
            yield index, small, tiny
            index += 1
        self.reader.release()

    def close(self):
        self.reader.release()


def tiny_buffer(frames: list[np.ndarray]) -> np.ndarray:
    """Stacks collected 48x27 RGB frames into the (N, 27, 48, 3) uint8 array."""
    return np.stack(frames).astype(np.uint8)
