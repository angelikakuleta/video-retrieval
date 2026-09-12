"""Scene signal -- a segment embedding assembled from the embeddings of its frames.

A segment (in VATEX: the whole clip) is represented by a single vector: frames
are sampled on a time grid, frames nearly identical to the previously kept one
(cosine similarity above the threshold) are dropped, and the remaining ones are
averaged and normalized. Dropping duplicates prevents static shots from
dominating the average.
"""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np

from src.segmentation.frames import STEP, sample_frames

if TYPE_CHECKING:
    from src.features.openclip import ClipEncoder

SIMILARITY_THRESHOLD = 0.9


def drop_duplicates(embeddings: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> np.ndarray:
    """Keeps the frames that differ from the last kept one (cosine <= threshold).

    The embeddings are normalized, so the dot product is the cosine similarity.
    The first frame is always kept.
    """
    kept = [0]
    for i in range(1, len(embeddings)):
        if float(embeddings[i] @ embeddings[kept[-1]]) <= threshold:
            kept.append(i)
    return embeddings[kept]


def build_clip_embedding(frame_embeddings: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> np.ndarray:
    """Averages the frame embeddings (after dropping duplicates) and normalizes."""
    selected = drop_duplicates(frame_embeddings, threshold)
    mean = selected.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 0 else mean.astype(np.float32)


class SceneExtractor:
    """Turns a clip file into a single normalized scene embedding."""

    def __init__(
        self,
        encoder: "ClipEncoder",
        step: float = STEP,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.encoder = encoder
        self.step = step
        self.threshold = threshold

    def embedding(self, path: Path | str) -> np.ndarray:
        frames, _ = sample_frames(path, self.step)
        frame_embeddings = self.encoder.encode_images(frames)
        return build_clip_embedding(frame_embeddings, self.threshold)
