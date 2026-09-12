"""Embeddings of the aligned face crops: the region signal of experiment E5.

The open-vocabulary half of E5: every accepted crop is encoded by the visual
encoder of the base representation and compared with the query directly, so the
fragment takes the highest similarity any of its faces reaches. Nothing here
names an expression, which is the property the comparison with HSEmotion is
about.

The embeddings depend on the encoder, so they are cached per (encoder, episode).
Switching the base representation recomputes the vectors but never the
detections.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.data import datasets
from src.features import episode_cache, faces

#: crops encoded at once
BATCH = 64


def cache_dir(encoder: str, dataset: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "faces" / "regions" / encoder
            / datasets.dataset_dir(dataset))


def episode_npz(encoder: str, dataset: str, episode: str) -> Path:
    return cache_dir(encoder, dataset) / f"{episode}.npz"


def load_episode(encoder: str, dataset: str, episode: str) -> np.ndarray:
    """Embeddings (n_faces, dim) aligned with the crop buffer of the episode."""
    with np.load(episode_npz(encoder, dataset, episode)) as data:
        return data["embeddings"]


def ensure_regions(
    dataset: str,
    episodes: dict[str, dict],
    encoder: str,
    encoder_factory: Callable[[], Any],
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Encodes the crops of every episode; returns the episodes covered."""

    def compute(episode: str, video: Path, target: Path) -> str:
        crops = faces.load_episode(dataset, episode)["crop"]
        if not len(crops):
            np.savez_compressed(target, embeddings=np.zeros((0, 1), np.float32))
            return "no faces"
        model = encoder_factory()
        chunks = [model.encode_images(list(crops[start:start + BATCH]))
                  for start in range(0, len(crops), BATCH)]
        embeddings = np.concatenate(chunks).astype(np.float32)
        np.savez_compressed(target, embeddings=embeddings)
        return f"{len(embeddings)} faces, dim {embeddings.shape[1]}"

    with_faces = {ep: entry for ep, entry in episodes.items()
                  if faces.episode_npz(dataset, ep).exists()}
    for episode in sorted(set(episodes) - set(with_faces)):
        log(f"  {episode}: no face buffer - run the face detection first")
    return episode_cache.ensure(dataset, with_faces,
                                lambda ep: episode_npz(encoder, dataset, ep),
                                compute, f"face region embeddings ({encoder})",
                                force=force, log=log)
