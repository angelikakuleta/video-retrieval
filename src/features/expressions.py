"""Facial expression recognition: the mimicry signal of experiment E5.

HSEmotion turns an aligned crop into a distribution over 8 expression classes and
the signal follows the motion formula: the query is compared with each class NAME
separately, then averaged with the model's probabilities. With only 8 classes
none is dropped.

This is the closed-vocabulary half of E5; the other half embeds the same crops
and compares them with the query directly, which is what makes the comparison
about the vocabulary rather than about the material.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data import datasets
from src.features import episode_cache, faces

#: the variant named in chapter 5 (EfficientNet-B0, 8 classes)
MODEL = "enet_b0_8_best_vgaf"

#: crops classified at once
BATCH = 64


def cache_dir(dataset: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "faces" / "hsemotion"
            / datasets.dataset_dir(dataset))


def episode_npz(dataset: str, episode: str) -> Path:
    return cache_dir(dataset) / f"{episode}.npz"


def load_episode(dataset: str, episode: str) -> dict:
    """``distribution`` (n_faces, 8) aligned with the crop buffer, plus ``classes``."""
    with np.load(episode_npz(dataset, episode)) as data:
        return {"distribution": data["distribution"], "classes": list(data["classes"])}


def class_names(model: str = MODEL) -> list[str]:
    """The 8 expression classes, in the order the classifier returns them."""
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    recognizer = HSEmotionRecognizer(model_name=model)
    return [recognizer.idx_to_class[i] for i in sorted(recognizer.idx_to_class)]


def ensure_expressions(
    dataset: str,
    episodes: dict[str, dict],
    model: str = MODEL,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Classifies the crops of every episode; returns the episodes covered."""
    box: dict = {}

    def recognizer():
        if "model" not in box:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

            log(f"loading HSEmotion ({model})")
            box["model"] = HSEmotionRecognizer(model_name=model)
        return box["model"]

    def compute(episode: str, video: Path, target: Path) -> str:
        crops = faces.load_episode(dataset, episode)["crop"]
        net = recognizer()
        names = [net.idx_to_class[i] for i in sorted(net.idx_to_class)]
        chunks = []
        for start in range(0, len(crops), BATCH):
            batch = list(crops[start:start + BATCH])
            _, scores = net.predict_multi_emotions(batch, logits=False)
            chunks.append(np.asarray(scores, dtype=np.float32)[:, :len(names)])
        distribution = (np.concatenate(chunks) if chunks
                        else np.zeros((0, len(names)), np.float32))
        np.savez_compressed(target, distribution=distribution,
                            classes=np.array(names))
        return f"{len(distribution)} faces over {len(names)} classes"

    def needed(episode: str) -> Path:
        return episode_npz(dataset, episode)

    with_faces = {ep: entry for ep, entry in episodes.items()
                  if faces.episode_npz(dataset, ep).exists()}
    missing = sorted(set(episodes) - set(with_faces))
    for episode in missing:
        log(f"  {episode}: no face buffer - run the face detection first")
    return episode_cache.ensure(dataset, with_faces, needed, compute,
                                f"expressions ({model})", force=force, log=log)
