"""Embeddings of the closed vocabularies a signal compares a query against.

Object classes, expression classes and Kinetics actions are fixed lists, so a
fragment stores identifiers and the comparison happens between the query and the
class NAME, encoded by the same text encoder as the query (chapter 4).

Cached once per (encoder, vocabulary). The names are stored next to the vectors:
a vocabulary whose contents changed must not be silently matched against
embeddings of the old one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.data import datasets


def vocabulary_npz(encoder: str, name: str) -> Path:
    return datasets.ROOT / "data" / "cache" / "vocab" / encoder / f"{name}.npz"


def load(encoder: str, name: str) -> tuple[list[str], np.ndarray] | None:
    """``(names, embeddings)`` of a cached vocabulary, or ``None``."""
    path = vocabulary_npz(encoder, name)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return list(data["names"]), data["embeddings"]


def ensure(name: str, names: list[str], encoder: str,
           encoder_factory: Callable[[], Any],
           log: Callable[[str], None] = print) -> np.ndarray:
    """Embeddings of ``names`` in the encoder's space, recomputed if the list changed."""
    cached = load(encoder, name)
    if cached is not None and cached[0] == list(names):
        return cached[1]
    if cached is not None:
        log(f"vocabulary {name!r} changed ({len(cached[0])} -> {len(names)} names)"
            " - re-encoding")

    embeddings = encoder_factory().encode_texts(list(names)).astype(np.float32)
    path = vocabulary_npz(encoder, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.array(list(names)), embeddings=embeddings)
    log(f"vocabulary -> {path.relative_to(datasets.ROOT)} ({len(names)} names)")
    return embeddings
