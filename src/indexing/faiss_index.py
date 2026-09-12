"""FAISS vector index and the catalog of segment metadata.

The collection stores one normalized vector per segment plus a catalog that
maps vectors to clip identifiers. Exact search (``IndexFlatIP``) is used: with a
few thousand vectors per dataset it is cheap, and it removes approximation
error as a confounding factor in the measurement. The dot product of normalized
vectors is the cosine similarity.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "scene.faiss"
CATALOG_FILE = "catalog.json"


@dataclass
class Collection:
    """FAISS index together with the catalog of segment identifiers."""

    index: faiss.Index
    vids: list[str]

    @property
    def size(self) -> int:
        return self.index.ntotal

    def matrix(self) -> np.ndarray:
        """Reconstructs the full matrix of segment embeddings (size, dim)."""
        # np.asarray: reconstruct_n returns a numpy array, but the faiss stub also
        # has a torch variant -- the explicit wrap pins the type to ndarray (a no-op at runtime).
        return np.asarray(self.index.reconstruct_n(0, self.index.ntotal))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (similarities, indices) of the ``k`` nearest segments."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        return self.index.search(queries, k)


def build_index(matrix: np.ndarray, vids: list[str]) -> Collection:
    """Builds a collection from a ready embedding matrix and a list of identifiers."""
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if matrix.shape[0] != len(vids):
        raise ValueError("number of vectors differs from the number of identifiers")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return Collection(index=index, vids=list(vids))


def build_matrix(
    pairs: Iterable[tuple[str, Path]],
    embedding: Callable[[Path], np.ndarray],
    progress: Callable[[str], None] | None = None,
    skip_errors: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Computes the embedding of every segment and returns (matrix, identifiers).

    ``embedding`` is a function file -> vector (e.g. ``SceneExtractor.embedding``).
    ``progress`` (optional) is called with the identifier after each clip.
    With ``skip_errors`` an unreadable clip is skipped with a warning, so that a
    single corrupt file does not abort the whole indexing run; the identifiers
    stay aligned with the rows of the matrix.
    """
    vectors, vids = [], []
    for vid, path in pairs:
        try:
            vector = embedding(path)
        except Exception as error:  # the batch must survive a single failure of any clip
            if not skip_errors:
                raise
            print(f"  SKIPPED {vid}: {error}", file=sys.stderr, flush=True)
            continue
        vectors.append(vector)
        vids.append(vid)
        if progress is not None:
            progress(vid)
    return np.stack(vectors).astype(np.float32), vids


def save(collection: Collection, directory: Path | str, metadata: dict | None = None) -> Path:
    """Saves the FAISS index and the catalog of identifiers to the directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(collection.index, str(directory / INDEX_FILE))
    content = {"vids": collection.vids, "dim": collection.index.d, **(metadata or {})}
    (directory / CATALOG_FILE).write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return directory


def load(directory: Path | str) -> Collection:
    """Loads a collection saved by :func:`save`."""
    directory = Path(directory)
    index = faiss.read_index(str(directory / INDEX_FILE))
    catalog = json.loads((directory / CATALOG_FILE).read_text(encoding="utf-8"))
    return Collection(index=index, vids=catalog["vids"])


def exists(directory: Path | str) -> bool:
    """Whether the directory holds a saved index ready to be loaded."""
    return (Path(directory) / INDEX_FILE).exists()


def build_or_load(
    directory: Path | str,
    build: Callable[[], tuple[np.ndarray, list[str]]],
    metadata: dict | None = None,
) -> Collection:
    """Loads the index from the directory, or builds and saves it when missing.

    ``build`` returns (matrix, identifiers) -- it is called only when the index
    does not exist yet. This lets subsequent runs skip the costly frame encoding.
    """
    if exists(directory):
        return load(directory)
    matrix, vids = build()
    collection = build_index(matrix, vids)
    save(collection, directory, metadata)
    return collection
