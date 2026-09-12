"""Single registry of dataset names and on-disk locations.

A dataset is named the same in the configuration and on disk. Every module
resolves paths through this registry, so the set of valid names and the layout
below it exist in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = {"tbbt": "tbbt", "office": "office", "vatex": "vatex"}
SPLITS = ("dev", "test")

#: which root the recordings of a dataset sit under. The series are normalized
#: into data/processed; the VATEX clips are downloaded excerpts that are already
#: what they are meant to be, so they stay in data/raw where the acquisition put
#: them -- moving 2560 files would invalidate the frozen correctness check.
DATASET_ROOTS = {"tbbt": "processed", "office": "processed", "vatex": "raw"}


def dataset_dir(dataset: str) -> str:
    """Directory name of a dataset; also validates the name."""
    try:
        return DATASET_DIRS[dataset]
    except KeyError:
        raise ValueError(f"unknown dataset: {dataset!r}") from None


def ranges_csv(dataset: str) -> Path:
    name = dataset_dir(dataset)
    return ROOT / "data" / "interim" / name / f"{name}_ranges.csv"


def holes_csv(dataset: str) -> Path:
    """Transition masks of every episode -- the holes of the content axis.

    A separate file from ``<dataset>_ranges.csv`` because the two answer
    different questions: the ranges say where the corpus is, the holes say
    which seconds inside it do not count. A dataset without transition masks
    (The Office, VATEX) has no such file, and its absence means "none".
    """
    name = dataset_dir(dataset)
    return ROOT / "data" / "interim" / name / f"{name}_holes.csv"


def black_csv(dataset: str) -> Path:
    """Stretches of (almost) black picture, from one ``blackdetect`` pass.

    A cache artifact, not an annotation: it is derived from the recordings by
    a fixed rule, so it lives under ``data/cache`` and may be deleted and
    recomputed at any time without touching anything a human decided.
    """
    return ROOT / "data" / "cache" / "black" / f"{dataset_dir(dataset)}_black.csv"


def queries_jsonl(dataset: str, split: str) -> Path:
    name = dataset_dir(dataset)
    return ROOT / "data" / "annotations" / name / f"{name}_queries_{split}.jsonl"


def video_path(dataset: str, video_file: str) -> Path:
    """Path of one recording; ``video_file`` may carry a subdirectory.

    The VATEX parts differ by subdirectory rather than by root: the test clips
    lie in ``data/raw/vatex/test/``, the development ones in ``dev/``.
    """
    name = dataset_dir(dataset)
    return ROOT / "data" / DATASET_ROOTS[dataset] / name / video_file


def segments_csv(strategy: str, dataset: str) -> Path:
    return (ROOT / "data" / "cache" / "segmentation" / strategy
            / f"{dataset_dir(dataset)}_segments.csv")


def frame_cache_dir(model: str, dataset: str) -> Path:
    return ROOT / "data" / "cache" / "embeddings" / model / dataset_dir(dataset)


def index_dir(dataset: str, strategy: str, model: str, split: str) -> Path:
    return (ROOT / "data" / "cache" / "indexes" / dataset_dir(dataset)
            / strategy / model / split)


def runs_dir() -> Path:
    return ROOT / "results" / "runs"
