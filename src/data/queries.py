"""Reading the test queries of a dataset split."""

from __future__ import annotations

from src.data import datasets
from src.utils.queries import Query, load_jsonl


def load_queries(dataset: str, split: str) -> list[Query]:
    """Queries of one split; raises when the file does not exist yet."""
    path = datasets.queries_jsonl(dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"no query file for split {split!r}: {path}")
    return load_jsonl(path)


def episode_of(query: Query) -> str:
    """Which recording a query points at, in the convention of its dataset.

    A series query names the recording with the dataset in front
    (``tbbt_s03e02`` -> ``s03e02``); a VATEX clip name IS the recording, and
    cutting it at the first underscore would take the video identifier apart.
    Deciding on ``source`` keeps that convention in ONE place -- the relevance
    sets used to carry their own copy of it, and it was wrong for VATEX.
    """
    if query.source == "vatex":
        return query.vid_name
    prefix, _, episode = query.vid_name.partition("_")
    return episode or prefix
