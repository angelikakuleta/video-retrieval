"""Assembly of the scene pipeline for the VATEX dataset -- driven by configuration.

Ties the ``src`` layers into two high-level operations used both by the script
``scripts/vatex_check.py`` and the notebook ``notebooks/vatex_check.ipynb``:
writing the query file of the correctness check and building (or loading from
cache) the scene index. The configuration is a dict loaded from YAML by
:mod:`src.utils.config`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.openclip import ClipEncoder
from src.features.scene import SceneExtractor
from src.indexing.faiss_index import Collection, build_matrix, build_or_load
from src.utils import config as conf
from src.utils import vatex
from src.utils.queries import save_jsonl


def write_check_queries(
    check_vids: list[str],
    descriptions: pd.DataFrame,
    cfg: dict,
) -> Path:
    """Saves the query file of the implementation correctness check.

    Ten descriptions of every clip over the full ``ok_clips`` set -- deliberately
    including the recordings of the training leak, because here comparability
    with the published values matters, not cleanliness with respect to training
    (chapter 6).

    The experiment queries are a different scope AND a different file: they are
    built in ``notebooks/prepare_data/vatex_03_test_annotations.ipynb``, where the
    hand-filled tags are merged into them. Writing them here as well would strip
    those tags off on the next run of the check.
    """
    return save_jsonl(vatex.build_check_queries(check_vids, descriptions),
                      conf.path(cfg["paths"]["check"]))


def build_collection(vids: list[str], encoder: ClipEncoder, cfg: dict) -> Collection:
    """Builds (or loads from cache) the scene index for the given clips."""
    extractor = SceneExtractor(
        encoder,
        step=cfg["frames"]["step"],
        threshold=cfg["frames"]["similarity_threshold"],
    )
    clips_dir = conf.path(cfg["paths"]["clips"])
    pairs = [(vid, vatex.clip_path(vid, clips_dir)) for vid in vids]

    counter = {"n": 0}

    def progress(_vid: str) -> None:
        counter["n"] += 1
        if counter["n"] % 100 == 0:
            print(f"  encoded {counter['n']}/{len(pairs)} clips", flush=True)

    metadata = {"model": cfg["model"]["encoder"], "weights": cfg["model"]["weights"]}
    return build_or_load(
        conf.path(cfg["paths"]["cache"]),
        build=lambda: build_matrix(pairs, extractor.embedding, progress),
        metadata=metadata,
    )
