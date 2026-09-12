"""Implementation correctness check on the VATEX dataset -- base pipeline.

Run from the command line, driven by a YAML file. Performs the full run: writes
the check queries, builds or loads the FAISS scene index, and then computes the
ranking metrics on the check set (all ten descriptions of every clip as separate
queries). The notebook ``notebooks/vatex_check.ipynb`` uses the same functions
from the ``src`` package.

    python scripts/vatex_check.py [configs/vatex_base.yaml]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate_matrix
from src.features.openclip import ClipEncoder
from src.retrieval.pipeline import Pipeline
from src.retrieval.signals import SceneSignal
from src.utils import config as conf
from src.utils import vatex
from src.vatex_pipeline import build_collection, write_check_queries


def main(cfg_file: str = "configs/vatex_base.yaml") -> dict:
    cfg = conf.load(conf.path(cfg_file))
    ks = tuple(cfg["evaluation"]["ks"])

    report = vatex.load_report(conf.path(cfg["paths"]["report"]))
    descriptions = vatex.load_descriptions(conf.path(cfg["paths"]["descriptions"]))
    split = vatex.load_split(conf.path(cfg["paths"]["split"]))

    clips_dir = conf.path(cfg["paths"]["clips"])
    ok_vids = vatex.ok_clips(report, clips_dir)                        # 2560 -- implementation check
    test_vids = vatex.test_split_clips(report, split, clips_dir)       # 2489 -- experiments
    print(f"ok clips: {len(ok_vids)} | test split: {len(test_vids)} "
          f"(K400 leak: {len(ok_vids) - len(test_vids)})", flush=True)

    # Only the check queries are written here. The experiment file carries the
    # hand-filled tags and is built in the annotation notebook -- rewriting it
    # from the descriptions alone would silently drop them.
    written = write_check_queries(ok_vids, descriptions, cfg)
    print(f"saved queries: {written.name} (check)", flush=True)

    # The implementation correctness check deliberately runs on the full 'ok' set --
    # the candidate pool must include all clips, including the leaked ones, so
    # that the result is comparable with the published values.
    encoder = ClipEncoder(cfg["model"]["encoder"], cfg["model"]["weights"])
    collection = build_collection(ok_vids, encoder, cfg)
    print(f"collection: {collection.size} segments", flush=True)

    queries = vatex.build_check_queries(ok_vids, descriptions)
    texts = [q.desc for q in queries]
    relevant = [{q.event_id} for q in queries]

    pipeline = Pipeline(collection, [SceneSignal(collection, encoder)])
    summary = evaluate_matrix(pipeline.scores(texts), collection.vids, relevant, ks)

    print("\n=== Implementation correctness check (VATEX, scene signal) ===", flush=True)
    for name, value in summary.items():
        print(f"  {name:14}: {value}", flush=True)

    output = conf.path(cfg["paths"]["results"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"config": cfg, "collection": collection.size, "metrics": summary},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved results: {output}", flush=True)
    return summary


if __name__ == "__main__":
    main(*sys.argv[1:2])
