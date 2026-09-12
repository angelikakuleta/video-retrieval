"""Extracts the component features of the given configurations.

    python scripts/run_features.py configs/e3b_office.yaml --split dev
    python scripts/run_features.py configs/e5b_office.yaml configs/e5c_office.yaml --split dev
    python scripts/run_features.py configs/e4b_office.yaml --split dev --only objects

The expensive half of a run, split off so it can be done in its own session.
Configurations are grouped by component, so a model is loaded once no matter how
many ask for it, and the face detection is shared by the region, expression and
identity components.

Resumable per episode; only the episodes of the evaluated split are touched. The
experiment runner guarantees the same artifacts, so this is optional -- but a
full extraction takes hours and an evaluation over ready features takes minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ranges import load_ranges, load_timelines, split_episodes
from src.features import encoders
from src.segmentation import black
from src.segmentation.build import ensure_segments
from src.utils import gpu
from src.utils.config import load_experiment

#: components in the order they are extracted -- cheapest first, so an
#: interrupted session leaves the most behind
ORDER = ["objects", "faces", "regions", "expressions", "identity", "captions", "motion"]


def wanted_components(config, only: set[str] | None) -> set[str]:
    """Which components a configuration asks for, narrowed by --only."""
    components = config.components
    asked = set()
    if components.objects.enabled:
        asked.add("objects")
    if components.face_regions.enabled:
        asked.add("faces")
        asked.add("expressions" if components.face_regions.mode == "hsemotion"
                  else "regions")
    if components.identity.enabled:
        asked |= {"faces", "identity"}
    if components.caption.enabled:
        asked.add("captions")
    if components.motion.enabled:
        asked.add("motion")
    return asked if only is None else asked & only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="YAML configuration files")
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of: " + ", ".join(ORDER))
    parser.add_argument("--force", action="store_true",
                        help="recompute episodes already cached")
    args = parser.parse_args()

    missing = [f for f in args.configs if not Path(f).exists()]
    if missing:
        raise SystemExit("configuration file not found: " + ", ".join(missing))
    only = {name.strip() for name in args.only.split(",")} if args.only else None
    if only and (unknown := only - set(ORDER)):
        raise SystemExit(f"unknown component(s): {sorted(unknown)}")

    # group the work: (dataset, component, variant) -> the configurations asking
    plans: dict[tuple, list] = {}
    # VATEX is NOT skipped here. It was, and the sentence "VATEX has no
    # per-episode components" was true until a clip was made to look like an
    # episode (section 03): its motion buffer is keyed by (strategy, dataset)
    # exactly like a series one, with `whole_clip` for the strategy. With the
    # skip in place the only producer of that buffer was a full run, which
    # cannot happen before the threshold is frozen -- and the buffer is needed
    # before that.
    for file in args.configs:
        config = load_experiment(file, args.split)
        for component in wanted_components(config, only):
            plans.setdefault((config.dataset, component, _variant(config, component)),
                             []).append(config)

    if not plans:
        raise SystemExit("nothing to extract - the configurations enable no component")

    for dataset in sorted({key[0] for key in plans}):
        _extract(dataset, plans, args.split, args.force)


def _variant(config, component: str) -> str:
    """What distinguishes two caches of the same component."""
    components = config.components
    model = components.scene_embedding.model
    return {
        "objects": components.objects.detector,
        "faces": "",
        "regions": model,
        "expressions": "",
        "identity": "",
        "captions": f"{components.caption.model}/{model}",
        "motion": config.collection_strategy,
    }[component]


def _extract(dataset: str, plans: dict, split: str, force: bool) -> None:
    """Runs every planned extraction of one dataset, one component at a time."""
    from src.data import datasets
    from src.features import captions, expressions, faces, identity, motion, objects, regions

    ranges = load_ranges(dataset)
    episodes = split_episodes(ranges, split)
    if not episodes:
        print(f"{dataset}: no episode in split {split!r} - skipped")
        return
    names = sorted(episodes)
    print(f"\n=== {dataset} / {split}: {len(names)} episodes ===")

    # The recordings are named from the ranges, not looked up by name: a VATEX
    # clip is not called sXXeYY and does not live under data/processed, so the
    # pattern that finds the series episodes finds none of them and every clip
    # would quietly measure as having no black frames. Same mapping as stages.py.
    black.ensure_black(dataset, episodes=names,
                       videos={episode: datasets.video_path(
                           dataset, ranges[episode]["video_file"])
                           for episode in names})
    timelines = load_timelines(dataset, ranges)

    for component in ORDER:
        for (owner, kind, variant), configs in sorted(plans.items()):
            if owner != dataset or kind != component:
                continue
            config = configs[0]
            step = config.frames.step_s
            print(f"\n--- {component}{' ' + variant if variant else ''}"
                  f" ({len(configs)} configuration(s)) ---")
            gpu.free()            # one model at a time - 16 GB does not hold two
            if component == "objects":
                objects.ensure_detections(dataset, episodes, variant, step, force=force)
            elif component == "faces":
                faces.ensure_faces(dataset, episodes, step, force=force)
            elif component == "regions":
                regions.ensure_regions(dataset, episodes, variant,
                                       encoders.lazy(variant), force=force)
            elif component == "expressions":
                expressions.ensure_expressions(dataset, episodes, force=force)
            elif component == "identity":
                identity.ensure_identity(dataset, episodes, force=force)
            elif component == "captions":
                generator, encoder = variant.split("/")
                captions.ensure_captions(dataset, episodes, generator, step, force=force)
                captions.ensure_caption_embeddings(dataset, episodes, generator,
                                                   encoder, encoders.lazy(encoder),
                                                   force=force)
            elif component == "motion":
                rows = ensure_segments(dataset, {variant: config.segmentation},
                                       episodes=names)[variant]
                rows = [r for r in rows if r["split"] == split]
                motion.ensure_motion(dataset, episodes, rows, variant, timelines,
                                     force=force)


if __name__ == "__main__":
    main()
