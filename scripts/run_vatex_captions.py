"""Scene descriptions for the VATEX clips: the caption cache of one part.

    python scripts/run_vatex_captions.py --split test --dry-run
    python scripts/run_vatex_captions.py --split test --limit 20
    python scripts/run_vatex_captions.py --split test

``scripts/run_features.py`` skips VATEX, and rightly so: there are no masks, no
black stretches and no segmentation to share between components. The caption
cache needs none of that. One clip is one "episode" carrying a single span over
the whole file (``src/data/ranges.py::vatex_ranges``), so the per-episode
component loop applies unchanged -- this script runs that one component alone.

Resumable per clip: a clip whose ``.jsonl`` already exists is left untouched, so
an interrupted session continues where it stopped. Generation is the most
expensive step of the whole pipeline, so run ``--dry-run`` first: it prints the
estimate and, more importantly, checks that every clip resolves to a file that
exists.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import datasets
from src.data.ranges import load_ranges, split_episodes
from src.features import captions, encoders
from src.utils import gpu

#: per-frame cost of each generator on the RTX 5080, measured in
#: results/measurements/cost_20260907_085629.json; used for the estimate only
SECONDS_PER_FRAME = {"llava_1_5_7b": 1.30441, "blip": 0.02192}

#: grid step of the series runs, kept identical so the clips are sampled the
#: same way the episodes are
STEP = 1.25


def clip_episodes(split: str, subdir: str, limit: int | None) -> dict[str, dict]:
    """The clips of one part, with their video files placed under ``subdir``.

    ``src/data/ranges.py`` places the clips of each part under a subdirectory
    named after it. This script does not depend on that: the directory is an
    argument, so the cache can be built against whatever layout the material
    actually has.
    """
    episodes = split_episodes(load_ranges("vatex"), split)
    prefix = f"{subdir}/" if subdir else ""
    for clip, entry in episodes.items():
        entry["video_file"] = f"{prefix}{clip}.mp4"
    if limit is not None:
        episodes = {clip: episodes[clip] for clip in sorted(episodes)[:limit]}
    return episodes


def preflight(episodes: dict[str, dict], generator: str, split: str) -> list[str]:
    """Reports what the run would do; returns the clips with no video file."""
    missing = [clip for clip, entry in sorted(episodes.items())
               if not datasets.video_path("vatex", entry["video_file"]).exists()]
    todo = [clip for clip in episodes
            if not captions.episode_jsonl(generator, "vatex", clip).exists()]
    print(f"vatex / {split}: {len(episodes)} clips, "
          f"{len(episodes) - len(missing)} with a video file, {len(missing)} missing")
    print(f"cached already: {len(episodes) - len(todo)} clips; to generate: {len(todo)}")

    # a VATEX clip is ten seconds long and the grid starts at zero, so the
    # frame count matches src.segmentation.frames._sample_times
    per_clip = math.ceil(10.0 / STEP)
    seconds = SECONDS_PER_FRAME.get(generator)
    if todo and seconds:
        hours = len(todo) * per_clip * seconds / 3600
        print(f"estimated time: {hours:.1f} h "
              f"({generator}, {seconds:.3f} s per frame, ~{per_clip} frames per clip)")
    if missing:
        example = datasets.video_path("vatex", episodes[missing[0]]["video_file"])
        print(f"NO VIDEO FILE for {len(missing)} clips, for instance:\n  {example}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--generator", default="llava_1_5_7b",
                        choices=sorted(captions.GENERATORS))
    parser.add_argument("--encoder", default="openclip_vit_h14",
                        help="base representation the captions are encoded with")
    parser.add_argument("--clips-subdir", default=None,
                        help="directory under data/raw/vatex holding the clips "
                             "(default: the name of the split; pass an empty "
                             "string for clips lying flat)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N clips only - for a smoke test before the "
                             "whole part")
    parser.add_argument("--force", action="store_true",
                        help="recompute clips already cached")
    parser.add_argument("--dry-run", action="store_true",
                        help="report and check the material, generate nothing")
    parser.add_argument("--allow-missing", action="store_true",
                        help="start even though some clips have no video file")
    args = parser.parse_args()

    subdir = args.split if args.clips_subdir is None else args.clips_subdir
    episodes = clip_episodes(args.split, subdir, args.limit)
    if not episodes:
        raise SystemExit(f"no clip in split {args.split!r}")
    print(f"clips read from: {datasets.video_path('vatex', subdir) if subdir else 'data/raw/vatex'}")

    missing = preflight(episodes, args.generator, args.split)
    if args.dry_run:
        return
    if missing and not args.allow_missing:
        raise SystemExit(
            "stopping: clips with no video file would be skipped silently and the "
            "cache would come out incomplete. Point --clips-subdir at the right "
            "directory, or pass --allow-missing to caption the rest anyway.")

    gpu.free()
    captions.ensure_captions("vatex", episodes, args.generator, STEP, force=args.force)
    gpu.free()
    captions.ensure_caption_embeddings("vatex", episodes, args.generator, args.encoder,
                                       encoders.lazy(args.encoder), force=args.force)


if __name__ == "__main__":
    main()
