"""Distribution of detected face sizes at a LOWERED detection floor.

    python scripts/face_size_survey.py --split dev
    python scripts/face_size_survey.py --split test --floor 8

The production face buffer never holds a face below MIN_FACE_PX, so the share of
faces the threshold rejects cannot be read from it. This pass runs the same
detector on the same grid with the floor lowered and keeps ONLY boxes and scores
(no crops), per episode, resumable, outside the production cache:

    data/interim/<dataset>/work/face_sizes_<episode>.npz

Only frames on the content axis count (corpus ranges minus masks and black
stretches), which is what the signals see. The summary -- histogram of the
shorter box side, share rejected by every candidate threshold, upscale factor
to the ArcFace crop -- goes to results/measurements/face_sizes_<date>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import datasets
from src.data.ranges import load_ranges, load_timelines, split_episodes
from src.features.faces import BATCH, CROP_SIZE, MIN_FACE_PX, FaceDetector
from src.segmentation.frames import STEP, iter_batches
from src.utils.notebook import save_measurement

#: bucket edges of the shorter side, in pixels of the original frame
EDGES = [0, 16, 24, 32, 48, 64, 96, 128, 192, float("inf")]
#: thresholds whose rejection share is reported
CANDIDATES = (16, 24, 32, 40, 48)


def survey_file(dataset: str, episode: str) -> Path:
    return (ROOT / "data" / "interim" / datasets.dataset_dir(dataset) / "work"
            / f"face_sizes_{episode}.npz")


def survey_episode(dataset: str, episode: str, entry: dict, detector: FaceDetector,
                   timeline, force: bool = False) -> Path:
    """Boxes and scores of every content-axis grid frame of one episode."""
    target = survey_file(dataset, episode)
    if target.exists() and not force:
        return target
    video = datasets.video_path(dataset, entry["video_file"])
    if not video.exists():
        raise FileNotFoundError(video)
    started = time.perf_counter()
    times, frames, boxes, scores = [], [], [], []
    for batch_times, batch_frames in iter_batches(video, STEP, BATCH):
        usable = np.asarray(timeline.select(np.asarray(batch_times)), dtype=bool)
        for offset, (frame, ok) in enumerate(zip(batch_frames, usable)):
            if not ok:
                continue
            found, _, confidence = detector.detect(np.asarray(frame))
            index = len(times) + offset
            frames += [index] * len(found)
            boxes.append(found)
            scores.append(confidence)
        times += batch_times
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.stem}.partial{target.suffix}")
    np.savez_compressed(
        partial,
        times=np.asarray(times, dtype=np.float32),
        frame=np.asarray(frames, dtype=np.int32),
        box=(np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)),
        score=(np.concatenate(scores) if scores else np.zeros((0,), np.float32)),
        floor_px=np.int32(detector.min_face_px))
    partial.replace(target)
    print(f"  {episode}: {len(times)} frames, {len(frames)} faces"
          f"  ({time.perf_counter() - started:.0f} s)", flush=True)
    return target


def shorter_sides(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    sides, scores = [], []
    for file in files:
        with np.load(file) as data:
            box = data["box"]
            sides.append(np.minimum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]))
            scores.append(data["score"])
    return np.concatenate(sides), np.concatenate(scores)


def summarize(sides: np.ndarray, scores: np.ndarray) -> dict:
    counts, _ = np.histogram(sides, bins=EDGES)
    labels = [f"{int(a)}-{int(b)}" if np.isfinite(b) else f">={int(a)}"
              for a, b in zip(EDGES[:-1], EDGES[1:])]
    labels[0] = f"<{int(EDGES[1])}"
    total = max(len(sides), 1)
    bucket_score = [float(scores[(sides >= a) & (sides < b)].mean())
                    if ((sides >= a) & (sides < b)).any() else None
                    for a, b in zip(EDGES[:-1], EDGES[1:])]
    return {
        "faces": int(len(sides)),
        "buckets": [{"range": lab, "count": int(n), "share": round(100 * n / total, 2),
                     "mean_score": None if s is None else round(s, 3)}
                    for lab, n, s in zip(labels, counts, bucket_score)],
        "rejected_share": {str(t): round(100 * float((sides < t).mean()), 2)
                           for t in CANDIDATES},
        "kept_count": {str(t): int((sides >= t).sum()) for t in CANDIDATES},
        "median_side": round(float(np.median(sides)), 1) if len(sides) else None,
        "upscale_to_crop": {str(t): round(CROP_SIZE / t, 2) for t in CANDIDATES},
    }


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label}: {summary['faces']} faces, median shorter side "
          f"{summary['median_side']} px")
    print(f"  {'shorter side [px]':<20}{'faces':>8}{'share':>9}{'mean score':>12}")
    for b in summary["buckets"]:
        score = "-" if b["mean_score"] is None else f"{b['mean_score']:.3f}"
        print(f"  {b['range']:<20}{b['count']:>8}{b['share']:>8.1f}%{score:>12}")
    print(f"  {'threshold [px]':<20}{'rejected':>9}{'kept':>8}{'upscale':>9}")
    for t in CANDIDATES:
        print(f"  {t:<20}{summary['rejected_share'][str(t)]:>8.1f}%"
              f"{summary['kept_count'][str(t)]:>8}{summary['upscale_to_crop'][str(t)]:>8.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--datasets", nargs="+", default=["tbbt", "office"])
    parser.add_argument("--floor", type=int, default=8,
                        help="lowered detection floor in px (production: "
                             f"{MIN_FACE_PX})")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"loading RetinaFace-R50 with min_face_px={args.floor}", flush=True)
    detector = FaceDetector(min_face_px=args.floor)
    report = {"split": args.split, "floor_px": args.floor,
              "production_min_face_px": MIN_FACE_PX, "datasets": {}}
    for dataset in args.datasets:
        ranges = load_ranges(dataset)
        episodes = split_episodes(ranges, args.split)
        timelines = load_timelines(dataset, ranges)
        print(f"\n{dataset} / {args.split}: {len(episodes)} episodes", flush=True)
        files = [survey_episode(dataset, ep, entry, detector, timelines[ep], args.force)
                 for ep, entry in sorted(episodes.items())]
        sides, scores = shorter_sides(files)
        summary = summarize(sides, scores)
        summary["episodes"] = sorted(episodes)
        report["datasets"][dataset] = summary
        print_summary(dataset, summary)
    save_measurement("face_sizes", report)


if __name__ == "__main__":
    main()
