"""Action recognition: the classical motion component of experiment E2.

SlowFast returns a distribution over the 400 Kinetics-400 actions and the WHOLE
of it is stored. Keeping only the K most likely classes looked cheap and was not:
a query phrase matched to one of the 400 names was usually absent from every
stored top-K, so the signal came out formally active and numerically empty. At
float16 the full row costs 800 bytes per fragment.

Windows follow the 8x8 training configuration: 64 source frames, the fast pathway
taking every second and the slow one every eighth. A longer fragment is covered
by consecutive windows whose distributions are averaged. The window is measured
on the CONTENT axis, exactly like the X-CLIP one, so masked and black frames
never enter it.

Windows are laid out over fragments, so unlike the frame-level components this
cache depends on the segmentation strategy.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data import datasets
from src.features import episode_cache
from src.segmentation.timeline import Timeline

#: source frames per window, and how each pathway samples them (the 8x8 variant)
WINDOW_FRAMES = 64
SLOW_FRAMES = 8
FAST_FRAMES = 32

#: the Kinetics-400 vocabulary; the buffer stores one probability per class
CLASSES = 400

#: preprocessing of the Kinetics protocol
SHORT_SIDE = 256
CROP = 224
MEAN = np.array((0.45, 0.45, 0.45), dtype=np.float32)
STD = np.array((0.225, 0.225, 0.225), dtype=np.float32)

#: windows pushed through the model at once
BATCH = 8

#: the class list, in the alphabetical order the pretrained head was trained with
CLASSES_CSV = datasets.ROOT / "data" / "interim" / "vatex" / "kinetics-400_train.csv"


def cache_dir(strategy: str, dataset: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "motion" / strategy
            / datasets.dataset_dir(dataset))


def episode_npz(strategy: str, dataset: str, episode: str) -> Path:
    return cache_dir(strategy, dataset) / f"{episode}.npz"


def load_episode(strategy: str, dataset: str, episode: str) -> dict:
    """``segment_id`` and ``probability`` of shape (n, 400).

    A file in the old top-K layout is refused rather than read: its rows carry
    ten classes out of four hundred, so the value of the matched class would
    silently come out zero for most fragments.
    """
    path = episode_npz(strategy, dataset, episode)
    with np.load(path) as data:
        loaded = {key: data[key] for key in data.files}
    probability = loaded.get("probability")
    if "class_id" in loaded or probability is None or probability.shape[1:] != (CLASSES,):
        raise ValueError(
            f"{path}: motion buffer in the old top-K layout"
            f" (shape {None if probability is None else probability.shape});"
            f" expected the full distribution (n, {CLASSES}). Delete the buffer"
            " and extract it again.")
    return loaded


def class_names(path: Path = CLASSES_CSV) -> list[str]:
    """The 400 Kinetics classes, sorted -- the order of the pretrained head."""
    if not path.exists():
        raise FileNotFoundError(
            f"no {path} - the Kinetics-400 class list comes with the VATEX data")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        labels = {row["label"] for row in csv.DictReader(handle)}
    return sorted(labels)


def window_starts(start: float, end: float, timeline: Timeline | None,
                  fps: float) -> list[tuple[float, float]]:
    """Consecutive windows covering one fragment, on the content axis.

    Tiling comes from :func:`src.features.xclip.window_offsets`, so both models
    see the same stretches of a fragment -- which is what makes the E2
    comparison about how motion is represented and not about what was shown.
    """
    from src.features.xclip import window_offsets

    span = WINDOW_FRAMES / fps
    if timeline is None:
        return [(start + o, min(start + o + span, end))
                for o in window_offsets(end - start, span, fps)]
    content = timeline.content_length(start, end)
    out = []
    for offset in window_offsets(content, span, fps):
        a = timeline.advance(start, min(offset, content))
        b = timeline.advance(start, min(offset + span, content))
        out.append((a, max(b, a + 1.0 / fps)))
    return out


def window_times(a: float, b: float, timeline: Timeline | None,
                 count: int) -> list[float]:
    """``count`` moments spread over one window, skipping what is masked out."""
    if timeline is None:
        return list(np.linspace(a, b, count, endpoint=False))
    return timeline.sample_content(a, b, count)


class SlowFast:
    """SlowFast R50 8x8 with the Kinetics preprocessing of its own protocol."""

    def __init__(self) -> None:
        import torch
        from pytorchvideo.models.hub import slowfast_r50

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = slowfast_r50(pretrained=True).eval().to(self.device)

    def _prepare(self, frames: list[np.ndarray]) -> np.ndarray:
        """RGB frames -> (T, 3, CROP, CROP) float32, normalized."""
        import cv2

        prepared = []
        for frame in frames:
            height, width = frame.shape[:2]
            scale = SHORT_SIDE / min(height, width)
            resized = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                 interpolation=cv2.INTER_LINEAR)
            top = (resized.shape[0] - CROP) // 2
            left = (resized.shape[1] - CROP) // 2
            patch = resized[top:top + CROP, left:left + CROP].astype(np.float32) / 255.0
            prepared.append(((patch - MEAN) / STD).transpose(2, 0, 1))
        return np.stack(prepared)

    def predict(self, windows: list[list[np.ndarray]]) -> np.ndarray:
        """One probability distribution per window: (n_windows, 400)."""
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(windows), BATCH):
                chunk = windows[start:start + BATCH]
                fast = np.stack([self._prepare(w) for w in chunk])   # (B, T, 3, H, W)
                fast_t = torch.from_numpy(fast).permute(0, 2, 1, 3, 4).to(self.device)
                slow_t = fast_t[:, :, ::FAST_FRAMES // SLOW_FRAMES]
                logits = self.model([slow_t, fast_t])
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def ensure_motion(
    dataset: str,
    episodes: dict[str, dict],
    rows: list[dict],
    strategy: str,
    timelines: dict[str, Timeline] | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Classifies every fragment of every episode; returns the episodes covered."""
    from src.features import xclip

    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)

    box: dict = {}

    def model():
        if "model" not in box:
            log("loading SlowFast R50 (8x8)")
            box["model"] = SlowFast()
        return box["model"]

    def compute(episode: str, video: Path, target: Path) -> str:
        import cv2

        reader = cv2.VideoCapture(str(video))
        fps = reader.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        reader.release()
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"could not read the video: {video}")

        timeline = (timelines or {}).get(episode)
        fragments = sorted(by_episode.get(episode, []), key=lambda r: r["start"])
        spans, needed = [], set()
        for fragment in fragments:
            windows = []
            for a, b in window_starts(float(fragment["start"]), float(fragment["end"]),
                                      timeline, fps):
                indices = [min(int(round(t * fps)), frame_count - 1)
                           for t in window_times(a, b, timeline, FAST_FRAMES)]
                windows.append(indices)
                needed.update(indices)
            spans.append(windows)

        frames = xclip.collect_frames(video, sorted(needed), height=SHORT_SIDE)
        flat, bounds = [], []
        for windows in spans:
            bounds.append((len(flat), len(flat) + len(windows)))
            flat += [[frames[i] for i in window] for window in windows]
        distribution = model().predict(flat)

        ids, probabilities = [], []
        for fragment, (a, b) in zip(fragments, bounds):
            ids.append(int(fragment["segment_id"]))
            # float16 over the whole vocabulary: the signal reads the value of
            # whichever class the query matched, and any class may be that one
            probabilities.append(distribution[a:b].mean(axis=0).astype(np.float16))
        np.savez_compressed(
            target,
            segment_id=np.asarray(ids, dtype=np.int32),
            probability=(np.stack(probabilities) if probabilities
                         else np.zeros((0, CLASSES), np.float16)))
        return f"{len(ids)} fragments, {len(flat)} windows"

    wanted = {ep: entry for ep, entry in episodes.items() if ep in by_episode}
    return episode_cache.ensure(dataset, wanted,
                                lambda ep: episode_npz(strategy, dataset, ep),
                                compute, f"motion ({strategy})",
                                force=force, log=log)
