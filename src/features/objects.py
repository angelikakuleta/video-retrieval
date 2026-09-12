"""Object detection on the frame grid: the object signal (E4).

YOLO11 works on the closed 80-class COCO vocabulary, YOLOE-11 prompt-free on a
built-in, much wider one. They share an architecture, so what E4 measures is
above all the width of the vocabulary.

A fragment stores the IDENTIFIER of every class detected, never a name and never
an embedding: the vocabulary is closed, so the names are encoded once
(:mod:`src.features.text_vocab`). Detections below 0.25 are dropped and a
fragment left without a label reports no value at all rather than a low one.

Detections cover the whole file, like the frame embeddings; which of them a
fragment may use is decided later, when the timeline is applied.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data import datasets
from src.features import episode_cache
from src.segmentation.frames import STEP, iter_batches

#: weights of each detector, resolved relative to the repository root.
#: `yoloe_prompted` is the QUERY-TIME variant (E4-D, section 08): different
#: weights from the prompt-free one, because the prompt-free checkpoint cannot
#: take text at all. It is named here so that every set of detector weights has
#: one home, and refused by `ensure_detections` below -- it detects per query and
#: fills no cache.
DETECTORS = {
    "yolo11": "yolo11l.pt",
    "yoloe_promptfree": "yoloe-11l-seg-pf.pt",
    "yoloe_prompted": "yoloe-11l-seg.pt",
}

#: detectors that build a per-episode cache ahead of any query
INDEXED = ("yolo11", "yoloe_promptfree")

#: the text encoder the promptable variant needs, 572 MB. Ultralytics fetches it
#: itself on the first `set_classes`; named here for the environment notes.
TEXT_ENCODER = "mobileclip_blt.ts"

#: detections below this confidence are dropped (the library's own default)
CONFIDENCE = 0.25

#: frames handed to the detector at once
BATCH = 16


def cache_dir(detector: str, dataset: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "objects" / detector
            / datasets.dataset_dir(dataset))


def episode_npz(detector: str, dataset: str, episode: str) -> Path:
    return cache_dir(detector, dataset) / f"{episode}.npz"


def vocabulary_json(detector: str) -> Path:
    return datasets.ROOT / "data" / "cache" / "objects" / detector / "vocab.json"


def load_episode(detector: str, dataset: str, episode: str) -> dict:
    """``times``, and one row per detection: ``frame``, ``class_id``, ``score``."""
    with np.load(episode_npz(detector, dataset, episode)) as data:
        return {key: data[key] for key in data.files}


def load_vocabulary(detector: str) -> list[str]:
    """Class names of the detector, in identifier order."""
    path = vocabulary_json(detector)
    if not path.exists():
        raise FileNotFoundError(
            f"no {path.name} for {detector!r} - run the extraction first")
    names = json.loads(path.read_text(encoding="utf-8"))["names"]
    return [names[str(i)] if isinstance(names, dict) else names[i]
            for i in range(len(names))]


def _model(detector: str):
    """Loads the detector; the two classes differ, the interface does not."""
    weights = str(datasets.ROOT / DETECTORS[detector])
    if detector == "yoloe_promptfree":
        from ultralytics import YOLOE

        return YOLOE(weights)
    from ultralytics import YOLO

    return YOLO(weights)


def ensure_detections(
    dataset: str,
    episodes: dict[str, dict],
    detector: str,
    step: float = STEP,
    confidence: float = CONFIDENCE,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Computes the missing per-episode detections; returns the episodes covered."""
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector: {detector!r}")
    if detector not in INDEXED:
        raise ValueError(
            f"{detector!r} detects at query time and fills no cache; the indexed "
            f"detectors are {list(INDEXED)} (section 08)")

    box: dict = {}

    def model():
        if "model" not in box:
            log(f"loading {detector} ({DETECTORS[detector]})")
            box["model"] = _model(detector)
        return box["model"]

    def compute(episode: str, video: Path, target: Path) -> str:
        net = model()
        times, frames, classes, scores = [], [], [], []
        names = None
        for batch_times, batch_frames in iter_batches(video, step, BATCH):
            # PIL images, never numpy arrays: the ultralytics loader converts a
            # PIL image to BGR itself, but takes a numpy array as ALREADY BGR, so
            # handing it np.asarray(frame) would feed the detector swapped channels
            results = net.predict(batch_frames, verbose=False, conf=confidence)
            for offset, result in enumerate(results):
                index = len(times) + offset
                names = names or result.names
                boxes = result.boxes
                if boxes is None:
                    continue
                for class_id, score in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    frames.append(index)
                    classes.append(int(class_id))
                    scores.append(float(score))
            times += batch_times
        np.savez_compressed(
            target,
            times=np.asarray(times, dtype=np.float32),
            frame=np.asarray(frames, dtype=np.int32),
            class_id=np.asarray(classes, dtype=np.int32),
            score=np.asarray(scores, dtype=np.float32))
        if names is not None:
            _save_vocabulary(detector, names)
        distinct = len(set(classes))
        return (f"{len(times)} frames, {len(classes)} detections,"
                f" {distinct} distinct classes")

    return episode_cache.ensure(dataset, episodes,
                                lambda ep: episode_npz(detector, dataset, ep),
                                compute, f"object detections ({detector})",
                                force=force, log=log)


def _save_vocabulary(detector: str, names) -> None:
    """Writes the class list of the detector, once, in identifier order."""
    path = vocabulary_json(detector)
    if path.exists():
        return
    ordered = {str(i): names[i] for i in sorted(names)} if isinstance(names, dict) \
        else {str(i): n for i, n in enumerate(names)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"detector": detector, "names": ordered},
                               ensure_ascii=False, indent=1), encoding="utf-8")
