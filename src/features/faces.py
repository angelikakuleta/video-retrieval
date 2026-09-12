"""Face detection and alignment: the buffer three signals are built on.

RetinaFace-R50 returns the box and the five landmarks that let the crop be warped
onto a fixed template. That aligned crop is what the region signal embeds, what
HSEmotion classifies and what ArcFace turns into an identity vector -- detecting
once is what keeps those three comparable.

The exported ONNX takes a fixed 640x640 input, so a frame is scaled to fit and
padded with the model's mean colour, which after mean subtraction is zero. Boxes
and landmarks come back in the coordinates of the original frame, which is what
the minimum-face criterion is expressed in.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data import datasets
from src.features import episode_cache
from src.segmentation.frames import STEP, iter_batches

#: RetinaFace-R50 converted to ONNX; :func:`_conversion_hint` says how
MODEL = datasets.ROOT / "data" / "models" / "retinaface_r50.onnx"


def _conversion_hint(target) -> str:
    """How to produce the detector file, printed when it is missing.

    The weights are ~109 MB and their licence keeps them out of the repository,
    so the file is made once by hand and the steps live here rather than in a
    document an error message would have to point at.
    """
    return (
        "Convert it once (outside the repository):\n"
        "  git clone https://github.com/biubug6/Pytorch_Retinaface\n"
        "  cd Pytorch_Retinaface\n"
        "  # fetch Resnet50_Final.pth from the Model zoo section of its README\n"
        "  # (~109 MB) and put it in .\\weights\\\n"
        "  python convert_to_onnx.py --trained_model weights/Resnet50_Final.pth"
        " --network resnet50\n"
        f"  copy FaceDetector.onnx {target}")

#: input the ONNX was exported with
INPUT_SIZE = 640
#: BGR means of the detector
MEANS = (104.0, 117.0, 123.0)
#: anchor configuration of RetinaFace-R50
MIN_SIZES = ((16, 32), (64, 128), (256, 512))
STEPS = (8, 16, 32)
VARIANCE = (0.1, 0.2)

#: a detection below this confidence is not a face
SCORE_THRESHOLD = 0.6
#: overlap above which two boxes are the same face
NMS_THRESHOLD = 0.4
#: shortest side of the box worth cropping, in pixels of the frame THE PIPELINE
#: READS -- the normalized file under data/processed, not the master it was
#: ripped from; the two agree only while the ripping step keeps the resolution
MIN_FACE_PX = 32
#: side of the aligned crop, fixed by what ArcFace expects
CROP_SIZE = 112
#: frames handed to the detector at once
BATCH = 16


def cache_dir(dataset: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "faces" / "crops"
            / datasets.dataset_dir(dataset))


def episode_npz(dataset: str, episode: str) -> Path:
    return cache_dir(dataset) / f"{episode}.npz"


def load_episode(dataset: str, episode: str) -> dict:
    """``times`` plus one row per face: ``frame``, ``box``, ``score``, ``crop``."""
    with np.load(episode_npz(dataset, episode)) as data:
        return {key: data[key] for key in data.files}


def resolve_index(dataset: str, episode: str, time: float, face_index: int) -> int | None:
    """Flat buffer index for a ``(time, face_index)`` pointer, or ``None``.

    ``time`` picks the nearest sampled frame and ``face_index`` its position
    among that frame's faces -- the convention the profile file uses
    (:mod:`src.features.identity`, :mod:`src.annotation.profiles`). ``None``
    means the frame does not have that many faces.
    """
    buffer = load_episode(dataset, episode)
    times, frame_of = buffer["times"], buffer["frame"]
    frame = int(np.argmin(np.abs(times - time)))
    in_frame = np.flatnonzero(frame_of == frame)
    if face_index >= len(in_frame):
        return None
    return int(in_frame[face_index])


# --------------------------------------------- decoding the network's output
def priors(size: int = INPUT_SIZE) -> np.ndarray:
    """Anchor boxes of the three feature maps, in the network's own order."""
    boxes = []
    for k, step in enumerate(STEPS):
        side = math.ceil(size / step)
        for i in range(side):
            for j in range(side):
                for min_size in MIN_SIZES[k]:
                    boxes.append([(j + 0.5) * step / size, (i + 0.5) * step / size,
                                  min_size / size, min_size / size])
    return np.asarray(boxes, dtype=np.float32)


def decode_boxes(loc: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Anchor offsets -> boxes as (x1, y1, x2, y2), in units of the input."""
    centres = prior[:, :2] + loc[:, :2] * VARIANCE[0] * prior[:, 2:]
    sides = prior[:, 2:] * np.exp(loc[:, 2:] * VARIANCE[1])
    return np.concatenate([centres - sides / 2, centres + sides / 2], axis=1)


def decode_landmarks(pre: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Anchor offsets -> five landmarks per box, shaped (n, 5, 2)."""
    points = [prior[:, :2] + pre[:, i * 2:i * 2 + 2] * VARIANCE[0] * prior[:, 2:]
              for i in range(5)]
    return np.stack(points, axis=1)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray,
                        threshold: float = NMS_THRESHOLD) -> list[int]:
    """Indices of the boxes that survive, most confident first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        overlap = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = overlap / (areas[i] + areas[order[1:]] - overlap)
        order = order[1:][iou <= threshold]
    return keep


class FaceDetector:
    """RetinaFace-R50 over a fixed 640x640 input, with the letterbox undone."""

    def __init__(self, model: Path | str = MODEL,
                 score_threshold: float = SCORE_THRESHOLD,
                 min_face_px: int = MIN_FACE_PX) -> None:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            # ORT >= 1.21: loads the CUDA libraries from the pip nvidia-* packages
            ort.preload_dlls()
        model = Path(model)
        if not model.exists():
            raise FileNotFoundError(f"no {model}\n{_conversion_hint(model)}")
        self.session = ort.InferenceSession(
            str(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.priors = priors()
        self.score_threshold = score_threshold
        self.min_face_px = min_face_px

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """RGB frame -> the network's blob, and the scale that was applied."""
        import cv2

        height, width = image.shape[:2]
        scale = INPUT_SIZE / max(height, width)
        resized = cv2.resize(image, (max(1, round(width * scale)),
                                     max(1, round(height * scale))),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        canvas[...] = MEANS[::-1]                      # padding cancels out
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = canvas[:, :, ::-1] - np.asarray(MEANS, dtype=np.float32)   # to BGR
        return blob.transpose(2, 0, 1)[None], scale

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One RGB frame -> ``(boxes, landmarks, scores)`` in its own pixels."""
        blob, scale = self._letterbox(image)
        loc, conf, landm = self.session.run(None, {self.input: blob})
        scores = conf[0][:, 1]
        keep = scores > self.score_threshold
        if not keep.any():
            return (np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32),
                    np.zeros((0,), np.float32))

        boxes = decode_boxes(loc[0][keep], self.priors[keep]) * INPUT_SIZE / scale
        landmarks = decode_landmarks(landm[0][keep], self.priors[keep]) * INPUT_SIZE / scale
        scores = scores[keep]

        order = non_max_suppression(boxes, scores)
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        big = sides >= self.min_face_px
        return boxes[big], landmarks[big], scores[big]

    def crops(self, image: np.ndarray, landmarks: np.ndarray) -> list[np.ndarray]:
        """Aligned crops (RGB, ``CROP_SIZE`` square) for the given landmarks."""
        from insightface.utils import face_align

        bgr = image[:, :, ::-1]
        return [face_align.norm_crop(bgr, lm.astype(np.float32),
                                     image_size=CROP_SIZE)[:, :, ::-1].copy()
                for lm in landmarks]


def ensure_faces(
    dataset: str,
    episodes: dict[str, dict],
    step: float = STEP,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Detects and aligns the faces of every episode; returns the ones covered."""
    box: dict = {}

    def detector():
        if "detector" not in box:
            log(f"loading RetinaFace-R50 ({MODEL.name})")
            box["detector"] = FaceDetector()
        return box["detector"]

    def compute(episode: str, video: Path, target: Path) -> str:
        net = detector()
        times: list[float] = []
        frames, boxes, scores, crops = [], [], [], []
        for batch_times, batch_frames in iter_batches(video, step, BATCH):
            for offset, frame in enumerate(batch_frames):
                image = np.asarray(frame)
                found, landmarks, confidence = net.detect(image)
                if not len(found):
                    continue
                index = len(times) + offset
                for crop in net.crops(image, landmarks):
                    crops.append(crop)
                frames += [index] * len(found)
                boxes.append(found)
                scores.append(confidence)
            times += batch_times
        np.savez_compressed(
            target,
            times=np.asarray(times, dtype=np.float32),
            frame=np.asarray(frames, dtype=np.int32),
            box=(np.concatenate(boxes) if boxes
                 else np.zeros((0, 4), np.float32)).astype(np.float32),
            score=(np.concatenate(scores) if scores
                   else np.zeros((0,), np.float32)).astype(np.float32),
            crop=(np.stack(crops) if crops
                  else np.zeros((0, CROP_SIZE, CROP_SIZE, 3), np.uint8)).astype(np.uint8))
        with_face = len(set(frames))
        return (f"{len(times)} frames, {len(crops)} faces in {with_face} frames"
                f" ({100 * with_face / max(len(times), 1):.0f}%)")

    return episode_cache.ensure(dataset, episodes,
                                lambda ep: episode_npz(dataset, ep),
                                compute, "face detection and alignment",
                                force=force, log=log)
