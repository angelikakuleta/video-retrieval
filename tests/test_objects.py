"""Object detection extraction: what the detector is actually handed.

The channel order is the whole point here. The ultralytics loader converts a PIL
image to BGR itself, but treats a numpy array as ALREADY BGR, so passing
``np.asarray(frame)`` silently swaps red and blue for every detection cached.
"""

import numpy as np
import pytest
from PIL import Image

from src.features import objects


class FakeResult:
    names = {0: "person"}
    boxes = None


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """One episode whose recording and cache live under tmp_path."""
    from src.data import datasets

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    folder = tmp_path / "data" / "processed" / "office"
    folder.mkdir(parents=True)
    (folder / "office_s01e01.mp4").write_bytes(b"x")
    return {"s01e01": {"video_file": "office_s01e01.mp4"}}


@pytest.fixture
def detector(monkeypatch):
    """A detector that records every frame it was given; returns that record."""
    seen: list = []

    class FakeModel:
        def predict(self, frames, **_):
            seen.extend(frames)
            return [FakeResult() for _ in frames]

    monkeypatch.setattr(objects, "_model", lambda _: FakeModel())
    return seen


@pytest.fixture
def grid(monkeypatch):
    """Two grid frames of a known, asymmetric colour."""
    frames = [Image.new("RGB", (8, 8), (200, 30, 10)),
              Image.new("RGB", (8, 8), (10, 30, 200))]

    def batches(video, step, batch):
        yield [0.0, 1.25], frames

    monkeypatch.setattr(objects, "iter_batches", batches)
    return frames


def test_the_detector_is_given_pil_images_not_arrays(dataset, detector, grid):
    objects.ensure_detections("office", dataset, "yolo11", log=lambda *_: None)

    assert detector, "the detector was never called"
    assert all(isinstance(frame, Image.Image) for frame in detector), \
        "a numpy array would be read as BGR and swap the channels"


def test_the_frames_reach_the_detector_untouched(dataset, detector, grid):
    objects.ensure_detections("office", dataset, "yolo11", log=lambda *_: None)

    assert len(detector) == len(grid)
    assert all(got is given for got, given in zip(detector, grid))
    assert np.asarray(detector[0])[0, 0].tolist() == [200, 30, 10]   # still RGB
