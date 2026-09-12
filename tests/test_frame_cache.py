"""The frame-embedding cache loop, on the branch that actually computes.

Every other test of this repository reaches the cache when it is already full,
and that is the branch that returns before doing anything. The loop below it --
the encoder, the progress log, the per-episode write -- went unexercised, and a
missing import of :func:`src.utils.progress.progress` sat in it undetected: the
first run that had to compute embeddings died with ``NameError`` after black
detection and segmentation had already finished. The series never hit it,
because their embeddings predate the helper; VATEX was the first collection that
did, and every test split would have been the next.

No model is loaded here: the encoder arrives through ``encoder_factory`` and the
decoder through a stubbed ``iter_batches``, so what is under test is the loop
rather than OpenCLIP.
"""

import numpy as np
import pytest

from src.features import frame_cache

MODEL = "openclip_vit_h14"


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A dataset rooted at tmp_path, with an EMPTY embedding cache."""
    from src.data import datasets

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    (tmp_path / "data" / "processed" / "office").mkdir(parents=True)
    return {"s01e01": {"video_file": "office_s01e01.mp4"}}


class Encoder:
    """Stands in for ClipEncoder; counts what it was asked to encode."""

    def __init__(self):
        self.frames = 0

    def encode_images(self, frames):
        self.frames += len(frames)
        return np.zeros((len(frames), 4), dtype=np.float32)


def test_a_recording_that_is_not_there_is_named_and_skipped(tmp_path, dataset):
    """The branch the missing import was hiding in.

    It runs before the first episode is touched -- the progress log is built from
    the count of what is missing -- so an absent recording is enough to reach it,
    and the loop has to come back rather than raise.
    """
    said = []
    encoder = Encoder()

    frame_cache.ensure_frame_embeddings(
        "office", dataset, MODEL, step=1.25,
        encoder_factory=lambda: encoder, log=said.append)

    assert any("no video file" in line and "s01e01" in line for line in said)
    assert encoder.frames == 0                      # nothing was decoded
    assert not frame_cache.episode_npz(MODEL, "office", "s01e01").exists()


def test_the_missing_episode_is_computed_and_written(tmp_path, dataset,
                                                     monkeypatch):
    """The same loop with a recording present: it reaches the per-item log too.

    ``report(...)`` is the second call on the helper and the one the run makes
    once per episode, so a test that stops at the skip above would leave half of
    the repaired line uncovered.
    """
    from src.data import datasets
    from src.segmentation import frames as frames_module

    datasets.video_path("office", "office_s01e01.mp4").write_bytes(b"x")
    monkeypatch.setattr(frames_module, "iter_batches",
                        lambda video, step, batch: iter([([0.0, 1.25], ["a", "b"])]))

    said = []
    encoder = Encoder()
    frame_cache.ensure_frame_embeddings(
        "office", dataset, MODEL, step=1.25,
        encoder_factory=lambda: encoder, log=said.append)

    assert encoder.frames == 2
    times, embeddings = frame_cache.load_episode(MODEL, "office", "s01e01")
    assert times.tolist() == [0.0, 1.25]
    assert embeddings.shape == (2, 4)
    assert any("2 frames" in line for line in said)


def test_a_full_cache_loads_no_encoder_at_all(tmp_path, dataset):
    """The branch every other test reaches: nothing missing, nothing done."""
    target = frame_cache.episode_npz(MODEL, "office", "s01e01")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, times=np.zeros(1, np.float32),
                        embeddings=np.zeros((1, 4), np.float32))

    def refuse():
        raise AssertionError("the encoder was built although nothing was missing")

    said = []
    frame_cache.ensure_frame_embeddings("office", dataset, MODEL, step=1.25,
                                        encoder_factory=refuse, log=said.append)
    assert said == []
