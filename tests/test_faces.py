"""Face buffer index resolution: a (time, face_index) pointer -> flat buffer index."""

import numpy as np

from src.features import faces


def fake_buffer(monkeypatch):
    # 3 sampled frames at t=0, 1, 2; frame 0 has 1 face, frame 1 has 3, frame 2 has 0
    buffer = {
        "times": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "frame": np.array([0, 1, 1, 1], dtype=np.int32),
    }
    monkeypatch.setattr(faces, "load_episode", lambda dataset, episode: buffer)
    return buffer


def test_resolve_index_picks_the_nearest_frame_and_position(monkeypatch):
    fake_buffer(monkeypatch)
    # closest to t=0.9 is frame 1 (t=1.0); its faces sit at buffer indices 1, 2, 3
    assert faces.resolve_index("office", "s02e03", 0.9, 0) == 1
    assert faces.resolve_index("office", "s02e03", 0.9, 2) == 3


def test_resolve_index_returns_none_when_the_frame_has_fewer_faces(monkeypatch):
    fake_buffer(monkeypatch)
    assert faces.resolve_index("office", "s02e03", 0.0, 1) is None   # frame 0 has 1 face
    assert faces.resolve_index("office", "s02e03", 2.0, 0) is None   # frame 2 has none
