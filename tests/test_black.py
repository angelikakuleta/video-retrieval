"""The black-stretch cache: what it remembers and what it makes ffmpeg repeat."""

import pytest

from src.segmentation import black


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A repository root under tmp_path, so the cache file lands there."""
    from src.data import datasets

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    return tmp_path


def clip(root, name: str = "clip"):
    video = root / f"{name}.mp4"
    video.write_bytes(b"x")
    return video


# ------------------------------------------------------------------ the file

def test_an_episode_with_no_stretches_survives_the_round_trip(tmp_path):
    path = tmp_path / "black.csv"
    black.save(path, {"a": [(1.0, 1.5)], "b": []})
    assert black.load(path) == {"a": [(1.0, 1.5)], "b": []}


def test_a_file_written_before_the_markers_still_loads(tmp_path):
    """Only the episodes with stretches were recorded then; that must still read."""
    path = tmp_path / "black.csv"
    path.write_text("episode;start;end;duration\na;1.0;1.5;0.5\n", encoding="utf-8-sig")
    assert black.load(path) == {"a": [(1.0, 1.5)]}


def test_a_marker_row_is_not_an_interval_for_the_timeline(root):
    """The timeline reads the same file through its own loader."""
    from src.data import datasets, ranges

    black.save(datasets.black_csv("vatex"), {"a": [(1.0, 1.5)], "b": []})
    assert ranges.load_black("vatex") == {"a": [(1.0, 1.5)], "b": []}


# ------------------------------------------------------------- the measuring

def test_an_episode_measured_once_is_not_measured_again(root, monkeypatch):
    """The whole point: a clip with no black frames costs one ffmpeg pass, ever.

    Without the marker row the cache held only the episodes that happen to have
    black frames, so every run paid for the rest again -- 2904 of 3023 VATEX
    clips.
    """
    video = clip(root)
    measured = []

    def detect(path, *args, **kwargs):
        measured.append(path)
        return []

    monkeypatch.setattr(black, "detect", detect)
    for _ in range(2):
        found = black.ensure_black("vatex", episodes=["clip"],
                                   videos={"clip": video}, log=lambda *_: None)
    assert measured == [video]
    assert found == {"clip": []}


def test_force_measures_again_even_with_a_marker(root, monkeypatch):
    video = clip(root)
    measured = []

    def detect(path, *args, **kwargs):
        measured.append(path)
        return []

    monkeypatch.setattr(black, "detect", detect)
    for force in (False, True):
        black.ensure_black("vatex", episodes=["clip"], videos={"clip": video},
                           force=force, log=lambda *_: None)
    assert measured == [video, video]
