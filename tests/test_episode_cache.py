"""The per-episode cache loop: skipping, resuming and never lying about progress."""

import numpy as np
import pytest

from src.features import episode_cache


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A dataset whose recordings and cache live under tmp_path."""
    from src.data import datasets

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    folder = tmp_path / "data" / "processed" / "office"
    folder.mkdir(parents=True)
    for episode in ("s01e01", "s01e02"):
        (folder / f"office_{episode}.mp4").write_bytes(b"x")
    return {"s01e01": {"video_file": "office_s01e01.mp4"},
            "s01e02": {"video_file": "office_s01e02.mp4"}}


def targets(tmp_path, suffix=".npz"):
    out = tmp_path / "cache"
    return lambda episode: out / f"{episode}{suffix}"


def test_every_episode_is_computed_once(tmp_path, dataset):
    seen = []

    def compute(episode, video, target):
        seen.append(episode)
        np.savez_compressed(target, a=np.zeros(3))
        return "ok"

    done = episode_cache.ensure("office", dataset, targets(tmp_path), compute,
                                "test", log=lambda *_: None)
    assert done == ["s01e01", "s01e02"]
    assert seen == ["s01e01", "s01e02"]


def test_a_finished_episode_is_skipped_on_the_next_run(tmp_path, dataset):
    seen = []

    def compute(episode, video, target):
        seen.append(episode)
        np.savez_compressed(target, a=np.zeros(3))
        return "ok"

    for _ in range(2):
        episode_cache.ensure("office", dataset, targets(tmp_path), compute,
                             "test", log=lambda *_: None)
    assert seen == ["s01e01", "s01e02"]      # the second run computed nothing


def test_an_interrupted_episode_leaves_no_file_that_looks_finished(tmp_path, dataset):
    def compute(episode, video, target):
        np.savez_compressed(target, a=np.zeros(3))
        if episode == "s01e02":
            raise KeyboardInterrupt("stopped halfway")
        return "ok"

    with pytest.raises(KeyboardInterrupt):
        episode_cache.ensure("office", dataset, targets(tmp_path), compute,
                             "test", log=lambda *_: None)

    target = targets(tmp_path)
    assert target("s01e01").exists()          # the finished one is kept
    assert not target("s01e02").exists()      # the interrupted one is not


def test_the_interrupted_episode_is_recomputed(tmp_path, dataset):
    attempts = []

    def flaky(episode, video, target):
        attempts.append(episode)
        np.savez_compressed(target, a=np.zeros(3))
        if episode == "s01e02" and attempts.count("s01e02") == 1:
            raise RuntimeError("first attempt fails")
        return "ok"

    with pytest.raises(RuntimeError):
        episode_cache.ensure("office", dataset, targets(tmp_path), flaky,
                             "test", log=lambda *_: None)
    episode_cache.ensure("office", dataset, targets(tmp_path), flaky,
                         "test", log=lambda *_: None)
    assert attempts == ["s01e01", "s01e02", "s01e02"]
    assert targets(tmp_path)("s01e02").exists()


def test_a_text_cache_is_renamed_the_same_way(tmp_path, dataset):
    def compute(episode, video, target):
        target.write_text("{}\n", encoding="utf-8")
        return "ok"

    episode_cache.ensure("office", dataset, targets(tmp_path, ".jsonl"), compute,
                         "test", log=lambda *_: None)
    assert targets(tmp_path, ".jsonl")("s01e01").read_text(encoding="utf-8") == "{}\n"


def test_an_episode_without_a_recording_is_skipped_not_fatal(tmp_path, dataset):
    dataset["s01e03"] = {"video_file": "office_s01e03.mp4"}      # never created
    messages = []

    def compute(episode, video, target):
        np.savez_compressed(target, a=np.zeros(3))
        return "ok"

    done = episode_cache.ensure("office", dataset, targets(tmp_path), compute,
                                "test", log=messages.append)
    assert done == ["s01e01", "s01e02"]
    assert any("no video file" in m for m in messages)


def test_force_recomputes_what_is_already_there(tmp_path, dataset):
    seen = []

    def compute(episode, video, target):
        seen.append(episode)
        np.savez_compressed(target, a=np.zeros(3))
        return "ok"

    episode_cache.ensure("office", dataset, targets(tmp_path), compute, "test",
                         log=lambda *_: None)
    episode_cache.ensure("office", dataset, targets(tmp_path), compute, "test",
                         force=True, log=lambda *_: None)
    assert seen == ["s01e01", "s01e02", "s01e01", "s01e02"]
