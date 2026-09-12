"""Character-profile picking: candidate windows, the skeleton file, quality control."""

import numpy as np
import pytest

from src.annotation import profiles


def register_row(desc, episode="s01e01", start=10.0, end=15.0):
    return {"desc": desc, "episode": episode, "start": start, "end": end}


def profile_row(character, episode="s01e01", time=1.0, face_index=0, note=""):
    return {"character": character, "episode": episode, "time": str(time),
            "face_index": str(face_index), "note": note}


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# --------------------------------------------------------- candidate windows

def test_candidate_windows_matches_whole_word_case_insensitively():
    rows = [register_row("sheldon sits on his spot"),
            register_row("Leonard walks in"),
            register_row("Shelly is not sheldon at all")]
    windows = profiles.candidate_windows(rows, "Sheldon", margin=2.0)
    assert windows == {"s01e01": [(8.0, 17.0), (8.0, 17.0)]}


def test_candidate_windows_groups_by_episode():
    rows = [register_row("Sheldon", episode="s01e01", start=0.0, end=5.0),
            register_row("Sheldon", episode="s01e02", start=100.0, end=105.0)]
    windows = profiles.candidate_windows(rows, "Sheldon", margin=1.0)
    assert set(windows) == {"s01e01", "s01e02"}


def test_candidate_windows_margin_never_goes_negative():
    windows = profiles.candidate_windows(
        [register_row("Sheldon", start=1.0, end=2.0)], "Sheldon", margin=10.0)
    assert windows["s01e01"] == [(0.0, 12.0)]


# ------------------------------------------------------------------ skeleton

def test_ensure_skeleton_writes_header_once_and_never_again(tmp_path):
    path = tmp_path / "office_profiles.csv"
    assert profiles.ensure_skeleton(path) is True
    header = path.read_text(encoding="utf-8-sig").strip()
    assert header == ";".join(profiles.identity.COLUMNS)

    path.write_text(header + "\nMichael;s02e03;10.0;0;\n", encoding="utf-8-sig")
    assert profiles.ensure_skeleton(path) is False
    assert "Michael" in path.read_text(encoding="utf-8-sig")


# ------------------------------------------------------------ quality report

def test_quality_report_rejects_a_test_episode():
    with pytest.raises(ValueError, match="development episodes only"):
        profiles.quality_report("tbbt", [profile_row("Sheldon", episode="s09e12")],
                                {"s01e01"})


def test_quality_report_margin_and_own_similarity(monkeypatch):
    vectors = {
        ("s01e01", "1.0", 0): unit(1, 0),
        ("s01e01", "2.0", 0): unit(0.9, 0.1),
        ("s01e01", "3.0", 0): unit(0, 1),
        ("s01e01", "4.0", 0): unit(0.1, 0.9),
    }
    monkeypatch.setattr(
        profiles.identity, "resolve_example",
        lambda dataset, episode, time, face_index:
        vectors[(episode, f"{time:.1f}", face_index)])

    rows = [profile_row("Sheldon", time=1.0), profile_row("Sheldon", time=2.0),
            profile_row("Leonard", time=3.0), profile_row("Leonard", time=4.0)]
    report = profiles.quality_report("tbbt", rows, {"s01e01"})

    assert report["margin"]["Sheldon"] > 0
    assert report["margin"]["Leonard"] > 0

    by_time = {(e["character"], e["time"]): e for e in report["examples"]}
    expected_own = float(vectors[("s01e01", "1.0", 0)] @ vectors[("s01e01", "2.0", 0)])
    expected_other = float(
        max(vectors[("s01e01", "1.0", 0)] @ vectors[("s01e01", "3.0", 0)],
            vectors[("s01e01", "1.0", 0)] @ vectors[("s01e01", "4.0", 0)]))
    entry = by_time[("Sheldon", "1.0")]
    assert entry["own_similarity"] == pytest.approx(expected_own)
    assert entry["best_other"] == pytest.approx(expected_other)
    assert entry["problem"] == ""


def test_quality_report_flags_an_out_of_range_pointer(monkeypatch):
    monkeypatch.setattr(profiles.identity, "resolve_example",
                        lambda dataset, episode, time, face_index: None)
    report = profiles.quality_report("tbbt", [profile_row("Sheldon")], {"s01e01"})
    assert report["examples"][0]["problem"] == "face_index out of range"
    assert report["examples"][0]["own_similarity"] is None
    assert np.isnan(report["margin"]["Sheldon"])
