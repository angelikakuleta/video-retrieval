"""Identity vectors of the detected faces and the character profiles (E6).

The signal is active only for queries naming a profiled character, and for
several names it is conjunctive.

A profile is not a set of pictures in the repository but a list of POINTERS to
faces the corpus already contains: character, episode, time, and which of the
faces detected in that frame. The example therefore goes through exactly the same
detection and alignment as everything else, and the file stays a few kilobytes of
text that can be versioned while the material itself stays out.

Chapter 4 profiles five characters per series, referred to by first name -- that
is what a query says and what :func:`src.utils.vocabulary.resolve_characters`
matches on.

Profiles may be built from development episodes only, and
:func:`build_profiles` enforces it rather than trusting the file.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data import datasets
from src.features import episode_cache, faces

#: ArcFace R100, from the antelopev2 package (see the setup instructions)
MODEL = Path.home() / ".insightface" / "models" / "antelopev2" / "glintr100.onnx"

#: how many examples one profile holds, per the protocol in chapter 4
PROFILE_SIZE = 8

#: columns of the profile file
COLUMNS = ["character", "episode", "time", "face_index", "note"]

#: crops encoded at once
BATCH = 64


def cache_dir(dataset: str) -> Path:
    return (datasets.ROOT / "data" / "cache" / "faces" / "arcface"
            / datasets.dataset_dir(dataset))


def episode_npz(dataset: str, episode: str) -> Path:
    return cache_dir(dataset) / f"{episode}.npz"


def profiles_csv(dataset: str) -> Path:
    name = datasets.dataset_dir(dataset)
    return datasets.ROOT / "data" / "annotations" / name / f"{name}_profiles.csv"


def profiles_npz(dataset: str) -> Path:
    name = datasets.dataset_dir(dataset)
    return datasets.ROOT / "data" / "cache" / "faces" / "profiles" / f"{name}.npz"


def load_episode(dataset: str, episode: str) -> np.ndarray:
    """Identity vectors (n_faces, dim) aligned with the crop buffer."""
    with np.load(episode_npz(dataset, episode)) as data:
        return data["embeddings"]


class ArcFace:
    """Identity encoder: aligned crop in, unit vector out."""

    def __init__(self, model: Path | str = MODEL) -> None:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        model = Path(model)
        if not model.exists():
            raise FileNotFoundError(
                f"no {model}\nDownload the antelopev2 package once:\n"
                "  python -c \"from insightface.app import FaceAnalysis;"
                " FaceAnalysis(name='antelopev2')\"")
        self.session = ort.InferenceSession(
            str(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def encode(self, crops) -> np.ndarray:
        """Encodes RGB crops (n, 112, 112, 3) into normalized vectors."""
        crops = np.asarray(crops, dtype=np.float32)
        if not len(crops):
            return np.zeros((0, 512), dtype=np.float32)
        blob = ((crops[:, :, :, ::-1] - 127.5) / 127.5).transpose(0, 3, 1, 2)
        out = []
        for start in range(0, len(blob), BATCH):
            chunk = np.ascontiguousarray(blob[start:start + BATCH])
            out.append(self.session.run(None, {self.input: chunk})[0])
        embeddings = np.concatenate(out).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms > 0, norms, 1.0)


def ensure_identity(
    dataset: str,
    episodes: dict[str, dict],
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Encodes the crops of every episode with ArcFace."""
    box: dict = {}

    def encoder():
        if "model" not in box:
            log(f"loading ArcFace ({MODEL.name})")
            box["model"] = ArcFace()
        return box["model"]

    def compute(episode: str, video: Path, target: Path) -> str:
        crops = faces.load_episode(dataset, episode)["crop"]
        embeddings = encoder().encode(crops)
        np.savez_compressed(target, embeddings=embeddings)
        return f"{len(embeddings)} faces, dim {embeddings.shape[1]}"

    with_faces = {ep: entry for ep, entry in episodes.items()
                  if faces.episode_npz(dataset, ep).exists()}
    for episode in sorted(set(episodes) - set(with_faces)):
        log(f"  {episode}: no face buffer - run the face detection first")
    return episode_cache.ensure(dataset, with_faces,
                                lambda ep: episode_npz(dataset, ep), compute,
                                "identity vectors (ArcFace)", force=force, log=log)


# ------------------------------------------------------------------ profiles
def load_profile_rows(dataset: str) -> list[dict]:
    """Rows of the profile file; an absent file means no profiles."""
    path = profiles_csv(dataset)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def resolve_example(dataset: str, episode: str, time: float,
                    face_index: int) -> np.ndarray | None:
    """The vector one profile-file row points at, or ``None`` if out of range.

    The same lookup :func:`build_profiles` does per row, factored out so a
    quality check can report on a single example instead of only a whole
    profile (:mod:`src.annotation.profiles`). Reloads the episode's buffers on
    every call rather than caching across rows -- a profile is a handful of
    examples per character, so the repeated I/O does not matter here.
    """
    index = faces.resolve_index(dataset, episode, time, face_index)
    if index is None:
        return None
    return load_episode(dataset, episode)[index]


def build_profiles(dataset: str, dev_episodes: set[str],
                   log: Callable[[str], None] = print) -> dict:
    """Turns the profile file into one matrix of examples per character.

    Returns ``{"characters", "margin"}``, where the margin is the mean similarity
    among a profile's own examples minus the highest similarity to another profile's
    -- the quality check of the protocol, reported in chapter 6.
    """
    rows = load_profile_rows(dataset)
    if not rows:
        return {"characters": {}, "margin": {}}

    outside = sorted({r["episode"] for r in rows} - dev_episodes)
    if outside:
        raise ValueError(
            f"profiles must be built on development episodes only; found {outside}")

    per_episode: dict[str, list[dict]] = {}
    for row in rows:
        per_episode.setdefault(row["episode"], []).append(row)

    characters: dict[str, list[np.ndarray]] = {}
    for episode, entries in sorted(per_episode.items()):
        buffer = faces.load_episode(dataset, episode)
        vectors = load_episode(dataset, episode)
        times, frame_of = buffer["times"], buffer["frame"]
        for entry in entries:
            wanted = float(entry["time"])
            frame = int(np.argmin(np.abs(times - wanted)))
            in_frame = np.flatnonzero(frame_of == frame)
            index = int(entry["face_index"])
            if index >= len(in_frame):
                log(f"  {entry['character']}: {episode} @ {wanted:.2f}s has"
                    f" {len(in_frame)} faces, index {index} out of range - skipped")
                continue
            characters.setdefault(entry["character"], []).append(
                vectors[in_frame[index]])

    matrices = {name: np.stack(items).astype(np.float32)
                for name, items in sorted(characters.items())}
    return {"characters": matrices, "margin": margins(matrices)}


def margins(characters: dict[str, np.ndarray]) -> dict[str, float]:
    """Per character: mean own similarity minus the best similarity to another."""
    out = {}
    for name, own in characters.items():
        if len(own) < 2:
            out[name] = float("nan")
            continue
        similarity = own @ own.T
        inside = (similarity.sum() - np.trace(similarity)) / (len(own) * (len(own) - 1))
        others = [other for label, other in characters.items() if label != name]
        outside = max((float((own @ other.T).max()) for other in others), default=0.0)
        out[name] = float(inside - outside)
    return out


def save_profiles(dataset: str, profiles: dict) -> Path:
    """Stores the built profiles next to the other face artifacts."""
    path = profiles_npz(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"profile:{name}": matrix
               for name, matrix in profiles["characters"].items()}
    np.savez_compressed(path, names=np.array(sorted(profiles["characters"])), **payload)
    return path


def load_profiles(dataset: str) -> dict[str, np.ndarray]:
    """``character -> (n_examples, dim)``; an absent file means no profiles."""
    path = profiles_npz(dataset)
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {name: data[f"profile:{name}"] for name in data["names"]}
