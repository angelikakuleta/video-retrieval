"""Building the character profiles of chapter 4 -- shared by both series.

The identity signal (E6, :mod:`src.features.identity`) is generic across
datasets, but PICKING the eight examples of a profile is the same manual task
for TBBT and *The Office* alike, so the browsing and quality-control helpers
live here next to :mod:`intervals`/:mod:`tags`/:mod:`registry` rather than
being written twice.

The file a human fills in, ``data/annotations/<series>/<series>_profiles.csv``,
follows the same rule as ``<series>_query_tags.csv``: a notebook may create it
empty (:func:`ensure_skeleton`) but never overwrites a row once one exists. This
module only helps look (:func:`candidate_windows`, :func:`contact_sheet`),
review the picks (:func:`final_contact_sheet`) and check
(:func:`quality_report`); it never decides who is who.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from src.features import faces, identity


def path_for(series: str, directory: Path | str) -> Path:
    """Location of the profile file of one series."""
    return Path(directory) / f"{series}_profiles.csv"


def ensure_skeleton(path: Path | str) -> bool:
    """Writes just the header if the file does not exist yet.

    Never touches an existing file, even an empty one -- once it is there, its
    content is hand work. Returns whether a file was created.
    """
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=identity.COLUMNS, delimiter=";").writeheader()
    return True


# -------------- looking: where a character might be, and its candidate faces
def candidate_windows(register_rows: list[dict], character: str,
                      margin: float = 10.0) -> dict[str, list[tuple[float, float]]]:
    """``episode -> [(start, end), ...]`` where ``character``'s name is mentioned.

    Matches the first name as a whole word, case-insensitive, in the ``desc`` of
    each row -- the same rows a query's description is built from. Narrows a
    browsing pass to the seconds where the character is doing something, instead
    of scanning an entire episode's detected faces.
    """
    pattern = re.compile(rf"\b{re.escape(character)}\b", re.IGNORECASE)
    windows: dict[str, list[tuple[float, float]]] = {}
    for row in register_rows:
        if not pattern.search(row.get("desc") or ""):
            continue
        start, end = float(row["start"]), float(row["end"])
        windows.setdefault(row["episode"], []).append(
            (max(0.0, start - margin), end + margin))
    return windows


def contact_sheet(dataset: str, episode: str, windows: list[tuple[float, float]],
                  out_path: Path | str, columns: int = 6, limit: int = 48) -> dict | None:
    """Saves a labelled grid of the episode's face crops inside ``windows``.

    Each thumbnail is captioned ``seconds (mm:ss)  #face_index  score`` --
    seconds to copy into the profile file, the clock only to check against a
    player. ``face_index`` is a face's position within its own frame, the same
    number :func:`src.features.identity.build_profiles` expects there.
    Returns ``None`` and writes nothing when no detected face falls in any
    window, so a character absent from an episode leaves no stray empty image.
    Otherwise returns ``{"path", "matched", "shown"}``.

    A character mentioned often in one episode can pull in far more faces than
    anyone would look at (a talkative episode easily matches a thousand), which
    would also make the image itself unwieldy. Past ``limit`` matches, this
    keeps ``limit`` of them, evenly spaced across the matched ones by time --
    still one sheet per episode, not a random crop of its first minute.
    """
    import matplotlib.pyplot as plt

    buffer = faces.load_episode(dataset, episode)
    times, frame_of, scores, crops = (buffer["times"], buffer["frame"],
                                      buffer["score"], buffer["crop"])
    face_times = times[frame_of]
    in_window = np.zeros(len(face_times), dtype=bool)
    for start, end in windows:
        in_window |= (face_times >= start) & (face_times <= end)
    picked = np.flatnonzero(in_window)
    matched = len(picked)
    if not matched:
        return None
    if matched > limit:
        picked = picked[np.linspace(0, matched - 1, limit).round().astype(int)]

    # position of a face within its own frame -- what the profile file's
    # face_index column means
    face_index = np.zeros(len(frame_of), dtype=int)
    for frame in np.unique(frame_of):
        in_frame = np.flatnonzero(frame_of == frame)
        face_index[in_frame] = np.arange(len(in_frame))

    rows = -(-len(picked) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2.4 * rows),
                             squeeze=False)
    flat = axes.flat
    for ax in flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, picked):
        time = float(face_times[index])
        minutes, seconds = divmod(time, 60)
        ax.imshow(crops[index])
        ax.set_title(f"{time:.2f}s ({int(minutes)}:{seconds:04.1f})"
                     f"  #{face_index[index]}  {scores[index]:.2f}", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return {"path": out_path, "matched": matched, "shown": len(picked)}


def _scale_titles(fig, axes, fraction: float = 0.95, size: float = 8.0,
                  ceiling: float = 40.0) -> float:
    """Grows the captions until the widest one nearly spans its thumbnail.

    A caption left at a fixed point size reads as a thin line above a large
    crop; matched to the width of the image it sits on, it stays legible at
    whatever zoom the face is being looked at. Every title keeps the same size
    -- the largest one that still fits the widest caption inside ``fraction``
    of its own thumbnail -- so the sheet does not end up ragged.

    Measures the drawn text rather than counting characters, because the font
    that ends up rendering a caption depends on the machine. Re-runs
    ``tight_layout`` between passes: taller captions take room away from the
    images below them, which moves the width being matched, so a couple of
    passes settle it. Returns the size that was applied.
    """
    for _ in range(4):
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ratio = None
        for ax in axes:
            if not ax.title.get_text():
                continue
            box = (ax.images[0] if ax.images else ax).get_window_extent(renderer)
            width = ax.title.get_window_extent(renderer).width
            if width > 0:
                fitted = box.width * fraction / width
                ratio = fitted if ratio is None else min(ratio, fitted)
        if ratio is None or abs(ratio - 1.0) < 0.01:
            break
        size = min(size * ratio, ceiling)
        for ax in axes:
            ax.title.set_fontsize(size)
    fig.tight_layout()
    return size


def final_contact_sheet(dataset: str, character: str, rows: list[dict],
                        out_path: Path | str, columns: int = 4) -> Path | None:
    """Saves a small grid of exactly the examples chosen for one character.

    Unlike :func:`contact_sheet`, which browses many candidates to choose from,
    this renders the picks already written to the profile file -- one thumbnail
    per row, captioned with its note -- so a look at eight faces confirms they
    are the right ones before running the numeric check in
    :func:`quality_report`. Returns ``None`` and writes nothing if ``character``
    has no rows. A pointer that no longer resolves (a typo'd ``time`` or
    ``face_index``) gets a red placeholder instead of failing the whole sheet.
    """
    import matplotlib.pyplot as plt

    picks = [r for r in rows if r["character"] == character]
    if not picks:
        return None

    grid_rows = -(-len(picks) // columns)
    fig, axes = plt.subplots(grid_rows, columns, figsize=(2.4 * columns, 2.7 * grid_rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    captioned = []
    for ax, row in zip(axes.flat, picks):
        captioned.append(ax)
        time = float(row["time"])
        label = f"{row['episode']}  {time:.2f}s  #{row['face_index']}"
        if row.get("note"):
            label += f"\n{row['note']}"
        index = faces.resolve_index(dataset, row["episode"], time, int(row["face_index"]))
        if index is None:
            ax.set_title(f"{label}\nOUT OF RANGE", fontsize=8, color="red")
            continue
        ax.imshow(faces.load_episode(dataset, row["episode"])["crop"][index])
        ax.set_title(label, fontsize=8)

    _scale_titles(fig, captioned)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ------------------------------ checking: quality control of protocol step 4
def quality_report(dataset: str, rows: list[dict], dev_episodes: set[str]) -> dict:
    """Per-character margin plus per-example diagnostics (protocol step 4).

    ``margin`` is the same figure :func:`src.features.identity.margins` reports
    for the chapter 6 table: mean similarity among a character's own examples
    minus the best similarity to another character's. ``examples`` breaks that
    down per row, so a human can see WHICH example is dragging a margin down
    instead of only that it is low -- the protocol calls for rejecting and
    replacing that one example, not the whole profile.
    """
    outside = sorted({r["episode"] for r in rows} - dev_episodes)
    if outside:
        raise ValueError(
            f"profiles must be built on development episodes only; found {outside}")

    by_character: dict[str, list[dict]] = {}
    for row in rows:
        by_character.setdefault(row["character"], []).append(row)

    valid: dict[str, list[dict]] = {}
    matrices: dict[str, np.ndarray] = {}
    examples: list[dict] = []
    for character, character_rows in by_character.items():
        kept = []
        for row in character_rows:
            vector = identity.resolve_example(
                dataset, row["episode"], float(row["time"]), int(row["face_index"]))
            if vector is None:
                examples.append({**row, "own_similarity": None, "best_other": None,
                                 "problem": "face_index out of range"})
            else:
                kept.append({**row, "vector": vector})
        valid[character] = kept
        matrices[character] = (np.stack([e["vector"] for e in kept]).astype(np.float32)
                               if kept else np.zeros((0, 512), dtype=np.float32))

    for character, kept in valid.items():
        own = matrices[character]
        others = [m for label, m in matrices.items() if label != character and len(m)]
        for i, entry in enumerate(kept):
            similarity = own @ own[i]
            own_similarity = (float((similarity.sum() - 1.0) / (len(own) - 1))
                              if len(own) > 1 else float("nan"))
            best_other = max((float((m @ own[i]).max()) for m in others), default=0.0)
            examples.append({**{k: v for k, v in entry.items() if k != "vector"},
                             "own_similarity": own_similarity, "best_other": best_other,
                             "problem": ""})

    return {"margin": identity.margins(matrices), "examples": examples}
