"""Stretches of (almost) black picture, found once per episode and cached.

These are NOT masks: they do not change what belongs to the corpus and do not
shorten content time. They say only that a frame taken from there carries no
information, so no model is fed one -- which is why the result lives under
``data/cache`` and may be deleted and recomputed at any time.

A frame counts as black when at least ``pic_th`` of its pixels are darker than
``pix_th`` of the luminance scale. The defaults (0.98, 0.10) are ffmpeg's own and
sit in a wide gap on this material: full black scores 100%, a fade halfway
through 19%, an ordinary interior 1%, and the atom animation of The Big Bang
Theory 5% -- which is why that animation needs a mask and is not caught here.

``min_duration`` is 0.04 s, just under one frame at 24 fps: ffmpeg's default of
2 s would miss the single black frames at a cut.

The cache holds one row per stretch and one row with empty times per episode
that was measured and has none, so a second run measures nothing twice.
"""

from __future__ import annotations

import csv
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from src.data import datasets
from src.utils import tools
from src.utils.progress import progress

#: share of pixels that must be dark for the frame to count as black
PIC_TH = 0.98
#: luminance below which a pixel counts as dark, as a share of the scale
PIX_TH = 0.10
#: shortest run reported [s] -- just under one frame at 24 fps
MIN_DURATION = 0.04

COLUMNS = ["episode", "start", "end", "duration"]
EPISODE = re.compile(r"s\d{2}e\d{2}", re.IGNORECASE)
REPORTED = re.compile(r"black_start:([0-9.]+) black_end:([0-9.]+)")


def episodes_of(dataset: str) -> dict[str, Path]:
    """``episode -> path of the .mp4``, from the processed recordings."""
    folder = datasets.ROOT / "data" / "processed" / datasets.dataset_dir(dataset)
    return {m.group(0).lower(): path
            for path in sorted(folder.glob("*.mp4"))
            if (m := EPISODE.search(path.name))}


def detect(video: Path, pix_th: float = PIX_TH, pic_th: float = PIC_TH,
           min_duration: float = MIN_DURATION) -> list[tuple[float, float]]:
    """One decoding pass -> the black stretches of the file, in seconds."""
    args = [tools.ffmpeg(), "-nostdin", "-hide_banner", "-nostats", "-i", str(video),
            "-vf", f"blackdetect=d={min_duration}:pix_th={pix_th}:pic_th={pic_th}",
            "-an", "-f", "null", "-"]
    finished = subprocess.run(args, capture_output=True, text=True)
    if finished.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video.name}:\n{finished.stderr[-400:]}")
    return [(float(a), float(b))
            for a, b in REPORTED.findall(finished.stdout + finished.stderr)]


def load(path: Path) -> dict[str, list[tuple[float, float]]]:
    """``episode -> stretches``; a measured episode with none is a key with ``[]``.

    A row with empty times is a marker, not a stretch: it says the episode WAS
    measured. Without it the file remembered only the episodes that happen to
    have black frames -- a few dozen of three thousand VATEX clips -- and every
    run re-measured the rest.
    """
    if not path.exists():
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            stretches = out.setdefault(r["episode"], [])
            if r["start"]:
                stretches.append((float(r["start"]), float(r["end"])))
    return out


def save(path: Path, found: dict[str, list[tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        for episode in sorted(found):
            if not found[episode]:
                # measured, nothing found - see the note in `load`
                writer.writerow({"episode": episode, "start": "", "end": "",
                                 "duration": ""})
                continue
            for start, end in sorted(found[episode]):
                writer.writerow({"episode": episode, "start": round(start, 3),
                                 "end": round(end, 3), "duration": round(end - start, 3)})


def ensure_black(
    dataset: str,
    episodes: list[str] | None = None,
    force: bool = False,
    pix_th: float = PIX_TH,
    pic_th: float = PIC_TH,
    min_duration: float = MIN_DURATION,
    videos: dict[str, Path] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, list[tuple[float, float]]]:
    """Guarantees the measurement for the requested episodes; returns all of it.

    ``episodes`` limits the work to what is needed: a development run must not decode
    the test recordings. An episode with no recording is skipped with a warning.

    ``videos`` names where each recording is. The runner passes it, because it
    already holds the ranges and knows; :func:`episodes_of` is the fallback for
    the standalone series scripts, and it finds nothing for VATEX -- its clips
    are not named ``sXXeYY`` and do not live under ``data/processed``.
    """
    path = datasets.black_csv(dataset)
    found = {} if force else load(path)
    # Existence, not just naming. `episodes_of` returns only the recordings it
    # FOUND, so the "no recording" guard below worked for the series; a `videos`
    # mapping is built from the ranges and names every episode whether its file
    # is on disk or not, which handed ffmpeg a path that is not there. VATEX
    # arrives in batches by design (section 15: no step may demand the full
    # set), so a named but absent clip has to be skipped, not fatal.
    available = (episodes_of(dataset) if videos is None else
                 {episode: path for episode, path in videos.items()
                  if Path(path).exists()})
    wanted = sorted(episodes) if episodes is not None else sorted(available)
    todo = [ep for ep in wanted if force or ep not in found]

    missing = [ep for ep in todo if ep not in available]
    for episode in missing:
        log(f"  {episode}: no recording - black stretches not measured")
    todo = [ep for ep in todo if ep in available]
    if not todo:
        return found

    log(f"{dataset}: black stretches for {len(todo)} episodes "
        f"(pix_th={pix_th}, pic_th={pic_th}, d={min_duration})")
    report = progress(log, total=len(todo))
    for episode in todo:
        started = time.perf_counter()
        found[episode] = detect(available[episode], pix_th, pic_th, min_duration)
        seconds = sum(b - a for a, b in found[episode])
        report(f"{episode}: {len(found[episode]):>3} stretches, {seconds:>6.1f} s black"
               f"  ({time.perf_counter() - started:.0f} s)")
        save(path, found)          # after every episode: the run is resumable
    log(f"cache -> {path.relative_to(datasets.ROOT)} "
        f"({sum(len(v) for v in found.values())} stretches in {len(found)} episodes)")
    return found
