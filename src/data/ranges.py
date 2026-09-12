"""Reading the corpus ranges of a dataset, and the timeline built from them.

For the series the ranges are annotated and read from ``<dataset>_ranges.csv``.
VATEX has no such file and needs none: a clip IS its own range, so the rows are
synthesized from the query files -- one clip, one span, one fragment. Writing
them out would be a second copy of what the query file already says, and a
second thing to keep in step with it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data import datasets
from src.segmentation.timeline import Timeline

#: subdirectory of data/raw/vatex holding the clips of each part.
#: Section 03 of the plan recorded the test clips as lying FLAT in
#: data/raw/vatex/ and argued against moving 2560 files; they have since been
#: moved into test/ anyway, and scripts/run_vatex_captions.py already defaults
#: its --clips-subdir to the name of the split. The code follows the disk: with
#: the empty prefix run_features.py finds none of the 2560 and skips every one.
VATEX_SUBDIR = {"dev": "dev/", "test": "test/"}


def vatex_range_files() -> list[Path]:
    """The query files the VATEX ranges are read from, those that exist.

    They stand in for ``<dataset>_ranges.csv`` in the run's input manifest: they
    are what the collection actually depends on.
    """
    return [path for path in
            (datasets.queries_jsonl("vatex", split) for split in datasets.SPLITS)
            if path.exists()]


def vatex_ranges() -> dict[str, dict]:
    """One row per clip, built from whichever query files are on disk.

    A clip is an episode with a single span covering the whole file, so
    everything downstream -- segmentation, timelines, caches -- keeps working
    against the same shape the series produce. Standard library only.
    """
    episodes: dict[str, dict] = {}
    for split in datasets.SPLITS:
        path = datasets.queries_jsonl("vatex", split)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                clip = record["vid_name"]
                start, end = (float(t) for t in record["ts"])
                episodes[clip] = {
                    "split": split,
                    "video_file": f"{VATEX_SUBDIR[split]}{clip}.mp4",
                    "spans": [(start, end)],
                }
    return episodes


def load_ranges(dataset: str) -> dict[str, dict]:
    """``episode -> {"split", "video_file", "spans": [(start, end), ...]}``.

    Episodes appear in the file only once their masks are annotated, so the
    result may cover a subset of the selection; callers decide whether that
    is an error or something to skip.
    """
    if dataset == "vatex":
        return vatex_ranges()
    path = datasets.ranges_csv(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"no {path.name} - run the annotation notebooks first ({path})")
    episodes: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            entry = episodes.setdefault(
                r["episode"],
                {"split": r["split"], "video_file": r["video_file"], "spans": []})
            entry["spans"].append((float(r["start"]), float(r["end"])))
    return episodes


def _load_spans(path: Path) -> dict[str, list[tuple[float, float]]]:
    """``episode -> [(start, end), ...]`` from any two-column interval file.

    A missing file is not an error: a dataset with no transition masks and a
    corpus for which ``blackdetect`` has not been run yet are both legitimate
    states, and both mean "no intervals".

    A row with empty times is a marker rather than an interval -- the black cache
    writes one for every episode it measured and found nothing in, so that the
    next run knows not to measure it again (see :mod:`src.segmentation.black`).
    """
    if not path.exists():
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            spans = out.setdefault(r["episode"], [])
            if r["start"]:
                spans.append((float(r["start"]), float(r["end"])))
    return out


def load_holes(dataset: str) -> dict[str, list[tuple[float, float]]]:
    """Transition masks per episode -- the seconds that do not count."""
    return _load_spans(datasets.holes_csv(dataset))


def load_black(dataset: str) -> dict[str, list[tuple[float, float]]]:
    """Black stretches per episode -- skipped when sampling frames."""
    return _load_spans(datasets.black_csv(dataset))


def load_timelines(dataset: str, ranges: dict[str, dict] | None = None) -> dict[str, Timeline]:
    """``episode -> Timeline``, built once and passed down.

    Reading the three files together is what guarantees that segmentation and
    every component measure the same axis: nothing below this function ever
    reconstructs it from masks on its own.
    """
    ranges = ranges if ranges is not None else load_ranges(dataset)
    holes, black = load_holes(dataset), load_black(dataset)
    return {ep: Timeline(entry["spans"], holes.get(ep, []), black.get(ep, []))
            for ep, entry in ranges.items()}


def split_episodes(episodes: dict[str, dict], split: str) -> dict[str, dict]:
    """Filters the ranges down to one split (``dev``/``test``)."""
    return {ep: e for ep, e in episodes.items() if e["split"] == split}
