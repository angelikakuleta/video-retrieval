"""Segmentation strategies for experiment E1.

A segmentation splits the corpus ranges of an episode into fragments -- the
smallest unit of retrieval:

* ``fixed_window``     -- non-overlapping windows of ``WINDOW`` seconds,
* ``shots_histogram``  -- boundaries from the histogram difference of
  neighbouring frames (:mod:`src.segmentation.histogram`),
* ``shots_transnetv2`` -- boundaries from TransNetV2
  (:mod:`src.segmentation.transnet`).

Everything counts in content time (:mod:`src.segmentation.timeline`): a fixed
window always holds exactly ``WINDOW`` seconds of material even when the file
interval it covers is longer. The strategies differ only in where they cut; all
run inside the corpus ranges, the grid restarts per range, and boundaries
detected on the full file are clipped to the ranges, so no fragment crosses a
structural mask.

Every strategy is followed by the same length correction: fragments under
``MIN_LEN`` of content are merged with the shorter neighbour, fragments over
``MAX_LEN`` are split into equal parts. ``MIN_RANGE`` in
:mod:`src.annotation.ranges` equals ``MIN_LEN``, so a range is never shorter
than the minimum on its own.

Intervals are half-open ``[start, end)``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from src.segmentation.timeline import EPS, Timeline

from src.utils import settings

#: length correction and window, defined once in :mod:`src.utils.settings`
MIN_LEN = settings.MIN_LEN
MAX_LEN = settings.MAX_LEN
WINDOW = settings.WINDOW

STRATEGIES = ("fixed_window", "shots_histogram", "shots_transnetv2")

COLUMNS = ["episode", "split", "video_file", "segment_id",
           "start", "end", "duration", "file_duration"]


# --------------------------------------------------------- Length correction
def correct_lengths(
    spans: list[tuple[float, float]],
    timeline: Timeline,
    min_len: float = MIN_LEN,
    max_len: float = MAX_LEN,
) -> list[tuple[float, float]]:
    """Applies the length correction to the spans of ONE corpus range.

    Merge pass first: while a span shorter than ``min_len`` exists, the shortest is
    merged with the shorter neighbour. Then the split pass: a span longer than
    ``max_len`` is divided into the fewest parts of equal content length, each at
    least ``max_len / 2``, so the split never creates a too-short span.
    """
    spans = [(float(s), float(e)) for s, e in spans
             if timeline.content_length(s, e) > EPS]

    def length(span):
        return timeline.content_length(*span)

    while len(spans) > 1:
        i = min(range(len(spans)), key=lambda j: length(spans[j]))
        if length(spans[i]) >= min_len:
            break
        left = length(spans[i - 1]) if i > 0 else None
        right = length(spans[i + 1]) if i + 1 < len(spans) else None
        if right is None or (left is not None and left <= right):
            spans[i - 1:i + 1] = [(spans[i - 1][0], spans[i][1])]
        else:
            spans[i:i + 2] = [(spans[i][0], spans[i + 1][1])]

    out: list[tuple[float, float]] = []
    for start, end in spans:
        content = length((start, end))
        parts = max(1, math.ceil(content / max_len - 1e-9))
        step = content / parts
        for k in range(parts):
            a = start if k == 0 else timeline.advance(start, k * step)
            b = end if k == parts - 1 else timeline.advance(start, (k + 1) * step)
            out.append((a, b))
    return out


# ---------------------------------------------------------------- Strategies
def fixed_windows(
    timeline: Timeline,
    window: float = WINDOW,
    min_len: float = MIN_LEN,
    max_len: float = MAX_LEN,
) -> list[tuple[float, float]]:
    """Fixed windows inside every corpus range; the grid restarts per range.

    Cuts fall every ``window`` seconds of content. The last window of a range is
    shorter and, if it falls under ``min_len``, the correction merges it into the
    previous one.
    """
    out = []
    for index, (start, end) in enumerate(timeline.ranges):
        content = timeline.content_length(start, end)
        cuts = [timeline.to_file(index, k * window)
                for k in range(1, math.ceil(content / window))]
        spans = _spans_between(start, end, cuts)
        out += correct_lengths(spans, timeline, min_len, max_len)
    return out


def from_boundaries(
    boundaries: list[float],
    timeline: Timeline,
    min_len: float = MIN_LEN,
    max_len: float = MAX_LEN,
) -> list[tuple[float, float]]:
    """Shot boundaries (file-axis times) -> corrected fragments.

    Boundaries are clipped to every corpus range, the range edges act as implicit
    boundaries, and the correction runs per range. A boundary inside a transition
    mask is moved to the end of that mask instead of being dropped -- the detector
    fired on a real scene change, just in the middle of the animation announcing
    it.
    """
    out = []
    for start, end in timeline.ranges:
        cuts = sorted({_snap(timeline, b) for b in boundaries
                       if start + EPS < b < end - EPS})
        cuts = [c for c in cuts if start + EPS < c < end - EPS]
        spans = _spans_between(start, end, cuts)
        out += correct_lengths(spans, timeline, min_len, max_len)
    return out


def _snap(timeline: Timeline, boundary: float) -> float:
    """A boundary inside a transition mask -> the first moment after it."""
    for a, b in timeline.holes:
        if a <= boundary < b:
            return b
    return boundary


def _spans_between(start: float, end: float, cuts: list[float]) -> list[tuple[float, float]]:
    edges = [start, *cuts, end]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


# -------------------------------------------------- CSV of the segment cache
def rows(episode: str, split: str, video_file: str,
         spans: list[tuple[float, float]],
         timeline: Timeline) -> list[dict]:
    """Fragments of one episode -> rows of the cache CSV, numbered from 1.

    ``duration`` is the content length (what the 3-15 s correction and every table
    refer to), ``file_duration`` is ``end - start``; they differ exactly when a
    transition mask sits inside the fragment.
    """
    return [{"episode": episode, "split": split, "video_file": video_file,
             "segment_id": i, "start": round(a, 3), "end": round(b, 3),
             "duration": round(timeline.content_length(a, b), 3),
             "file_duration": round(b - a, 3)}
            for i, (a, b) in enumerate(sorted(spans), 1)]


def save(rows_: list[dict], path: Path | str) -> Path:
    """Writes the cache CSV (``;``, UTF-8 BOM), sorted by episode and time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows_, key=lambda r: (r["episode"], float(r["start"])))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(ordered)
    tmp.replace(path)
    return path


def load(path: Path | str) -> list[dict]:
    """Reads the cache CSV back; numeric fields become floats/ints.

    A cache written before the content axis existed has no ``file_duration``
    column; it is filled from ``end - start``, which is what it meant there.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        out = []
        for r in csv.DictReader(f, delimiter=";"):
            r["segment_id"] = int(r["segment_id"])
            for key in ("start", "end", "duration"):
                r[key] = float(r[key])
            r["file_duration"] = float(r.get("file_duration") or r["end"] - r["start"])
            out.append(r)
        return out
