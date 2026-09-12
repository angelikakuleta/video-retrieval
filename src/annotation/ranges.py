"""Corpus ranges of an episode -- everything not covered by a mask.

The ranges are the complement of the masks inside ``[0, duration)``, read by
the segmentation step. The .mp4 files stay untouched. Masks come from the
verified ``<series>_<episode>_intervals.csv``, so the ranges follow what was
accepted by hand in the annotator, not automatic detection.

Ranges are half-open ``[start, end)``, so a range may touch a mask without
overlapping it.
"""

from __future__ import annotations

#: shortest range written to the ranges table [s] -- a shorter leftover between
#: two masks carries no usable material
from src.utils.settings import MIN_RANGE  # noqa: F401  (re-exported)

COLUMNS = ["episode", "split", "video_file", "range_id", "start", "end", "duration"]
HOLE_COLUMNS = ["episode", "split", "video_file", "hole_id", "start", "end", "duration"]


def merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merges overlapping or touching spans into a sorted, disjoint list."""
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def from_masks(masks: list[tuple[float, float]], duration: float,
               min_range: float = MIN_RANGE) -> list[tuple[float, float]]:
    """Ranges of an episode -> ``[(start, end)]``, in time order.

    Ranges shorter than ``min_range`` are dropped; with no mask at all the whole
    episode is a single range.
    """
    out, position = [], 0.0
    for start, end in merge(masks):
        if start > position:
            out.append((round(position, 3), round(min(start, duration), 3)))
        position = max(position, end)
        if position >= duration:
            break
    if position < duration:
        out.append((round(position, 3), round(duration, 3)))
    return [(a, b) for a, b in out if b - a >= min_range]


def rows(episode: str, split: str, video_file: str,
         spans: list[tuple[float, float]]) -> list[dict]:
    """Ranges of one episode as table rows; ``range_id`` numbers them from 1."""
    return [{"episode": episode, "split": split, "video_file": video_file,
             "range_id": range_id, "start": round(a, 3), "end": round(b, 3),
             "duration": round(b - a, 3)}
            for range_id, (a, b) in enumerate(spans, start=1)]


def overlap(start: float, end: float, spans: list[tuple[float, float]]) -> float:
    """How many seconds of ``[start, end)`` fall inside the ranges."""
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in spans)


def trim(start: float, end: float, spans: list[tuple[float, float]]):
    """Trims ``[start, end)`` to the range holding the largest part of it.

    Returns ``(start, end, remaining_seconds)``, or ``None`` when the interval lies
    entirely outside the ranges.
    """
    parts = [(max(start, a), min(end, b)) for a, b in spans
             if min(end, b) > max(start, a)]
    if not parts:
        return None
    a, b = max(parts, key=lambda p: p[1] - p[0])
    return round(a, 3), round(b, 3), b - a


def total(spans: list[tuple[float, float]]) -> float:
    """Total length of the ranges [s]."""
    return round(sum(b - a for a, b in spans), 3)


# ------------------------------------------------- the two classes of a mask

def split_masks(items) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Masks of one episode -> ``(structural, transitions)``, both sorted.

    The class comes from the tag, never from the length.
    """
    masks = [i for i in items if i.is_mask]
    structural = merge([(i.start, i.end) for i in masks if not i.is_transition])
    transitions = merge([(i.start, i.end) for i in masks if i.is_transition])
    return structural, transitions


def hole_rows(episode: str, split: str, video_file: str,
              spans: list[tuple[float, float]]) -> list[dict]:
    """Transition masks of one episode as rows of the holes table."""
    return [{"episode": episode, "split": split, "video_file": video_file,
             "hole_id": hole_id, "start": round(a, 3), "end": round(b, 3),
             "duration": round(b - a, 3)}
            for hole_id, (a, b) in enumerate(spans, start=1)]
