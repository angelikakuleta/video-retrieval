"""The relevance rule of chapter 4: annotation -> set of correct fragments.

An annotation marks the time interval of an event, not a ready-made fragment, so
for every segmentation strategy the answer set is derived: a fragment
``[f_start, f_end)`` is correct for an event when it contains the event's
midpoint, or when its intersection with the event covers at least half of the
FRAGMENT. The same annotation therefore yields different answer sets under
different segmentations, which is what experiment E1 compares.

Records sharing an ``event_id`` describe occurrences of one event and share the
union of their sets, so the event -- not the record -- stays the unit of
evaluation.
"""

from __future__ import annotations

from collections import defaultdict

from src.data.queries import episode_of
from src.utils.queries import Query


def fragment_id(episode: str, segment_id: int) -> str:
    """Identifier of a fragment in collections and rankings: ``s01e01_007``."""
    return f"{episode}_{segment_id:03d}"


def is_relevant(fragment: tuple[float, float], event: tuple[float, float]) -> bool:
    (f_start, f_end), (z_start, z_end) = fragment, event
    midpoint = (z_start + z_end) / 2
    if f_start <= midpoint < f_end:
        return True
    overlap = min(f_end, z_end) - max(f_start, z_start)
    return overlap >= (f_end - f_start) / 2


def relevance_sets(queries: list[Query], segments: list[dict]) -> dict[int, set[str]]:
    """``desc_id -> set of relevant fragment identifiers``.

    ``segments`` are the cache rows of ONE strategy, already filtered to the split. A
    query whose episode has no segments, or whose event matches no fragment, gets an
    empty set and the caller decides how to report it.
    """
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in segments:
        by_episode[row["episode"]].append(row)

    by_event: dict[object, set[str]] = defaultdict(set)
    for query in queries:
        episode = episode_of(query)
        event = (float(query.ts[0]), float(query.ts[1]))
        found = {
            fragment_id(row["episode"], row["segment_id"])
            for row in by_episode.get(episode, [])
            if is_relevant((row["start"], row["end"]), event)
        }
        by_event[query.event_id or ("desc", query.desc_id)] |= found

    return {q.desc_id: set(by_event[q.event_id or ("desc", q.desc_id)]) for q in queries}
