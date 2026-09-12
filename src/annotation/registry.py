"""The collective register of annotations of one series -- one row per annotation.

Every annotation of every episode lands here BEFORE any filtering and keeps its
verdict: accepted, or rejected with the reason. Masks do not enter this table;
they live in the interval files and produce the corpus ranges
(:mod:`src.annotation.ranges`).

Written twice in the life of a dataset. First from the source descriptions, with
the ORIGINAL times and a verdict passed on length alone (``too_short``,
``too_long``); fitting the annotations to the masks belongs to the candidate
files and differs between their variants, so the register has no single value to
write for it. Then by the notebook reading the verified files back, where
:func:`update` records what the annotator did. Because a row identifier survives
that trip, the comparison is a plain set difference.

The ``tags`` column carries the labels put on an annotation in the annotator;
:func:`update` overwrites it from the file, which is its source of truth.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from src.annotation import intervals as iv
from src.utils import settings

COLUMNS = [
    "annotation_id", "desc_id", "episode", "split", "video_file",
    "vid_name", "ts_start", "ts_end",            # provenance (TVR); empty when there is none
    "source_start", "source_end",                # times written to the interval file
    "start", "end", "start_mmss", "end_mmss", "duration",   # current times
    "origin", "status", "reason",
    "edited_duration", "edited_desc",            # what a human changed, if anything
    "flags", "tags", "desc",
]

# The register holds the ORIGINAL times, so it says nothing about how an
# annotation was later fitted to the masks: the flags of that step travel with
# the candidates, in data/interim/<series>/work/. What is left here is the one
# warning about the times themselves.

#: the interval crosses a place where our recording differs from the source
FLAG_UNCERTAIN = "uncertain_mapping"

#: the annotation comes from an external source (TVR)
ORIGIN_EXTERNAL = "tvr"
#: the annotation was created in the annotator
ORIGIN_MANUAL = "manual"

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"

#: shorter than the frame sampling step -- it cannot land in a segment of its own
REASON_TOO_SHORT = "too_short"
#: longer than one segment is meant to hold
REASON_TOO_LONG = "too_long"
#: lies inside a mask (intro, closing credits), so it is outside the corpus
#: removed by hand in the annotator during verification
REASON_MANUAL = "manual"

#: every value the `reason` column may hold; empty means the row was accepted
REASONS = [REASON_TOO_SHORT, REASON_TOO_LONG, REASON_MANUAL]

YES, NO = "yes", "no"

#: the columns recording what a human changed in the annotator; ``yes`` or ``no``
EDITS = ["edited_duration", "edited_desc"]


def mmss(seconds: float) -> str:
    """Seconds -> ``minutes:seconds`` string, as shown by a player."""
    minutes = int(seconds) // 60
    return f"{minutes}:{seconds - 60 * minutes:04.1f}"


def row(annotation_id: str, episode: str, split: str, video_file: str,
        start: float, end: float, desc: str, *,
        desc_id: int | str = "", origin: str = ORIGIN_EXTERNAL,
        status: str = STATUS_ACCEPTED, reason: str = "",
        vid_name: str = "", ts_start: float | str = "", ts_end: float | str = "",
        flags: list[str] | None = None, tags: list[str] | None = None) -> dict:
    """Builds one register row. ``start``/``end`` are on the episode time axis."""
    return {
        "annotation_id": annotation_id,
        "desc_id": desc_id if desc_id != "" else iv.desc_id_of(annotation_id),
        "episode": episode, "split": split, "video_file": video_file,
        "vid_name": vid_name, "ts_start": ts_start, "ts_end": ts_end,
        # duration is derived from the ROUNDED bounds, so that `end - start` read
        # back from the file always reproduces it; computing it from the raw ones
        # made a row look re-lengthened when only the rounding differed
        "source_start": round(start, 2), "source_end": round(end, 2),
        "start": round(start, 2), "end": round(end, 2),
        "start_mmss": mmss(start), "end_mmss": mmss(end),
        "duration": round(round(end, 2) - round(start, 2), 2),
        "origin": origin, "status": status, "reason": reason,
        "edited_duration": NO, "edited_desc": NO,
        "flags": " ".join(flags or []),
        "tags": " ".join(tags or []), "desc": desc,
    }


def sort_rows(rows: list[dict]) -> list[dict]:
    """Register order: development part first, then test; inside each by episode,
    then by time, then by identifier.

    Episode identifiers (``s01e15``) sort as seasons and episodes on their own,
    so the only thing that has to be said explicitly is that ``dev`` comes first.
    """
    return sorted(rows, key=lambda r: (settings.split_order(r.get("split", "")),
                                       r["episode"], float(r["start"]),
                                       str(r["annotation_id"])))


# ----------------------------------------------------------------------- I/O

def save(rows: list[dict], path: Path | str) -> Path:
    """Writes the register (``;`` separator, dot decimal, UTF-8 with BOM)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=";",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sort_rows(rows))
    return path


def load(path: Path | str) -> list[dict]:
    """Reads the register back; numeric columns come back as floats."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    for item in rows:
        for column in ("source_start", "source_end", "start", "end", "duration",
                       "ts_start", "ts_end"):
            if item.get(column) not in (None, ""):
                item[column] = float(item[column])
    return rows


# ----------------------------------------------------------- accepted subset

def accepted(rows: list[dict], split: str | None = None) -> list[dict]:
    """Rows that enter the corpus, optionally narrowed to one split."""
    return [r for r in rows if r["status"] == STATUS_ACCEPTED
            and (split is None or r["split"] == split)]


# ------------------------------------- regenerating without losing hand work

# ----------------------------------------------- the diff after verification

def update(rows: list[dict], verified: dict[str, list[iv.Interval]],
           split: dict[str, str], video_file: dict[str, str],
           revalidate: Callable[[str, float, float], str] | None = None
           ) -> dict[str, dict]:
    """Brings the register in line with the hand-verified interval files.

    ``verified`` maps an episode to the intervals read back from its file; only
    those episodes are touched. What each kind of change does:

    ==============================  ==========================================
    in the annotator                in the register
    ==============================  ==========================================
    a rejected row was deleted      nothing -- the automatic reason stays
    an accepted row was deleted     ``rejected`` / ``manual``
    times moved, length unchanged   the new times, nothing flagged
    length changed                  the new times, ``edited_duration``
    wording rewritten               the new text, ``edited_desc``
    a new row appeared              added with ``origin`` ``manual``
    ==============================  ==========================================

    Given ``revalidate(episode, start, end)`` -- which returns the reason the
    times fail (``too_short``, ``too_long``) or ``""`` when they pass -- a row
    whose LENGTH changed is measured again and its verdict follows: one rejected
    on its length comes back once repaired (``recovered``), one stretched past
    the limit leaves (``demoted``). A pure shift needs no re-measuring, because
    the length that decides is the same.

    Returns ``{episode: {"kept", "removed", "moved", "edited_duration",
    "edited_desc", "recovered", "demoted", "added", "restored"}}``. Modifies
    ``rows`` in place.
    """
    by_id = {r["annotation_id"]: r for r in rows}
    summary = {}

    for episode, items in verified.items():
        present = {i.id: i for i in items if not i.is_mask}
        written = {r["annotation_id"] for r in rows
                   if r["episode"] == episode and r["status"] == STATUS_ACCEPTED}
        counts = {"kept": 0, "removed": 0, "moved": 0, "edited_duration": 0,
                  "edited_desc": 0, "recovered": 0, "demoted": 0, "added": 0,
                  "restored": 0}

        for annotation_id in written - set(present):
            entry = by_id[annotation_id]
            entry["status"] = STATUS_REJECTED
            entry["reason"] = REASON_MANUAL
            counts["removed"] += 1

        for annotation_id, item in present.items():
            entry = by_id.get(annotation_id)
            if entry is None:
                entry = row(annotation_id, episode, split.get(episode, ""),
                            video_file.get(episode, ""), item.start, item.end,
                            item.desc, origin=ORIGIN_MANUAL, tags=item.tags)
                rows.append(entry)
                by_id[annotation_id] = entry
                counts["added"] += 1
                continue

            was_rejected = entry["status"] == STATUS_REJECTED
            if was_rejected and entry["reason"] == REASON_MANUAL:
                entry["status"], entry["reason"] = STATUS_ACCEPTED, ""
                was_rejected = False
                counts["restored"] += 1

            moved = (round(float(entry["start"]), 2) != round(item.start, 2)
                     or round(float(entry["end"]), 2) != round(item.end, 2))
            relengthened = (round(float(entry["duration"]), 2)
                            != round(round(item.end, 2) - round(item.start, 2), 2))
            if moved:
                entry["start"], entry["end"] = round(item.start, 2), round(item.end, 2)
                entry["start_mmss"], entry["end_mmss"] = mmss(item.start), mmss(item.end)
                entry["duration"] = round(round(item.end, 2) - round(item.start, 2), 2)
                counts["moved"] += 1
            if relengthened:
                # only a change of LENGTH is an edit; sliding the whole interval
                # is ordinary annotator work and the verdict cannot change
                entry["edited_duration"] = YES
                counts["edited_duration"] += 1
                if revalidate is not None:
                    reason_now = revalidate(episode, item.start, item.end)
                    entry["status"] = (STATUS_REJECTED if reason_now
                                       else STATUS_ACCEPTED)
                    entry["reason"] = reason_now
                    if was_rejected and not reason_now:
                        counts["recovered"] += 1
                    elif reason_now and not was_rejected:
                        counts["demoted"] += 1

            if entry["desc"] != item.desc:
                entry["desc"] = item.desc
                entry["edited_desc"] = YES
                counts["edited_desc"] += 1

            # tags are hand work done in the annotator, so the file always wins;
            # they are not an edit of the annotation, hence outside the two above
            entry["tags"] = " ".join(item.tags)
            counts["kept"] += 1

        summary[episode] = counts
    return summary


# ---------------------------------------------------------------- statistics

def stats(rows: list[dict], episodes: list[str] | None = None) -> list[dict]:
    """Counts per episode: how many annotations there were and what happened.

    Columns ``episode``, ``split``, ``total``, ``accepted``, one per reason from
    :data:`REASONS` and one per column from :data:`EDITS` (rows marked ``yes``).
    """
    if episodes is None:
        episodes = sorted({r["episode"] for r in rows})
    out = []
    for episode in episodes:
        group = [r for r in rows if r["episode"] == episode]
        entry = {"episode": episode,
                 "split": group[0]["split"] if group else "",
                 "total": len(group),
                 "accepted": sum(1 for r in group if r["status"] == STATUS_ACCEPTED)}
        for reason in REASONS:
            entry[reason] = sum(1 for r in group if r["reason"] == reason)
        for column in EDITS:
            entry[column] = sum(1 for r in group if r.get(column) == YES)
        out.append(entry)
    return out


def totals(rows: list[dict], splits: tuple[str, ...] = ("dev", "test")) -> list[dict]:
    """The same counts summed over each split, plus a row for the whole set."""
    out = []
    for split in (*splits, ""):
        group = [r for r in rows if not split or r["split"] == split]
        entry = {"split": split or "total",
                 "episodes": len({r["episode"] for r in group}),
                 "total": len(group),
                 "accepted": sum(1 for r in group if r["status"] == STATUS_ACCEPTED)}
        for reason in REASONS:
            entry[reason] = sum(1 for r in group if r["reason"] == reason)
        for column in EDITS:
            entry[column] = sum(1 for r in group if r.get(column) == YES)
        out.append(entry)
    return out
