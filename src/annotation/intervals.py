"""Reading and writing the interval files of ``tools/interval-annotator``.

Format: ``id;type;start;end;desc`` -- semicolon separator, dot decimal, two
decimals, UTF-8 with BOM. ``type`` is ``event`` (annotation) or ``mask``
(stretch excluded from the corpus). Must stay compatible with ``js/csv.js`` in
the annotator: the same file travels in both directions.

Intervals are half-open ``[start, end)``.

Identifiers carry provenance, which is what makes it possible to tell later
which annotations were removed by hand::

    tbbt_s01e01_d97650   imported from TVR, 97650 is its desc_id
    office_s02e03_001    created in the annotator
    office_s02e03_m001   mask
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

#: value of the ``type`` column for an annotation
EVENT = "event"
#: value of the ``type`` column for a mask
MASK = "mask"

COLUMNS = ["id", "type", "start", "end", "desc", "tags"]

#: tag marking a mask absorbed by the content axis instead of cutting a range
TAG_TRANSITION = "przejscie"

#: identifier of an annotation imported from an external source
KIND_EXTERNAL = "external"
#: identifier minted by the annotator for a row created by hand
KIND_LOCAL = "local"
#: identifier of a mask
KIND_MASK = "mask"

#: largest desc_id in the TVR release is 98069; synthetic ones start above this
MAX_EXTERNAL_DESC_ID = 99_999

_ID = re.compile(r"^(?P<series>[a-z0-9]+)_(?P<episode>s(?P<season>\d{2})e(?P<number>\d{2}))"
                 r"_(?P<tail>m\d+|d\d+|\d+)$")
_EPISODE = re.compile(r"^s(\d{2})e(\d{2})$")


@dataclass
class Interval:
    """One row of an interval file."""

    id: str
    type: str          # EVENT or MASK
    start: float
    end: float
    desc: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def is_mask(self) -> bool:
        return self.type == MASK

    @property
    def is_transition(self) -> bool:
        """A mask absorbed by the content axis rather than cutting a range.

        Told apart by its tag, not by its length.
        """
        return self.is_mask and TAG_TRANSITION in self.tags


# --------------------------------------------------------------- identifiers

def stem(series: str, episode: str) -> str:
    """Common prefix of the identifiers of one episode, e.g. ``tbbt_s01e01``."""
    return f"{series}_{episode}"


def external_id(series: str, episode: str, desc_id: int | str) -> str:
    """Identifier of an annotation imported from an external source."""
    return f"{stem(series, episode)}_d{int(desc_id)}"


def parse_id(item_id: str) -> dict:
    """Splits an identifier into its parts.

    Returns ``{"series", "episode", "season", "number", "kind", "value"}``, where
    ``kind`` is :data:`KIND_EXTERNAL`, :data:`KIND_LOCAL` or :data:`KIND_MASK`.
    Raises ``ValueError`` on an identifier outside the convention.
    """
    match = _ID.match(item_id.strip())
    if not match:
        raise ValueError(f"identifier outside the convention: {item_id!r}")
    tail = match.group("tail")
    if tail.startswith("m"):
        kind, value = KIND_MASK, int(tail[1:])
    elif tail.startswith("d"):
        kind, value = KIND_EXTERNAL, int(tail[1:])
    else:
        kind, value = KIND_LOCAL, int(tail)
    return {"series": match.group("series"), "episode": match.group("episode"),
            "season": int(match.group("season")), "number": int(match.group("number")),
            "kind": kind, "value": value}


def synthetic_desc_id(episode: str, seq: int) -> int:
    """Query identifier for an annotation that has no external one.

    ``s01e01`` + 1 -> ``1010001``. Above :data:`MAX_EXTERNAL_DESC_ID`, so it never
    clashes with a desc_id taken from TVR.
    """
    match = _EPISODE.match(episode)
    if not match:
        raise ValueError(f"episode outside the convention: {episode!r}")
    season, number = (int(g) for g in match.groups())
    value = season * 1_000_000 + number * 10_000 + seq
    if value <= MAX_EXTERNAL_DESC_ID:
        raise ValueError(f"synthetic desc_id {value} falls into the external range")
    return value


def desc_id_of(item_id: str) -> int:
    """Query identifier of an annotation, whatever its provenance."""
    parts = parse_id(item_id)
    if parts["kind"] == KIND_MASK:
        raise ValueError(f"a mask has no desc_id: {item_id!r}")
    if parts["kind"] == KIND_EXTERNAL:
        return parts["value"]
    return synthetic_desc_id(parts["episode"], parts["value"])


# ----------------------------------------------------------------------- I/O

def read(path: Path | str) -> list[Interval]:
    """Reads an interval file. A row without a ``type`` column is an annotation."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    items = []
    for row in rows:
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        if not row.get("id") and not row.get("start"):
            continue
        kind = row.get("type", "").lower()
        items.append(Interval(
            id=row.get("id", ""),
            type=MASK if kind in (MASK, "m") else EVENT,
            start=round(float(row["start"].replace(",", ".")), 2),
            end=round(float(row["end"].replace(",", ".")), 2),
            desc=row.get("desc", ""),
            tags=parse_tags(row.get("tags", "")),
        ))
    return items


def parse_tags(value: str) -> list[str]:
    """``a,b`` -> ``['a', 'b']``; an empty or missing column -> ``[]``."""
    return [tag.strip() for tag in (value or "").split(",") if tag.strip()]


def write(path: Path | str, items: list[Interval]) -> Path:
    """Writes an interval file in the format the annotator imports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        # CRLF (the csv default) -- every CSV in this repository uses it
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(COLUMNS)
        for item in items:
            writer.writerow([item.id, item.type, f"{item.start:.2f}",
                             f"{item.end:.2f}", item.desc, ",".join(item.tags)])
    return path


def file_name(series: str, episode: str) -> str:
    """Name of the interval file of one episode."""
    return f"{stem(series, episode)}_intervals.csv"


def sort_key(item: Interval) -> tuple:
    """Order used in the written files: by start, the shorter interval first."""
    return (item.start, item.end)


# ---------------------------------------------------------------- validation

def validate(items: list[Interval], series: str = "", episode: str = "") -> list[str]:
    """Checks an interval file the way the annotator does -> list of errors.

    An empty list means the file can be read on: identifiers follow the convention
    and are unique, every interval has a positive length, and no two masks overlap
    (touching boundaries are fine -- the intervals are half-open).

    Overlapping masks are an error because the masks are read as a partition: the
    structural ones say where the corpus is, the transitions which seconds inside
    it do not count. Two masks over one stretch leave that undecided.

    An annotation sitting on a mask is NOT an error here -- see :func:`warnings`.
    """
    problems = []
    seen = set()
    for item in items:
        if not item.id:
            problems.append(f"row without an identifier at {item.start:.2f}")
            continue
        try:
            parts = parse_id(item.id)
        except ValueError as error:
            problems.append(str(error))
            continue
        if series and parts["series"] != series:
            problems.append(f"{item.id}: series other than {series}")
        if episode and parts["episode"] != episode:
            problems.append(f"{item.id}: episode other than {episode}")
        if item.is_mask != (parts["kind"] == KIND_MASK):
            problems.append(f"{item.id}: identifier does not match type '{item.type}'")
        if item.id in seen:
            problems.append(f"{item.id}: duplicated identifier")
        seen.add(item.id)
        if item.end <= item.start:
            problems.append(
                f"{item.id}: end {item.end:.2f} not later than start {item.start:.2f}")

    masks = sorted((i for i in items if i.is_mask), key=lambda i: (i.start, i.end))
    for first, second in zip(masks, masks[1:]):
        if second.start < first.end:
            problems.append(
                f"{first.id} [{first.start:.2f}, {first.end:.2f}) overlaps mask "
                f"{second.id} [{second.start:.2f}, {second.end:.2f})")
    return problems


def warnings(items: list[Interval]) -> list[str]:
    """Things worth reporting that do not make a file unreadable.

    Today one thing: an annotation overlapping a mask. Either the TVR marker starts
    too early or the mask was drawn too wide -- a case for a human, so the row
    travels on flagged instead of being dropped.
    """
    masks = [i for i in items if i.is_mask]
    return [f"{item.id}: overlaps mask {mask.id}"
            for item in items if not item.is_mask
            for mask in masks
            if item.start < mask.end and mask.start < item.end]
