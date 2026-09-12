"""Requirement tags, complexity label and named characters of a test query.

Chapter 4 gives every query five binary tags, a complexity label and the
characters it refers to. All of it is hand work, so it lives in its own file
rather than in the ready query files, which are rewritten from the register on
every run:

    data/annotations/<series>/<series>_query_tags.csv

The file is a round trip. :func:`skeleton` adds the missing rows and never
touches a filled cell, the same rule that protects the interval files;
:func:`merge` carries the values into the query records.

Any tag can also be marked on the annotation in the annotator, which is the only
way to assign one that cannot be told from the query text -- ``wymaga_osoby``
compares the query against the other events of the same episode. Whatever was
marked reaches this module through the register's ``tags`` column
(:func:`from_register`).

VATEX uses the same file with a shorter column set (:data:`VATEX_COLUMNS`),
because only part of the schema can be settled from a single clip description.
Every function takes the column set as an argument and defaults to the series
one, so which columns a dataset has is decided in one place.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.utils import settings
from src.utils.vocabulary import (
    COMPLEXITY_VALUES,
    REQUIREMENT_TAGS,
    RequirementTag,
    resolve_characters,
)

#: rewritten from the register on every run -- they only say which row is which,
#: so nothing typed into them survives
CONTEXT_COLUMNS = ["episode", "split"]

#: `desc` sits outside CONTEXT_COLUMNS on purpose: it is the wording the query
#: files are built from and it is corrected HERE, so :func:`skeleton` fills it in
#: only when it is empty and the register never overwrites it afterwards.
COLUMNS = (["desc_id"] + CONTEXT_COLUMNS + ["desc"] + REQUIREMENT_TAGS
           + ["complexity", "identities"])

#: VATEX carries fewer columns, and that is a decision of chapter 4 rather than a
#: shortcut: a one-sentence clip description settles the object, the scenery and
#: the motion, and nothing beyond them. ``wymaga_osoby`` would have to be read
#: against the other events of the same recording, and VATEX clips are
#: independent; ``wymaga_mimiki`` needs a facial expression the descriptions
#: never mention; ``identities`` needs recurring characters with profiles, which
#: VATEX has not. An absent column means "not assigned here", never "0".
VATEX_COLUMNS = (["desc_id"] + CONTEXT_COLUMNS + ["desc"]
                 + [RequirementTag.OBJECT.value, RequirementTag.SCENERY.value,
                    RequirementTag.MOTION.value]
                 + ["complexity"])

#: datasets whose tag file differs from the series one; the series share COLUMNS
COLUMN_SETS = {"vatex": VATEX_COLUMNS}

#: separator of the ``identities`` cell
IDENTITY_SEPARATOR = ","


class TagError(ValueError):
    """A cell outside the documented set of values."""


def columns_for(source: str) -> list[str]:
    """Column set of the tag file of one dataset."""
    return list(COLUMN_SETS.get(source, COLUMNS))


def tags_in(columns: list[str]) -> list[str]:
    """Requirement tags the given column set carries, in vocabulary order."""
    present = set(columns)
    return [tag for tag in REQUIREMENT_TAGS if tag in present]


# ----------------------------------------------------------------------- I/O

def path_for(series: str, directory: Path | str) -> Path:
    """Location of the tag file of one series."""
    return Path(directory) / f"{series}_query_tags.csv"


def load(path: Path | str, columns: list[str] = COLUMNS) -> dict[int, dict]:
    """``desc_id -> row``; an absent file is an empty mapping, not an error."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    out: dict[int, dict] = {}
    for row in rows:
        raw = (row.get("desc_id") or "").strip()
        if not raw:
            continue
        out[int(raw)] = {key: (row.get(key) or "").strip() for key in columns}
    return out


def save(rows: dict[int, dict], path: Path | str,
         columns: list[str] = COLUMNS) -> Path:
    """Writes the file back: development part first, then test; inside each by
    episode and identifier.

    The order matters because this is the file filled in by hand -- the rows that
    are worked on now have to be together at the top, not interleaved with the
    part that is not annotated yet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(),
                     key=lambda r: (settings.split_order(r.get("split", "")),
                                    r.get("episode", ""), int(r["desc_id"])))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return path


# ---------------------------------------------------------------- validation

def _binary(value: str, column: str, desc_id: int) -> int | None:
    value = (value or "").strip()
    if value == "":
        return None
    if value not in ("0", "1"):
        raise TagError(f"{desc_id}: {column} = {value!r}, expected 0, 1 or empty")
    return int(value)


def is_filled(row: dict, columns: list[str] = COLUMNS) -> bool:
    """Whether every decision column carries a value.

    ``identities`` is not part of the test: an empty cell is a legitimate answer.
    Only the tags ``columns`` carries are asked for -- a tag the dataset does not
    assign at all must not make every row look half-filled.
    """
    return (all((row.get(tag) or "").strip() != "" for tag in tags_in(columns))
            and (row.get("complexity") or "").strip() != "")


def validate(rows: dict[int, dict], columns: list[str] = COLUMNS) -> list[str]:
    """The problems found; an empty list means the file is usable."""
    problems = []
    for desc_id, row in sorted(rows.items()):
        for tag in tags_in(columns):
            try:
                _binary(row.get(tag, ""), tag, desc_id)
            except TagError as error:
                problems.append(str(error))
        complexity = (row.get("complexity") or "").strip()
        if complexity not in ("", *COMPLEXITY_VALUES):
            problems.append(f"{desc_id}: complexity = {complexity!r}, "
                            f"expected {'/'.join(COMPLEXITY_VALUES)} or empty")
    return problems


# ------------------------------------------------------------- reading a row

def requirements_of(row: dict) -> list[str]:
    """Tags set to 1, in the order of :data:`REQUIREMENT_TAGS`."""
    return [tag for tag in REQUIREMENT_TAGS if (row.get(tag) or "").strip() == "1"]


def identities_of(row: dict) -> list[str]:
    """Character names of the row, split on the separator and trimmed."""
    raw = (row.get("identities") or "").strip()
    if not raw:
        return []
    return [name.strip() for name in raw.split(IDENTITY_SEPARATOR) if name.strip()]


def complexity_of(row: dict) -> str | None:
    value = (row.get("complexity") or "").strip()
    return value or None


# --------------------------------------------------- building and refreshing

def tags_of(register_row: dict) -> set[str]:
    """Labels put on one annotation in the annotator."""
    return set(str(register_row.get("tags", "")).split())


def from_register(register_rows: list[dict]) -> dict[int, dict[str, str]]:
    """``desc_id -> {tag: "1"/"0"/""}`` for every tag marked in the annotator.

    A marked row gives ``"1"``. An unmarked one is ambiguous, so the distinction is
    drawn per episode and per tag: an episode where a tag was used at least once
    counts as reviewed for it and its remaining rows become ``"0"``; an episode
    where it never appears stays empty, so a pass that has not happened is never
    mistaken for a pass that found nothing.
    """
    reviewed = {
        tag: {r["episode"] for r in register_rows if tag in tags_of(r)}
        for tag in REQUIREMENT_TAGS
    }
    out: dict[int, dict[str, str]] = {}
    for row in register_rows:
        marked = tags_of(row)
        out[int(row["desc_id"])] = {
            tag: "1" if tag in marked else ("0" if row["episode"] in reviewed[tag] else "")
            for tag in REQUIREMENT_TAGS
        }
    return out


def annotator_report(register_rows: list[dict]) -> dict:
    """What the annotator's labels contribute, per tag, plus anything unknown."""
    from src.utils.vocabulary import unknown_tags

    marked = {tag: sum(1 for r in register_rows if tag in tags_of(r))
              for tag in REQUIREMENT_TAGS}
    episodes = {tag: sorted({r["episode"] for r in register_rows if tag in tags_of(r)})
                for tag in REQUIREMENT_TAGS}
    seen = {name for r in register_rows for name in tags_of(r)}
    return {"marked": marked, "episodes": episodes, "unknown": unknown_tags(seen)}


def misaligned(rows: dict[int, dict],
               episode_of: dict[int, str]) -> list[tuple[int, str, str]]:
    """Rows whose ``episode`` no longer matches what their identifier now names.

    The file is keyed by ``desc_id``, so an identifier that moved to another
    recording would drag the hand-filled cells along without a word. Rows the
    current set no longer has are NOT a mismatch -- :func:`skeleton` drops those.

    Returns ``(desc_id, in the file, now)`` for each row that diverges.
    """
    out = []
    for desc_id, row in sorted(rows.items()):
        was = (row.get("episode") or "").strip()
        now = episode_of.get(desc_id)
        if now is not None and was and was != now:
            out.append((desc_id, was, now))
    return out


def skeleton(register_rows: list[dict], existing: dict[int, dict],
             from_annotator: dict[int, dict[str, str]] | None = None,
             columns: list[str] = COLUMNS
             ) -> tuple[dict[int, dict], dict]:
    """Fresh row set for the accepted annotations, keeping every filled cell.

    ``desc`` is NOT a context column: it is the wording the query files are built
    from, so a text already in the file is kept and the register never overwrites
    it. ``retexted`` lists the rows where the register now says something else --
    a typo fixed in the annotator after the tag file had been filled -- so they
    can be looked at instead of being silently resolved either way.

    Returns the mapping and ``{"added", "kept", "dropped", "retexted",
    "from_annotator"}``.
    """
    from_annotator = from_annotator or {}
    counts: dict = {"added": 0, "kept": 0, "dropped": 0, "retexted": [],
                    "from_annotator": 0}
    fresh: dict[int, dict] = {}

    for row in register_rows:
        desc_id = int(row["desc_id"])
        desc = str(row["desc"]).strip()
        previous = existing.get(desc_id)
        entry = {key: "" for key in columns}
        if previous is not None:
            entry.update({key: previous.get(key, "") for key in columns})
            if previous.get("desc", "") and previous["desc"] != desc:
                counts["retexted"].append(desc_id)
            counts["kept"] += 1
        else:
            counts["added"] += 1
        # context columns always follow the register; `desc` does not -- it is
        # hand-corrected here and this file is what the query files read
        entry["desc_id"] = str(desc_id)
        entry["episode"] = row["episode"]
        entry["split"] = row["split"]
        if not entry["desc"]:
            entry["desc"] = desc
        for tag, value in from_annotator.get(desc_id, {}).items():
            # a tag outside `columns` is not assigned in this dataset at all
            if tag in entry and value and not entry[tag]:
                entry[tag] = value
                counts["from_annotator"] += 1
        fresh[desc_id] = entry

    counts["dropped"] = len(set(existing) - set(fresh))
    return fresh, counts


def coverage(rows: dict[int, dict], split: str | None = None,
             columns: list[str] = COLUMNS) -> dict:
    """How much of the file is filled in, per episode and in total."""
    selected = [r for r in rows.values()
                if split is None or r.get("split") == split]
    per_episode: dict[str, dict] = {}
    for row in selected:
        entry = per_episode.setdefault(row.get("episode", ""), {"total": 0, "filled": 0})
        entry["total"] += 1
        entry["filled"] += int(is_filled(row, columns))
    return {
        "total": len(selected),
        "filled": sum(1 for r in selected if is_filled(r, columns)),
        "per_episode": dict(sorted(per_episode.items())),
    }


# --------------------------------------------------------------------- merge

def merge(queries: list, rows: dict[int, dict],
          profiled: list[str] | None = None,
          columns: list[str] = COLUMNS) -> dict:
    """Writes the tags into ready query records, in place.

    The wording travels with the tags: a row that carries a ``desc`` replaces the
    query text, because typos are fixed in this file and nowhere else afterwards.

    ``profiled`` names the characters that have a profile. Given it, ``identities``
    keeps only those (spelled as ``profiled`` spells them) and ``wymaga_osoby`` is
    recomputed as "does anything remain" instead of being taken from the file: the
    tag marks the queries the identity signal is active for, and after the filter
    that is exactly the queries with a profiled character left.

    Without ``profiled`` both fields are taken as they are. A query with no row
    keeps its register wording and its empty tags.
    """
    counts = {"tagged": 0, "untagged": 0, "identities_dropped": 0, "retexted": 0}
    for query in queries:
        row = rows.get(int(query.desc_id))
        if row is None:
            counts["untagged"] += 1
            continue
        # the wording is taken even from a row whose tags are still empty
        text = (row.get("desc") or "").strip()
        if text and text != query.desc:
            query.desc = text
            counts["retexted"] += 1
        if not is_filled(row, columns):
            counts["untagged"] += 1
            continue
        requirements = requirements_of(row)
        identities = identities_of(row)
        if profiled is not None:
            kept = resolve_characters(identities, profiled)
            counts["identities_dropped"] += len(identities) - len(kept)
            identities = kept
            requirements = [t for t in requirements if t != RequirementTag.PERSON]
            if identities:
                requirements.append(RequirementTag.PERSON.value)
            requirements = [t for t in REQUIREMENT_TAGS if t in set(requirements)]
        query.requirements = requirements
        query.complexity = complexity_of(row)
        query.identities = identities
        counts["tagged"] += 1
    return counts


# ---------------------------------------------------------------- statistics

def statistics(queries: list, columns: list[str] = COLUMNS) -> dict:
    """Counts of the query set by tag, complexity and number of characters.

    The last one is the row chapter 6 asks for, broken down by how many PROFILED
    characters a query names. Only the tags ``columns`` carries are counted, so a
    dataset that does not assign a tag reports no row for it instead of a zero.
    """
    counts = {"count": len(queries),
              "requirements": {tag: 0 for tag in tags_in(columns)},
              "complexity": {value: 0 for value in COMPLEXITY_VALUES},
              "complexity_missing": 0,
              "identities": {}}
    for query in queries:
        for tag in query.requirements or []:
            if tag in counts["requirements"]:
                counts["requirements"][tag] += 1
        if query.complexity in counts["complexity"]:
            counts["complexity"][query.complexity] += 1
        else:
            counts["complexity_missing"] += 1
        many = len(query.identities or [])
        counts["identities"][many] = counts["identities"].get(many, 0) + 1
    counts["identities"] = dict(sorted(counts["identities"].items()))
    return counts
