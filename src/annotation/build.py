"""Turning the register of a series into the query files read by the pipeline.

One JSONL per split (``dev``, ``test``) in the schema shared by all three
datasets (:mod:`src.utils.queries`); only accepted annotations enter it.

The retrieval unit is a fragment of an episode, so ``vid_name`` identifies the
recording (``tbbt_s01e01``) and ``ts`` gives the event boundaries on its axis.
Every annotation is a separate event, so ``event_id`` is the annotation
identifier -- which still carries the original ``desc_id``.

Requirement tags, complexity and named characters come from the hand-filled tag
file (:mod:`src.annotation.tags`) and are merged in here. They cannot live in
the query files: those are rewritten from the register on every run.
"""

from __future__ import annotations

from pathlib import Path

from src.annotation import registry, tags
from src.utils.queries import Query, save_jsonl

#: splits written as separate files
SPLITS = ("dev", "test")


def video_stem(row: dict) -> str:
    """Recording identifier of a register row: the video file without extension."""
    return str(row["video_file"]).rsplit(".", 1)[0]


def queries(rows: list[dict], source: str,
            tag_rows: dict[int, dict] | None = None,
            profiled: list[str] | None = None) -> list[Query]:
    """Accepted register rows -> query records, ordered by episode and time.

    ``tag_rows`` is the tag file. Records it does not cover keep empty tags,
    complexity and identity list, so a half-filled file narrows the tag analyses
    instead of blocking the build.
    """
    ordered = sorted(rows, key=lambda r: (r["episode"], float(r["start"]),
                                          str(r["annotation_id"])))
    records = [
        Query(
            desc_id=int(row["desc_id"]),
            desc=str(row["desc"]).strip(),
            vid_name=video_stem(row),
            ts=(round(float(row["start"]), 2), round(float(row["end"]), 2)),
            source=source,
            event_id=str(row["annotation_id"]),
        )
        for row in ordered
    ]
    if tag_rows:
        tags.merge(records, tag_rows, profiled)
    return records


def write_splits(rows: list[dict], directory: Path | str, series: str,
                 splits: tuple[str, ...] = SPLITS,
                 tag_rows: dict[int, dict] | None = None,
                 profiled: list[str] | None = None) -> dict[str, Path]:
    """Writes ``<series>_queries_<split>.jsonl`` for every split.

    Returns ``{split: path}``. A split with no accepted annotations still gets a
    file, so a missing one always means the step was not run. Without ``tag_rows``
    the tag file is read from ``directory``.
    """
    directory = Path(directory)
    if tag_rows is None:
        tag_rows = tags.load(tags.path_for(series, directory))
    written = {}
    for split in splits:
        records = queries(registry.accepted(rows, split), series, tag_rows, profiled)
        written[split] = save_jsonl(records, directory / f"{series}_queries_{split}.jsonl")
    return written


def summary(rows: list[dict], splits: tuple[str, ...] = SPLITS) -> list[dict]:
    """How many queries and episodes each split file holds."""
    out = []
    for split in splits:
        group = registry.accepted(rows, split)
        lengths = sorted(float(r["duration"]) for r in group)
        out.append({
            "split": split,
            "episodes": len({r["episode"] for r in group}),
            "queries": len(group),
            "median_duration": round(lengths[len(lengths) // 2], 2) if lengths else 0.0,
            "total_duration": round(sum(lengths), 1),
        })
    return out
