"""Test query record and its serialization in JSONL format.

The format is shared by all three datasets. The record matches the structure
from the thesis (table "Structure of a test query record"): the fields
``desc_id``, ``desc``, ``vid_name`` and ``ts`` come from the TVR schema, and the
remaining ones extend it with the information needed for the per-tag analysis.

A JSONL file holds one record per line, so it can be read as a stream and
appended to without loading the whole file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Query:
    """A single test query in the schema shared by all datasets."""

    desc_id: int              # unique query identifier
    desc: str                 # query text in English
    vid_name: str             # episode or clip identifier
    ts: tuple[float, float]   # event boundaries on the recording's time axis, in seconds
    source: str               # origin of the query, e.g. "vatex"
    # the vocabularies of the three fields below live in src.utils.vocabulary;
    # they are filled in from the hand-written tag file (src.annotation.tags)
    requirements: list[str] = field(default_factory=list)  # RequirementTag values
    complexity: str | None = None                          # Complexity value: "P" / "Z"
    identities: list[str] = field(default_factory=list)    # characters mentioned
    event_id: str = ""        # shared by the occurrences of the same event

    def to_dict(self) -> dict:
        record = asdict(self)
        record["ts"] = list(self.ts)  # JSON has no tuples -- stored as a list
        return record

    @classmethod
    def from_dict(cls, data: dict) -> Query:
        data = dict(data)
        data["ts"] = tuple(data["ts"])
        return cls(**data)


def save_jsonl(queries: Iterable[Query], file: Path | str) -> Path:
    """Saves the queries to a JSONL file (one record per line)."""
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.to_dict(), ensure_ascii=False) + "\n")
    return file


def load_jsonl(file: Path | str) -> list[Query]:
    """Loads the queries from a JSONL file."""
    with Path(file).open(encoding="utf-8") as handle:
        return [Query.from_dict(json.loads(line)) for line in handle if line.strip()]
