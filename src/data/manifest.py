"""Hashes of the input files of a run -- part of the run's metadata.

A path alone does not identify data (the file may change between runs);
the manifest records the SHA-256 and size of every input, so any result can
be traced back to the exact data it was computed on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.data.datasets import ROOT


def file_entry(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def manifest(paths: list[Path]) -> dict[str, dict]:
    """``relative path -> {sha256, bytes}`` for every existing input file."""
    out = {}
    for path in paths:
        if path.exists():
            key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            out[key] = file_entry(path)
    return out
