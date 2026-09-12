#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen list of the VATEX clips one part is built from.

    python scripts/prepare_data/vatex_candidates.py --split dev
    python scripts/prepare_data/vatex_candidates.py --split validation

Step one of the same procedure for both parts: what gets downloaded is written
down first, and only then fetched. The parts differ in how the list arises, not
in what it is -- one identifier per line, sorted, frozen once written.

    dev          drawn from the VATEX TRAINING split, DEV_SIZE clips: a
                 quarter of the 3000 the test list holds. The development part
                 measures the matching threshold and checks the motion signal;
                 it must NOT come out of the test split, whose size (2489 clips
                 downloaded of 3000 listed) matches the fragment count of 18
                 episodes of a series and is kept for cross-domain comparison.
    validation   every clip of the VATEX VALIDATION split -- the test collection
                 is the whole of it, so the list is an inventory, not a choice.

Why the list is frozen BEFORE downloading: what fails to download is lost and
nothing is drawn in its place, so the composition cannot depend on what YouTube
happened to serve. The same discipline as the episode selection of the series.
For the validation part the list was written after its clips had already been
downloaded; there it proves nothing, and it does not need to -- "all of them" is
not a choice availability could have influenced. It exists so that both parts
enter the download step the same way.

Reads vatex_<part>_v1.0.json, and for the draw also kinetics-600_val.csv (class
of every clip) and kinetics-400_train.csv (training leak of the motion model).
The K600 file is the VALIDATION one on purpose, for both parts: the whole of
VATEX is built on the Kinetics-600 validation pool and the VATEX train/val split
partitions that same pool -- verified, it covers 25991 of 25991 training clips.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERIM_VATEX = ROOT / "data" / "interim" / "vatex"

K600_CSV = INTERIM_VATEX / "kinetics-600_val.csv"
K400_CSV = INTERIM_VATEX / "kinetics-400_train.csv"

#: clips drawn for the development part -- a quarter of the 3000 the test list
#: holds. The development part only has to settle the matching threshold and
#: check the motion signal, so it is deliberately smaller than the collection
#: the results are read off.
DEV_SIZE = 750

#: one entry per part: where the annotations are, where the list goes, and
#: whether the part is drawn from the pool or taken whole
SPLITS = {
    "dev": {"json": INTERIM_VATEX / "vatex_training_v1.0.json",
            "list": INTERIM_VATEX / "vatex_dev_candidates.txt",
            "draw": True},
    "validation": {"json": INTERIM_VATEX / "vatex_validation_v1.0.json",
                   "list": INTERIM_VATEX / "vatex_test_candidates.txt",
                   "draw": False},
}

#: The draw is UNIFORM over the clean pool -- no stratification. The proportion
#: of clips whose class is in the Kinetics-400 vocabulary is whatever the pool
#: gives; it is reported, not imposed. Imposing the proportion of the test part
#: would let a property of a collection built LATER shape the development part.


def load_kinetics(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in ("youtube_id", "label"):
        if not rows or column not in rows[0]:
            raise SystemExit(f"{path.name}: missing column {column!r}")
    return rows


def youtube_id(vid_name: str) -> str:
    """'G9zN5TTuGO4_000179_000189' -> 'G9zN5TTuGO4'.

    Split from the RIGHT: a YouTube identifier may itself contain '_' and '-'.
    """
    return vid_name.rsplit("_", 2)[0]


def draw(records: list[dict], size: int, seed: int) -> list[str]:
    """Clips without a Kinetics-400 training leak, drawn uniformly."""
    k600 = {r["youtube_id"]: r["label"] for r in load_kinetics(K600_CSV)}
    k400_rows = load_kinetics(K400_CSV)
    k400_ids = {r["youtube_id"] for r in k400_rows}
    k400_vocab = {r["label"] for r in k400_rows}

    missing = [r["videoID"] for r in records if youtube_id(r["videoID"]) not in k600]
    if missing:
        raise SystemExit(
            f"{len(missing)} clips have no class in {K600_CSV.name} "
            f"(e.g. {missing[0]}) - wrong K600 file for this part")

    # Groups first, draw second: the leak has to be known before anything is
    # picked, otherwise the sample would come out of an already polluted pool.
    groups: dict[tuple[bool, bool], list[str]] = {}
    for record in records:
        vid = record["videoID"]
        ytid = youtube_id(vid)
        groups.setdefault((k600[ytid] in k400_vocab, ytid in k400_ids), []).append(vid)

    total = len(records)
    for in_vocab in (True, False):
        for leak in (False, True):
            count = len(groups.get((in_vocab, leak), []))
            label = "in K400 " if in_vocab else "outside "
            print(f"  class {label}| leak {'yes' if leak else 'no '}: "
                  f"{count:>6}  ({100 * count / total:5.2f}%)")

    clean_in = groups.get((True, False), [])
    clean = sorted(clean_in + groups.get((False, False), []))
    print(f"\nno leak: {len(clean)} clips ({len(clean_in)} in K400, "
          f"{len(clean) - len(clean_in)} outside)")
    if len(clean) < size:
        raise SystemExit(f"only {len(clean)} clips without a leak, {size} wanted")

    # Sorted pool, one seeded generator: the draw must not depend on the order
    # the records happen to have in the json.
    drawn = sorted(random.Random(seed).sample(clean, size))
    in_vocab = set(clean_in)
    drawn_in = sum(1 for vid in drawn if vid in in_vocab)
    print(f"\ndrawn: {len(drawn)} clips  (seed {seed}, uniform over the clean pool)")
    print(f"  class in K400: {drawn_in:>4}  ({100 * drawn_in / len(drawn):5.1f}%)")
    print(f"  class outside: {len(drawn) - drawn_in:>4}  "
          f"({100 * (len(drawn) - drawn_in) / len(drawn):5.1f}%)")
    return drawn


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Frozen list of the clips one VATEX part is built from.")
    ap.add_argument("--split", choices=sorted(SPLITS), required=True,
                    help="'dev' draws from the training split, 'validation' takes "
                         "the whole validation split")
    ap.add_argument("--size", type=int, default=DEV_SIZE,
                    help=f"clips to draw, 'dev' only (default {DEV_SIZE}, a quarter "
                         "of the test list; part of them is lost to the download, "
                         "as 3000 -> 2489 was for the test part)")
    ap.add_argument("--seed", type=int, default=1234, help="draw seed, 'dev' only")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing list; without it the script stops, "
                         "because the list is frozen once written")
    args = ap.parse_args()

    part = SPLITS[args.split]
    output = part["list"]
    inputs = [part["json"]] + ([K600_CSV, K400_CSV] if part["draw"] else [])
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    if output.exists() and not args.force:
        raise SystemExit(
            f"{output.name} already exists - the list is frozen before the "
            "download, so it is not rewritten. Pass --force only if nothing has "
            "been downloaded from it yet.")

    with open(part["json"], encoding="utf-8") as handle:
        records = json.load(handle)
    print(f"VATEX {args.split} split: {len(records)} clips")

    if part["draw"]:
        clips = draw(records, args.size, args.seed)
    else:
        clips = sorted(r["videoID"] for r in records)
        print(f"\ntaken whole: {len(clips)} clips (the test collection is the "
              "entire validation split)")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(clips) + "\n", encoding="utf-8")
    print(f"saved -> {output.relative_to(ROOT)}")
    if part["draw"]:
        print("\nCOMMIT THIS FILE BEFORE DOWNLOADING - what fails to download is "
              "lost and nothing is drawn in its place.")
    else:
        print("\nThe part is taken whole, so this list records what was "
              "downloaded rather than constraining it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
