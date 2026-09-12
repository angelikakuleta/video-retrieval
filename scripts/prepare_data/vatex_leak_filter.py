#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the VATEX clips by Kinetics-400 training leak and by K400 vocabulary.

    python vatex_leak_filter.py --split validation
    python vatex_leak_filter.py --split dev
    python vatex_leak_filter.py --split validation --dry-run

VATEX comes from Kinetics-600 validation, which absorbed part of the K400
TRAINING set the motion models were trained on; such a clip cannot tell
recognition from recall, so it leaves the test split. Familiarity with the
CLASSES stays and is the object of measurement, hence the second split.

Both parts are computed the same way, only from different reports: --split
validation reads vatex_report_test.csv and writes vatex_split_test.csv, --split
dev reads vatex_report_dev.csv and writes vatex_split_dev.csv (the split names
follow the table SPLITS in vatex_download.py). Also reads kinetics-400_train.csv
and kinetics-600_val.csv. Output columns: videoID; class_k600; leak_k400;
class_in_k400_vocab. The test split is leak_k400 = no. Leaked clips stay on disk
- the correctness check needs them. Neither the report nor the files are touched.
"""

import argparse
import csv
import sys
from pathlib import Path

# Paths relative to the repository root (variant B): intermediate files in data/interim.
ROOT = Path(__file__).resolve().parents[2]
INTERIM_VATEX = ROOT / "data" / "interim" / "vatex"
K400_CSV = INTERIM_VATEX / "kinetics-400_train.csv"
K600_CSV = INTERIM_VATEX / "kinetics-600_val.csv"

# One part = one report in, one split out. Same shape as the table SPLITS in
# vatex_download.py, so the two scripts cannot drift apart.
SPLITS = {
    "validation": {
        "report": INTERIM_VATEX / "vatex_report_test.csv",
        "output": INTERIM_VATEX / "vatex_split_test.csv",
    },
    "dev": {
        "report": INTERIM_VATEX / "vatex_report_dev.csv",
        "output": INTERIM_VATEX / "vatex_split_dev.csv",
    },
}
DEFAULT_SPLIT = "validation"
MIN_GROUP = 50  # below this many queries a group is too small for a separate result
YES, NO = "yes", "no"  # coded values in the split file


def load_kinetics(path: Path):
    """Returns (map youtube_id -> row, set of labels)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "youtube_id" not in rows[0]:
        raise ValueError(f"{path.name}: missing column 'youtube_id'")
    return {r["youtube_id"]: r for r in rows}, {r["label"] for r in rows}


def main():
    ap = argparse.ArgumentParser(
        description="K400 training leak filter and class split for the VATEX dataset.")
    ap.add_argument("--split", choices=sorted(SPLITS), default=DEFAULT_SPLIT,
                    help="which part: 'validation' = the test collection (default), "
                         "'dev' = the development clips drawn from the training split; "
                         "picks the input report and the output file")
    ap.add_argument("--report", type=Path, default=None,
                    help="overrides the report of the chosen --split")
    ap.add_argument("--k400", type=Path, default=K400_CSV)
    ap.add_argument("--k600", type=Path, default=K600_CSV)
    ap.add_argument("--output", type=Path, default=None,
                    help="overrides the output file of the chosen --split")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, but do not create the output file")
    args = ap.parse_args()

    split = SPLITS[args.split]
    if args.report is None:
        args.report = split["report"]
    if args.output is None:
        args.output = split["output"]
    print(f"part: {args.split}")

    for p in (args.report, args.k400, args.k600):
        if not p.exists():
            print(f"File not found: {p.resolve()}")
            return 1

    with open(args.report, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=";") if r]
    header, data = rows[0], rows[1:]
    i_status = header.index("status")

    ok = [r[0] for r in data if (r[i_status] if i_status < len(r) else "") == "ok"]
    if not ok:
        print("The report contains no clips with status 'ok'.")
        return 1

    # VATEX videoID = "YouTubeID_start_end"; the ID may contain '_', hence rsplit
    clips = {}
    for vid in ok:
        ytid, start, end = vid.rsplit("_", 2)
        clips[ytid] = (vid, int(start), int(end))

    k400, k400_vocab = load_kinetics(args.k400)
    k600_rows, _ = load_kinetics(args.k600)
    k600 = {y: r["label"] for y, r in k600_rows.items()}   # id -> class name

    print(f"clips with status 'ok':      {len(ok)}")
    print(f"unique source recordings:    {len(clips)}")
    print(f"K400 train: {len(k400)} recordings, {len(k400_vocab)} classes")
    print(f"K600 val:   {len(k600)} recordings")

    # --- 1. training leak ---
    leak = set(clips) & set(k400)
    matching_windows = sum(1 for y in leak
                           if int(k400[y]["time_start"]) == clips[y][1]
                           and int(k400[y]["time_end"]) == clips[y][2])

    print("\n--- training leak (Kinetics-400 train) ---")
    print(f"leaked clips:          {len(leak):>5}  ({100*len(leak)/len(ok):.1f}%)")
    print(f"  time window matches: {matching_windows:>5} of {len(leak)}")
    if matching_windows != len(leak):
        print("  WARNING: not all windows match - the sentence about window agreement")
        print("           in chapter 6 needs a correction.")

    test_split = [y for y in clips if y not in leak]
    print(f"test split:            {len(test_split):>5}  "
          f"({100*len(test_split)/len(ok):.1f}%)")

    # --- 2. split by the K400 vocabulary ---
    unlabeled = [y for y in test_split if y not in k600]
    in_vocab = [y for y in test_split if k600.get(y) in k400_vocab]
    outside = [y for y in test_split if y in k600 and k600[y] not in k400_vocab]

    print("\n--- test split by the K400 vocabulary (for E2) ---")
    m = len(test_split)
    print(f"class in K400 vocab:   {len(in_vocab):>5}  ({100*len(in_vocab)/m:.1f}%)")
    print(f"class outside vocab:   {len(outside):>5}  ({100*len(outside)/m:.1f}%)")
    if unlabeled:
        print(f"without K600 label:    {len(unlabeled):>5}  <- check the K600 file")
    print(f"classes in the test split: {len({k600[y] for y in test_split if y in k600})}")

    for name, group in [("in vocab", in_vocab), ("outside vocab", outside)]:
        if len(group) < MIN_GROUP:
            print(f"\nWARNING: group '{name}' has {len(group)} clips, below the threshold {MIN_GROUP}.")
            print("         Note it as a limitation in the summary chapter.")

    if args.dry_run:
        print("\n--dry-run: no file created.")
        return 0

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["videoID", "class_k600", "leak_k400", "class_in_k400_vocab"])
        for ytid, (vid, _, _) in sorted(clips.items(), key=lambda p: p[1][0]):
            cls = k600.get(ytid, "")
            w.writerow([vid, cls,
                        YES if ytid in leak else NO,
                        (YES if cls in k400_vocab else NO) if cls else ""])

    print(f"\nSaved {args.output} ({len(clips)} rows).")
    print("The test split is the rows with leak_k400 = no.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
