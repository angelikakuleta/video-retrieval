#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transcribe manual clip rejections into vatex_report_test.csv.

    python vatex_exclude_corrupt.py             # PREVIEW, changes nothing
    python vatex_exclude_corrupt.py --execute

THE DIRECTORY IS THE SOURCE OF TRUTH: fitness for the dataset is a human call, so
you move a file to clips/rejected and the script only writes it down. Works
both ways - a file moved back becomes "ok" again. Reason is not recorded; every
rejection gets the status "corrupt", which is permanent and never retried.

Files are not deleted and the report gets a backup, so the step is reversible.
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Paths relative to the repository root (variant B).
ROOT = Path(__file__).resolve().parents[2]
REPORT_CSV = ROOT / "data" / "interim" / "vatex" / "vatex_report_test.csv"
CLIPS_DIR = ROOT / "data" / "raw" / "vatex"
STATUS = "corrupt"


def main():
    ap = argparse.ArgumentParser(
        description="Assigns the 'corrupt' status to clips moved to the quarantine directory.")
    ap.add_argument("--report", type=Path, default=REPORT_CSV)
    ap.add_argument("--clips", type=Path, default=CLIPS_DIR)
    ap.add_argument("--dir", default="rejected", metavar="NAME",
                    help="quarantine subdirectory inside clips/ (default rejected)")
    ap.add_argument("--no-restore", action="store_true",
                    help="do not revert the 'corrupt' status for files that returned to clips/")
    ap.add_argument("--execute", action="store_true",
                    help="actually save the report (without this flag only a preview)")
    args = ap.parse_args()

    if not args.report.exists():
        print(f"File not found: {args.report.resolve()}")
        print("Pass --report with the path of the download report.")
        return 1

    quarantine = args.clips / args.dir
    if not quarantine.exists():
        print(f"Directory not found: {quarantine.resolve()}")
        print("Move the clips to reject there and run again.")
        return 1

    with open(args.report, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=";") if r]
    if not rows or rows[0][0] != "videoID":
        print("Report without a header row - cannot tell which column is the status.")
        return 1

    header, data = rows[0], rows[1:]
    col = {n: i for i, n in enumerate(header)}
    i_status = col["status"]
    i_msg = col.get("message", len(header) - 1)
    i_duration = col.get("duration")

    def status(r):
        return r[i_status] if i_status < len(r) else ""

    def duration(r):
        if i_duration is None or i_duration >= len(r) or not r[i_duration].strip():
            return None
        try:
            return float(r[i_duration].replace(",", "."))
        except ValueError:
            return None

    # --- what lies in the quarantine ---
    in_quarantine = {p.stem for p in quarantine.glob("*.mp4")}
    other_files = [p.name for p in quarantine.iterdir()
                   if p.is_file() and p.suffix.lower() != ".mp4"]

    rows_by_id = {r[0]: r for r in data}
    unknown = sorted(in_quarantine - set(rows_by_id))

    to_mark = sorted(v for v in in_quarantine & set(rows_by_id)
                     if status(rows_by_id[v]) != STATUS)
    already_marked = sorted(v for v in in_quarantine & set(rows_by_id)
                            if status(rows_by_id[v]) == STATUS)

    # --- reverted decision: status 'corrupt', but the file returned to clips/ ---
    to_restore = []
    if not args.no_restore:
        for r in data:
            if status(r) == STATUS and r[0] not in in_quarantine \
                    and (args.clips / f"{r[0]}.mp4").exists():
                to_restore.append(r[0])

    ok_total = sum(1 for r in data if status(r) == "ok")

    print(f"Report:     {args.report.resolve()}")
    print(f"Quarantine: {quarantine.resolve()}")
    print(f"\n.mp4 files in quarantine:   {len(in_quarantine)}")
    if other_files:
        print(f"files other than .mp4:      {len(other_files)}  <- skipping: "
              f"{', '.join(other_files[:3])}")
    print(f"to mark:                    {len(to_mark)}")
    if already_marked:
        print(f"already marked earlier:     {len(already_marked)}")
    if to_restore:
        print(f"to restore to 'ok':         {len(to_restore)}  <- files returned to clips/")
    if unknown:
        print(f"WITHOUT a row in the report: {len(unknown)}  <- check the names!")
        for v in unknown[:5]:
            print(f"    {v}")

    print(f"\n'ok' clips now:             {ok_total}")
    print(f"after the change:           {ok_total - len(to_mark) + len(to_restore)}")

    if to_mark:
        print("\nTo mark as 'corrupt' (max 15):")
        for v in to_mark[:15]:
            d = duration(rows_by_id[v])
            print(f"  {v:<32} {status(rows_by_id[v]):<10} "
                  f"{f'{d:.2f} s' if d is not None else ''}")

    if to_restore:
        print("\nTo restore to 'ok' (max 15):")
        for v in to_restore[:15]:
            print(f"  {v}")

    if not to_mark and not to_restore:
        print("\nThe report agrees with the directory contents. Nothing to do.")
        return 0

    if not args.execute:
        print("\n--- PREVIEW, nothing saved. Repeat with the --execute flag. ---")
        return 0

    # --- backup copy and save ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.report.with_suffix(args.report.suffix + f".bak_{stamp}")
    shutil.copy2(args.report, backup)

    mark_set = set(to_mark)
    restore_set = set(to_restore)
    for r in data:
        while len(r) <= max(i_status, i_msg):
            r.append("")
        if r[0] in mark_set:
            d = duration(r)
            r[i_status] = STATUS
            # no semicolon: it is the column separator, csv would have to quote the whole field
            r[i_msg] = ("rejected manually after viewing"
                        + (f", duration {d:.2f} s" if d is not None else ""))
        elif r[0] in restore_set:
            r[i_status] = "ok"
            r[i_msg] = ""

    with open(args.report, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(data)

    print(f"\nBackup copy: {backup.name}")
    print(f"Marked as '{STATUS}': {len(mark_set)}")
    if restore_set:
        print(f"Restored to 'ok':    {len(restore_set)}")
    print("\nThe script can be run repeatedly - it always brings the report in line with the directories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
