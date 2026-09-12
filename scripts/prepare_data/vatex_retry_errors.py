#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clear the failed rows from the report so they can be downloaded again.

    python vatex_retry_errors.py                       # PREVIEW, changes nothing
    python vatex_retry_errors.py --execute
    python vatex_retry_errors.py --split dev --execute
    python vatex_retry_errors.py --execute --statuses error download_error

Needed because vatex_download.py skips every videoID present in the report,
whatever its status. Makes a backup first.

Retried by default: error, download_error, bot, rate_limited, no_format,
login_required. Left alone: missing, private, geo_blocked, members_only, corrupt.
age_restricted only with --include-age-restricted, and only with cookies.
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Paths relative to the repository root (variant B).
ROOT = Path(__file__).resolve().parents[2]
INTERIM_VATEX = ROOT / "data" / "interim" / "vatex"
RAW_VATEX = ROOT / "data" / "raw" / "vatex"

# One part = one report AND one clips directory. They have to move together:
# checking the test directory against a development report would find none of
# the files and clear every 'ok' row, ordering a full re-download.
SPLITS = {
    "validation": {"report": INTERIM_VATEX / "vatex_report_test.csv",
                   "clips": RAW_VATEX},
    "dev": {"report": INTERIM_VATEX / "vatex_report_dev.csv",
            "clips": RAW_VATEX / "dev"},
}
HEADER = ["videoID", "status", "source_height", "source_fps",
          "file_width", "file_height", "file_fps"]
TRANSIENT_STATUSES = ["error", "download_error", "bot", "rate_limited", "no_format",
                      "login_required"]
MIN_FILE_SIZE = 10 * 1024  # below 10 kB the file is considered corrupt


def load_report(path: Path):
    """Returns (header_or_None, list_of_rows)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=";") if r]
    header = None
    if rows and rows[0][0] == "videoID":
        header, rows = rows[0], rows[1:]
    return header, rows


def file_suspicious(video_id: str, clips_dir: Path) -> bool:
    """True when the status says 'ok', but the file is missing or too small."""
    file = clips_dir / f"{video_id}.mp4"
    if not file.exists():
        return True
    try:
        return file.stat().st_size < MIN_FILE_SIZE
    except OSError:
        return True


def main():
    ap = argparse.ArgumentParser(
        description="Removes from the download report the rows of clips to be downloaded again.")
    ap.add_argument("--split", choices=sorted(SPLITS), default="validation",
                    help="which part: 'validation' = the test collection (default), "
                         "'dev' = the development clips")
    ap.add_argument("--report", type=Path, default=None,
                    help="path to the report (default vatex_report_test.csv in data/interim/vatex)")
    ap.add_argument("--statuses", nargs="+", default=None,
                    help=f"statuses to retry (default: {' '.join(TRANSIENT_STATUSES)})")
    ap.add_argument("--include-age-restricted", action="store_true",
                    help="also retry 'age_restricted' (makes sense only with cookies)")
    ap.add_argument("--check-files", action="store_true",
                    help="also retry 'ok' rows whose mp4 file is missing or too small")
    ap.add_argument("--execute", action="store_true",
                    help="actually save the changes (without this flag only a preview)")
    args = ap.parse_args()

    split = SPLITS[args.split]
    clips_dir = split["clips"]
    if args.report is None:
        args.report = split["report"]

    if not args.report.exists():
        print(f"File not found: {args.report.resolve()}")
        print("Pass --report with the path of the download report.")
        return 1

    to_retry = set(args.statuses) if args.statuses else set(TRANSIENT_STATUSES)
    if args.include_age_restricted:
        to_retry.add("age_restricted")

    header, rows = load_report(args.report)

    # --- input statistics ---
    counters = {}
    for r in rows:
        status = r[1] if len(r) > 1 else "(empty)"
        counters[status] = counters.get(status, 0) + 1

    print(f"Report: {args.report.resolve()}")
    print(f"Rows: {len(rows)}")
    print("Statuses:")
    for status, count in sorted(counters.items(), key=lambda p: -p[1]):
        share = 100 * count / len(rows) if rows else 0
        print(f"  {status:<22} {count:>6}  ({share:5.1f}%)")

    # --- row split ---
    kept, removed, missing_files = [], [], []
    for r in rows:
        status = r[1] if len(r) > 1 else ""
        if status in to_retry:
            removed.append(r)
        elif args.check_files and status == "ok" and file_suspicious(r[0], clips_dir):
            missing_files.append(r)
            removed.append(r)
        else:
            kept.append(r)

    print()
    print(f"To retry (statuses {sorted(to_retry)}): "
          f"{len(removed) - len(missing_files)}")
    if args.check_files:
        print(f"To retry (status 'ok', but missing/corrupt file):         {len(missing_files)}")
    print(f"Remaining in the report:                                 {len(kept)}")

    if removed:
        print("\nExamples (max 20):")
        for r in removed[:20]:
            print(f"  {r[0]:<32} {r[1] if len(r) > 1 else ''}")

    if not removed:
        print("\nNothing to retry.")
        return 0

    if not args.execute:
        print("\n--- PREVIEW, nothing saved. Repeat with the --execute flag. ---")
        return 0

    # --- move the corrupt files aside ---
    # vatex_download.py skips the download when the target file already exists,
    # so a corrupt file has to be moved out of the way, otherwise the retry
    # achieves nothing.
    if missing_files:
        quarantine = clips_dir / "rejected"
        quarantine.mkdir(parents=True, exist_ok=True)
        moved = 0
        for r in missing_files:
            file = clips_dir / f"{r[0]}.mp4"
            if file.exists():
                file.replace(quarantine / file.name)
                moved += 1
        print(f"\nMoved {moved} corrupt files to {quarantine}")

    # --- backup copy + save ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.report.with_suffix(args.report.suffix + f".bak_{stamp}")
    shutil.copy2(args.report, backup)

    with open(args.report, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header or HEADER)
        w.writerows(kept)

    print(f"\nBackup copy: {backup.name}")
    print(f"Saved {len(kept)} rows; {len(removed)} clips return to the queue.")
    print("Now run: python vatex_download.py --mode download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
