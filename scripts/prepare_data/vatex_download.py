#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and verification of VATEX clips from YouTube (both parts).

    python vatex_download.py --split validation --mode descriptions  # enCap to CSV
    python vatex_download.py --split validation --mode download   # 10-second segments
    python vatex_download.py --split validation --mode check      # availability only
    python vatex_download.py --split dev --mode download          # frozen dev draw

Every artifact carries the suffix of its part. For --split validation these are
vatex_report_test.csv (status, resolution, fps, duration, error),
work/vatex_descriptions_test.csv and clips/; for --split dev the _dev ones.
Resumable: a videoID already in the report is skipped, so retrying needs
vatex_retry_errors.py first.

Both parts are downloaded from their own frozen list, written first by
vatex_candidates.py ("list" key of the part): the draw of 750 clips for
--split dev, the inventory of the whole validation split for --split
validation. When the list is absent the run STOPS instead of taking the
part whole, which for "dev" would be nearly 26 thousand training clips.

--mode download exports the descriptions of the part first when their file is
missing. That export is local and takes seconds, so it is not worth a separate
step; --mode descriptions stays available for regenerating it after a change of
the list.

TWO FLAGS THAT ARE EASY TO GET WRONG, see scripts/prepare_data/README_vatex.md:

* ``--remote-components ejs:github`` (hard-coded, EJS_ARGS) is MANDATORY. Without
  it yt-dlp reports no error and silently downloads at a reduced resolution.
  Mass "no_format" means the solver is missing, not that the clips are gone.
* ``--no-force-keyframes`` recovers clips that fail on an ffmpeg crash but
  loosens cut precision; allowed only together with checks B1 and B2.

Cookies were compared and are not needed for quality - only for rate, at the
risk of a ban. A YouTube ID may contain '_' and '-', so "YouTubeID_start_end" is
parsed from the RIGHT.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Paths computed relative to the repository root (variant B), independent of
# the working directory: source material in data/raw, intermediate files in
# data/interim.
ROOT = Path(__file__).resolve().parents[2]
RAW_VATEX = ROOT / "data" / "raw" / "vatex"
INTERIM_VATEX = ROOT / "data" / "interim" / "vatex"
# Intermediate files of a single procedure live one level down, so that the
# dataset directory keeps only what carries a decision or a result.
WORK_VATEX = INTERIM_VATEX / "work"

VATEX_VAL_URL = "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json"

# Two source splits, two sets of outputs. The test part comes from the VATEX
# validation split; the development part -- which measures the matching
# threshold and checks the motion signal -- from the training split, so that the
# test collection keeps the size that matches 18 episodes of a series. Clips go
# to separate directories: `ok_clips` globs one level, so neither sees the other.
# The report carries the acquisition decision and stays at the top level; the
# description export is a plain derivative of the json and goes to work/.
# "only" is the default identifier list of the part: the development part is a
# frozen draw and is never processed whole, the test part IS the whole split.
SPLITS = {
    "validation": {
        "url": VATEX_VAL_URL,
        "json": INTERIM_VATEX / "vatex_validation_v1.0.json",
        # under test/, like the development part under dev/: the clips were
        # moved there after section 03 argued against it, and a re-download that
        # put them back flat would leave two layouts on one disk
        "clips": RAW_VATEX / "test",
        "report": INTERIM_VATEX / "vatex_report_test.csv",
        "descriptions": WORK_VATEX / "vatex_descriptions_test.csv",
        "list": INTERIM_VATEX / "vatex_test_candidates.txt",
    },
    "dev": {
        "url": None,   # the training annotation file is put in place by hand
        "json": INTERIM_VATEX / "vatex_training_v1.0.json",
        "clips": RAW_VATEX / "dev",
        "report": INTERIM_VATEX / "vatex_report_dev.csv",
        "descriptions": WORK_VATEX / "vatex_descriptions_dev.csv",
        "list": INTERIM_VATEX / "vatex_dev_candidates.txt",
    },
}

LOCAL_JSON = SPLITS["validation"]["json"]
CLIPS_DIR = SPLITS["validation"]["clips"]
REPORT_CSV = SPLITS["validation"]["report"]
DESCRIPTIONS_CSV = SPLITS["validation"]["descriptions"]
MAX_HEIGHT = 1080  # upper limit of the download quality
HEADER = ["videoID", "status", "source_height", "source_fps",
          "file_width", "file_height", "file_fps", "duration", "message"]
NOMINAL_DURATION = 10.0  # seconds - excerpt length according to the VATEX annotation
BLOCK_STATUSES = {"bot", "rate_limited"}
BLOCK_THRESHOLD = 10  # this many blocks in a row = stop the run

# Script solving the JS challenges - see the long comment in the docstring.
# DO NOT REMOVE without repeating the format-list comparison (yt-dlp -F with
# and without the flag).
EJS_ARGS = ["--remote-components", "ejs:github"]

# --force-keyframes-at-cuts forces a re-encode so that a keyframe lands exactly
# on the excerpt boundary - without it ffmpeg cuts to the nearest keyframe and
# the window may shift relative to the VATEX annotation.
# The flag is however a source of ffmpeg failures for clips that require
# seeking deep into the video; --no-force-keyframes recovers them at the cost
# of cutting precision. AFTER USING IT measure the durations of the recovered
# clips (notebook, "duration" section).
KEYFRAME_ARGS: list = ["--force-keyframes-at-cuts"]

# set in main() based on --cookies-from-browser
COOKIES_ARGS: list = []


def fetch_json_if_missing(url: str | None = VATEX_VAL_URL) -> dict:
    if not LOCAL_JSON.exists():
        if url is None:
            raise SystemExit(
                f"missing {LOCAL_JSON} - the training annotation file is not "
                "published under a stable URL, put it in place by hand")
        INTERIM_VATEX.mkdir(parents=True, exist_ok=True)
        print(f"Downloading the annotation file: {url}")
        urllib.request.urlretrieve(url, LOCAL_JSON)
    with open(LOCAL_JSON, encoding="utf-8") as f:
        return json.load(f)


def split_video_id(video_id: str):
    """'G9zN5TTuGO4_000179_000189' -> ('G9zN5TTuGO4', 179, 189)  (rsplit: the ID may contain '_')"""
    ytid, start, end = video_id.rsplit("_", 2)
    return ytid, int(start), int(end)


def run(cmd: list, timeout: int = 180):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def print_versions():
    """Prints the tool versions - to be noted in the data chapter.

    A missing Deno does not stop the run, but it must be visible: without it
    (and without the EJS script) yt-dlp silently lowers the quality of the
    downloaded material.
    """
    print(f"Run date: {datetime.now():%Y-%m-%d %H:%M}")
    for name, cmd in [("yt-dlp", ["yt-dlp", "--version"]),
                      ("ffprobe", ["ffprobe", "-version"]),
                      ("deno", ["deno", "--version"])]:
        try:
            result = run(cmd, timeout=30)
            version = (result.stdout or result.stderr).strip().splitlines()[0]
            print(f"  {name:<8} {version}")
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired, IndexError):
            print(f"  {name:<8} NOT FOUND")
            if name == "deno":
                print("           !!! Without Deno yt-dlp cannot decrypt the stream URLs")
                print("           !!! and will download the material at a reduced resolution.")
                print("           !!! winget install DenoLand.Deno -e --source winget")


def classify_error(stderr: str) -> str:
    """Maps a yt-dlp message to a report status.

    THE ORDER OF THE CONDITIONS MATTERS. Three traps:
      - "Sign in to confirm your age" and "Sign in to confirm you're not a bot"
        start identically,
      - "Requested format is not available" contains the phrase "not available",
        but means a format problem, NOT a removed video,
      - "The page needs to be reloaded" hides the substring "age" inside "page";
        the age test uses a word boundary precisely so that it does not swallow
        this message.
    The special cases must be checked before the general ones.
    """
    t = (stderr or "").lower()

    if re.search(r"\bage\b", t) and ("confirm" in t or "restricted" in t or "verify" in t):
        return "age_restricted"
    if "not a bot" in t or "sign in to confirm" in t:
        return "bot"
    # YouTube session challenge: the request reaches the player, but the session
    # data is stale (rotated cookies, expired visitor data). Same family as bot
    # detection and the same cure - back off and come back later - so it shares
    # the 'bot' status. That also makes it trip BLOCK_THRESHOLD; as a plain
    # 'error' a dead session would be ground through the whole remaining list at
    # full speed.
    if "page needs to be reloaded" in t:
        return "bot"
    # "Please sign in." is a separate case: the video exists and is not private,
    # but YouTube demands authentication. It cannot be checked AFTER "private
    # video", because the private-video message also contains "Sign in" - that
    # is why we match the exact phrase "please sign in".
    if "please sign in" in t:
        return "login_required"
    # The bare number is NOT enough: stderr carries the whole googlevideo URL and
    # its parameters are long digit strings, so "429" shows up by pure chance
    # (lmt=1724554429907053 marked real 403s as rate limits). The number has to
    # come with its HTTP context, and 'rate_limited' blocks the run, so a false
    # positive here stops a healthy run for no reason.
    if "error 429" in t or "too many requests" in t:
        return "rate_limited"
    if "requested format is not available" in t or "no video formats found" in t \
            or "only images are available" in t:
        return "no_format"
    if "private video" in t or "is private" in t:
        return "private"
    if "members" in t and ("channel" in t or "only" in t):
        return "members_only"
    if ("country" in t or "geo" in t or "region" in t) and \
            ("not available" in t or "unavailable" in t or "blocked" in t):
        return "geo_blocked"
    if ("unavailable" in t or "not available" in t or "removed" in t
            or "terminated" in t or "has been closed" in t
            or "does not exist" in t or "copyright" in t):
        return "missing"
    return "error"


def extract_message(stderr: str) -> str:
    """Last ERROR line from yt-dlp, truncated and cleaned for writing to CSV."""
    lines = [l.strip() for l in (stderr or "").splitlines() if l.strip()]
    errors = [l for l in lines if l.upper().startswith("ERROR")]
    text = errors[-1] if errors else (lines[-1] if lines else "")
    return text.replace(";", ",").replace("\r", " ").replace("\n", " ")[:200]


def check_availability(ytid: str):
    """Returns (status, source_height, source_fps, stderr) - without downloading."""
    result = run([
        "yt-dlp", "--skip-download", "--no-warnings",
        *EJS_ARGS, *COOKIES_ARGS,
        "--print", "%(height)s|%(fps)s",
        f"https://www.youtube.com/watch?v={ytid}",
    ])
    if result.returncode == 0 and result.stdout.strip():
        height, _, fps = result.stdout.strip().partition("|")
        return "ok", height, fps, ""
    return classify_error(result.stderr), "", "", result.stderr


def download_segment(video_id: str, ytid: str, start: int, end: int,
                     player_client: str | None = None):
    """Downloads the excerpt [start, end]. Returns (succeeded, stderr).

    player_client forces a yt-dlp player client. The default one hands out
    stream URLs that YouTube sometimes refuses to serve (HTTP 403), which
    surfaces as "ffmpeg exited with code" because ffmpeg is the process that
    fetches them. Another client hands out a URL that is still honoured - at
    the cost of the format choice it offers, so this is a rescue, not a default.
    """
    target = CLIPS_DIR / f"{video_id}.mp4"
    if target.exists():
        return True, ""
    client_args = (["--extractor-args", f"youtube:player_client={player_client}"]
                   if player_client else [])
    result = run([
        "yt-dlp", "--no-warnings",
        *EJS_ARGS, *COOKIES_ARGS, *client_args,
        "-f", f"bestvideo[height<={MAX_HEIGHT}]+bestaudio/best[height<={MAX_HEIGHT}]/best",
        "--download-sections", f"*{start}-{end}",
        *KEYFRAME_ARGS,
        "--merge-output-format", "mp4",
        "-o", str(target),
        f"https://www.youtube.com/watch?v={ytid}",
    ], timeout=600)
    if result.returncode == 0 and target.exists():
        return True, ""
    return False, result.stderr


def probe_file(path: Path):
    """ffprobe: resolution, fps and DURATION of the downloaded file.

    The duration is measured right after the download, because measuring the
    whole set later means several thousand ffprobe calls. It serves to detect
    clips shorter than the nominal 10 s: the source recording then ends before
    the VATEX annotation window closes, so the enCap description may have no
    coverage in the material. An upward deviation is harmless (a margin was
    added when cutting to a keyframe), a downward one is not.

    ffprobe reads the container metadata and does not decode the picture, so
    the cost is negligible compared with the download itself.
    """
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ])
    if result.returncode != 0 or not result.stdout.strip():
        return "", "", "", ""
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "", "", "", ""

    streams = data.get("streams") or [{}]
    s = streams[0] if streams else {}
    width = str(s.get("width") or "")
    height = str(s.get("height") or "")

    fps = s.get("avg_frame_rate") or ""
    try:
        a, _, b = fps.partition("/")
        fps = f"{float(a) / float(b):.1f}" if b else fps
    except (ValueError, ZeroDivisionError):
        fps = ""

    # decimal point on purpose: this is a data file, not thesis text
    try:
        duration = f"{float(data.get('format', {}).get('duration')):.2f}"
    except (TypeError, ValueError):
        duration = ""

    return width, height, fps, duration


def export_descriptions(data: dict) -> int:
    DESCRIPTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(DESCRIPTIONS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["videoID", "desc_no", "desc_en"])
        for record in data:
            for i, desc in enumerate(record.get("enCap", []), 1):
                w.writerow([record["videoID"], i, desc])
    print(f"Descriptions saved to {DESCRIPTIONS_CSV}")
    return 0


def main():
    global COOKIES_ARGS, KEYFRAME_ARGS
    global LOCAL_JSON, CLIPS_DIR, REPORT_CSV, DESCRIPTIONS_CSV
    ap = argparse.ArgumentParser(description="VATEX: checking/downloading clips from YouTube")
    ap.add_argument("--split", choices=sorted(SPLITS), default="validation",
                    help="which part: 'validation' = the test collection (default), "
                         "'dev' = the development clips drawn from the training split")
    ap.add_argument("--mode", choices=["check", "download", "descriptions"], default="check",
                    help="'check' = availability on YouTube only, 'download' = the "
                         "clips themselves (exports the descriptions of the part first "
                         "when they are missing), 'descriptions' = only the enCap "
                         "export, locally")
    ap.add_argument("--limit", type=int, default=0,
                    help="how many NEW clips to process in this run (0 = all)")
    ap.add_argument("--pause", type=float, default=12.0, metavar="SECONDS",
                    help="pause between clips (default 12 - guest limit; "
                         "with account cookies 3 is possible)")
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER",
                    help="e.g. firefox - passed to yt-dlp as --cookies-from-browser. "
                         "NOT NEEDED for quality (verified) - only for rate or for the "
                         "'bot'/'age_restricted' statuses. Never with the main Google account.")
    ap.add_argument("--no-force-keyframes", action="store_true",
                    help="skips --force-keyframes-at-cuts. A rescue for clips that fail "
                         "on an ffmpeg crash, but the cut may be less precise - after "
                         "such a run MEASURE the durations of the recovered clips. NOTE: "
                         "it does NOT help against 'ffmpeg exited with code' caused by a "
                         "403 - for that see --player-client")
    ap.add_argument("--player-client", default=None, metavar="CLIENT",
                    help="yt-dlp player client used to RETRY a clip whose first download "
                         "failed (e.g. mweb). Meant for 'ffmpeg exited with code', which "
                         "is a 403 on the stream URL, not an ffmpeg fault. The rescue "
                         "client may expose fewer formats, so rescued clips can be of "
                         "lower resolution - each one is marked 'player_client=...' in "
                         "the message column of the report.")
    ap.add_argument("--check-source", action="store_true",
                    help="in 'download' mode make an extra request that fills the "
                         "source_height/source_fps columns; DOUBLES the number of YouTube requests")
    args = ap.parse_args()

    # Rebound before anything reads them: one split = one json, one clips
    # directory and one pair of output files.
    split = SPLITS[args.split]
    LOCAL_JSON = split["json"]
    CLIPS_DIR = split["clips"]
    REPORT_CSV = split["report"]
    DESCRIPTIONS_CSV = split["descriptions"]

    if args.no_force_keyframes:
        KEYFRAME_ARGS = []
        print("WARNING: without --force-keyframes-at-cuts. After the run measure the durations")
        print("         of the recovered clips - the cut may be shifted relative to the annotation.")

    if args.cookies_from_browser:
        COOKIES_ARGS = ["--cookies-from-browser", args.cookies_from_browser]
        print(f"Cookies from browser: {args.cookies_from_browser} "
              f"(close the browser - YouTube rotates cookies in open tabs)")

    data = fetch_json_if_missing(split["url"])

    # Every part is downloaded from its own frozen list, written by
    # vatex_candidates.py before anything is fetched. There is no flag to forget:
    # without the list the run stops, instead of quietly taking the whole split
    # (nearly 26 thousand clips for 'dev').
    listing = split["list"]
    if not listing.exists():
        raise SystemExit(
            f"missing {listing} - the frozen list of the '{args.split}' part.\n"
            f"Run first: python scripts/prepare_data/vatex_candidates.py "
            f"--split {args.split}")

    wanted = {line.strip() for line in
              listing.read_text(encoding="utf-8").splitlines() if line.strip()}
    known = {r["videoID"] for r in data}
    unknown = wanted - known
    if unknown:
        raise SystemExit(
            f"{listing.name}: {len(unknown)} identifiers are absent from "
            f"{LOCAL_JSON.name} (e.g. {sorted(unknown)[0]}) - wrong part?")
    data = [r for r in data if r["videoID"] in wanted]
    print(f"List {listing.name}: {len(data)} of {len(known)} clips of the part")

    if args.mode == "descriptions":
        return export_descriptions(data)

    # The export is local and takes seconds, so it is done here instead of being
    # a separate step that is easy to forget; the query files are built from it.
    if args.mode == "download" and not DESCRIPTIONS_CSV.exists():
        print(f"No {DESCRIPTIONS_CSV.name} - exporting the descriptions first.")
        export_descriptions(data)

    print_versions()
    print(f"Records to process: {len(data)}")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_VATEX.mkdir(parents=True, exist_ok=True)
    already_done = set()
    if REPORT_CSV.exists():  # resuming
        with open(REPORT_CSV, encoding="utf-8-sig") as f:
            already_done = {row[0] for row in csv.reader(f, delimiter=";") if row}
        print(f"The report already has {max(0, len(already_done) - 1)} clips - they will be skipped.")

    to_do = sum(1 for r in data if r["videoID"] not in already_done)
    if args.limit:
        to_do = min(to_do, args.limit)
    print(f"To process in this run: {to_do}")

    write_mode = "a" if REPORT_CSV.exists() else "w"
    counters = {}
    processed = 0
    blocks_in_a_row = 0
    start_time = time.monotonic()

    with open(REPORT_CSV, write_mode, newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        if write_mode == "w":
            w.writerow(HEADER)

        for record in data:
            vid = record["videoID"]
            if vid in already_done:
                continue
            if args.limit and processed >= args.limit:
                break

            ytid, start, end = split_video_id(vid)
            status, height, fps, stderr = "ok", "", "", ""
            width_f = height_f = fps_f = duration_f = ""
            note = ""

            # The metadata request is made only when it is really needed: in
            # "check" mode (because that is its whole job) or on explicit
            # request. In "download" mode the download itself returns the same
            # error message, so a second request would only eat the YouTube limit.
            if args.mode == "check" or args.check_source:
                status, height, fps, stderr = check_availability(ytid)

            if args.mode == "download" and status == "ok":
                succeeded, stderr_dl = download_segment(vid, ytid, start, end)
                # A refused stream URL is worth one more try with another player
                # client; a private or removed video is not, and retrying it
                # would only double the requests against the YouTube limit.
                if (not succeeded and args.player_client
                        and classify_error(stderr_dl) == "error"):
                    succeeded, stderr_retry = download_segment(
                        vid, ytid, start, end, player_client=args.player_client)
                    if succeeded:
                        note = f"player_client={args.player_client}"
                    else:
                        stderr_dl = stderr_retry
                if succeeded:
                    width_f, height_f, fps_f, duration_f = probe_file(CLIPS_DIR / f"{vid}.mp4")
                else:
                    stderr = stderr_dl
                    status = classify_error(stderr_dl)
                    if status == "error":
                        status = "download_error"

            w.writerow([vid, status, height, fps, width_f, height_f, fps_f, duration_f,
                        note or extract_message(stderr)])
            f.flush()

            counters[status] = counters.get(status, 0) + 1
            processed += 1

            # safeguard: a series of blocks means further grinding is pointless
            blocks_in_a_row = blocks_in_a_row + 1 if status in BLOCK_STATUSES else 0
            if blocks_in_a_row >= BLOCK_THRESHOLD:
                print(f"\nSTOPPED: {BLOCK_THRESHOLD} blocks in a row (status '{status}').")
                print("YouTube is throttling requests. What to do:")
                print("  - wait an hour and run again (the script will resume),")
                print("  - increase --pause,")
                print("  - as a last resort --cookies-from-browser firefox (spare account!).")
                print("Clips blocked in this run can be retried with vatex_retry_errors.py")
                break

            if processed % 25 == 0:
                elapsed = time.monotonic() - start_time
                rate = processed / elapsed * 3600
                remaining = (to_do - processed) / max(rate, 1e-9)
                print(f"[{processed}/{to_do}] {rate:.0f} clips/h, "
                      f"about {remaining:.1f} h remaining | {counters}")

            time.sleep(args.pause)

    print(f"\nDone. Processed {processed} new clips "
          f"in {(time.monotonic() - start_time) / 60:.1f} min.")
    print("Status summary:", counters)
    print(f"Report: {REPORT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
