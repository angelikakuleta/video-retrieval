"""Do the annotator time and the pipeline frame number point to the same frame?

    python scripts/check_frame_offset.py data/processed/tbbt/S01E01.mp4 83.46
    python scripts/check_frame_offset.py data/processed/tbbt/S01E01.mp4 83.46 --window 4

The time is the annotator reading of the FIRST frame after a hard cut. The script
fetches the neighbourhood twice - (A) by seek, as the pipeline does, (B) by
sequential read, the numbering reference - saves the PNGs under
data/cache/frame_check/, finds the cut itself (largest jump between neighbours)
and prints the offset in frames.

|offset| < 1 frame: consistent. >= 1 frame and the same elsewhere in the episode:
a constant correction to record. Growing along the episode: the file is not CFR.
A and B differing from each other: OpenCV seek is failing, not the timestamps.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def frames_seek(file, numbers):
    """Frames with the given numbers, fetched as in the pipeline (seek by number)."""
    reader = cv2.VideoCapture(str(file))
    out = {}
    for n in numbers:
        reader.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = reader.read()
        if ok:
            out[n] = frame
    reader.release()
    return out


def frames_sequential(file, numbers):
    """The same numbers, but counted from the first frame of the file without any seek."""
    reader = cv2.VideoCapture(str(file))
    wanted, out, n = set(numbers), {}, 0
    last = max(numbers)
    while n <= last:
        ok, frame = reader.read()
        if not ok:
            break
        if n in wanted:
            out[n] = frame
        n += 1
    reader.release()
    return out


def signature(frame):
    """A small, grey version of the frame -- for comparing neighbours."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)


def find_cut(frames):
    """Number of the first frame after the largest jump between neighbours -> (number, jumps)."""
    numbers = sorted(frames)
    jumps = {}
    for a, b in zip(numbers, numbers[1:]):
        jumps[b] = float(np.abs(signature(frames[b]) - signature(frames[a])).mean())
    return max(jumps, key=jumps.get), jumps


def save(frames, directory, prefix, fps):
    for n, frame in frames.items():
        cv2.imwrite(str(directory / f"{prefix}_{n:06d}_{n / fps:.3f}s.png"), frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("time", type=float, help="time of the first frame after the cut (from the annotator)")
    ap.add_argument("--window", type=int, default=3, help="how many frames on each side")
    ap.add_argument("--no-sequential", action="store_true",
                    help="skip the read from the start of the file (faster, but without the reference)")
    a = ap.parse_args()

    reader = cv2.VideoCapture(str(a.file))
    fps = reader.get(cv2.CAP_PROP_FPS)
    count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.release()
    if not fps:
        sys.exit(f"could not open: {a.file}")

    number = round(a.time * fps)
    numbers = [n for n in range(number - a.window, number + a.window + 1) if 0 <= n < count]
    print(f"file: {a.file.name}   fps: {fps:.6f}   frames: {count}")
    print(f"annotator time: {a.time:.3f} s  ->  time*fps = {a.time * fps:.2f}, "
          f"number as in the pipeline: {number}\n")

    directory = (Path("data/cache/frame_check")
                 / f"{a.file.stem}_{a.time:.2f}".replace(".", "_"))
    directory.mkdir(parents=True, exist_ok=True)

    sets = [("A_seek", frames_seek(a.file, numbers))]
    if not a.no_sequential:
        sets.append(("B_seq", frames_sequential(a.file, numbers)))

    results = {}
    for name, frames in sets:
        if len(frames) < 2:
            print(f"{name}: too few frames ({len(frames)})")
            continue
        save(frames, directory, name, fps)
        cut, jumps = find_cut(frames)
        results[name] = cut
        print(f"{name}: jumps between neighbouring frames (number -> difference)")
        for n, s in jumps.items():
            print(f"   {n:6d} ({n / fps:8.3f} s)  {s:6.1f}" + ("   <-- cut" if n == cut else ""))
        print(f"   first frame after the cut: {cut}   "
              f"offset relative to the annotator: {cut - a.time * fps:+.2f} frames\n")

    if len(results) == 2 and results["A_seek"] != results["B_seq"]:
        print("WARNING: seek by number (A) and sequential read (B) point to different frames --"
              " OpenCV seek is failing, not the timestamps.")
    print(f"frames to inspect: {directory}")


if __name__ == "__main__":
    main()
