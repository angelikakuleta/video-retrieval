"""Tools for measuring the properties of video files (ffprobe).

Used by the data preparation notebooks (``tbbt_02_annotation_candidates``,
``office_02_annotations``, ``vatex_02_test_acquisition``).
Reusable for any dataset (tbbt, the office, ...).

The binary is resolved through :mod:`src.utils.tools`: it belongs to the conda
environment, which is not on PATH when a script is started by calling the
environment's interpreter directly.
"""
import csv
import json
import subprocess
from pathlib import Path

from src.utils import tools


def _probe_one(path: Path, exact: bool = False) -> dict:
    if exact:
        entries = "stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration"
        args = [tools.ffprobe(), "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", entries, "-of", "json", str(path)]
    else:
        entries = "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration"
        args = [tools.ffprobe(), "-v", "error", "-select_streams", "v:0",
                "-show_entries", entries, "-of", "json", str(path)]
    out = subprocess.run(args, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        detail = (out.stderr or "").strip() or (out.stdout or "").strip()
        if not detail:
            detail = ("no message on stderr - most likely ffprobe itself failed to start "
                      "(missing DLL, broken installation, wrong build)")
        raise RuntimeError(f"ffprobe error for {path.name} "
                           f"(exit code {out.returncode}): {detail}")
    j = json.loads(out.stdout)
    v = j["streams"][0]

    ratio = v.get("r_frame_rate", "0/0")
    num, den = (ratio.split("/") + ["1"])[:2]
    fps = round(int(num) / int(den), 6) if int(den) else None
    dur = round(float(j["format"]["duration"]), 3) if j.get("format", {}).get("duration") else None

    if exact:
        frames = int(v["nb_read_frames"]); method = "exact"
    elif v.get("nb_frames") not in (None, "N/A"):
        frames = int(v["nb_frames"]); method = "container"
    elif fps and dur:
        frames = round(dur * fps); method = "estimate"
    else:
        frames = None; method = "-"

    return {"file": path.name, "width": v.get("width"), "height": v.get("height"),
            "fps": fps, "fps_fraction": ratio, "frames": frames, "frames_method": method,
            "duration": dur, "codec": v.get("codec_name")}


def probe_videos(directory, pattern: str = "*.mp4", exact: bool = False) -> list:
    """Measures all files in a directory -> list of dicts (sorted by name)."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files ({pattern}) in {directory}")
    return [_probe_one(p, exact) for p in files]


def format_summary(rows: list) -> str:
    """Text: table + uniformity verdict (resolution + fps)."""
    lines = []
    header = f"{'file':<16}{'resolution':<16}{'fps':<12}{'frames':<10}{'duration':<10}{'codec'}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        res = f"{r['width']}x{r['height']}"
        lines.append(f"{r['file']:<16}{res:<16}{str(r['fps_fraction']):<12}"
                     f"{str(r['frames']):<10}{str(r['duration']):<10}{r['codec']}")
    res_set = {(r["width"], r["height"]) for r in rows}
    fps_set = {r["fps_fraction"] for r in rows}
    lines.append("")
    if len(res_set) == 1 and len(fps_set) == 1:
        w, h = next(iter(res_set))
        lines.append(f"OK: all {len(rows)} files identical -> {w}x{h}, fps {next(iter(fps_set))}")
    else:
        lines.append("WARNING: files are NOT uniform!")
        if len(res_set) > 1:
            lines.append("  different resolutions:")
            for w, h in sorted(res_set):
                bad = [r["file"] for r in rows if (r["width"], r["height"]) == (w, h)]
                lines.append(f"    {w}x{h}: {', '.join(bad)}")
        if len(fps_set) > 1:
            lines.append("  different fps:")
            for f in sorted(fps_set):
                bad = [r["file"] for r in rows if r["fps_fraction"] == f]
                lines.append(f"    {f}: {', '.join(bad)}")
    return "\n".join(lines)


def save_csv(rows: list, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    return path


def format_hhmmss(seconds) -> str:
    """Converts seconds to an hh:mm:ss string."""
    s = int(round(float(seconds)))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def total_duration(directory, pattern: str = "*.mp4") -> dict:
    """Sums the duration of all files in a directory (fast probe: only
    format=duration). Returns a dict: files, measured, total, hhmmss.
    Reusable for any dataset (vatex, tbbt, the office...)."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    total = 0.0
    ok = 0
    for p in files:
        out = subprocess.run(
            [tools.ffprobe(), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(p)],
            check=False, capture_output=True, text=True)
        try:
            total += float(out.stdout.strip())
            ok += 1
        except (ValueError, TypeError):
            pass
    return {"files": len(files), "measured": ok,
            "total": round(total, 3), "hhmmss": format_hhmmss(total)}
