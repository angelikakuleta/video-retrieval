"""Anchoring the time of TVQA/TVR clips on the scale of the whole episode.

TVR timestamps are counted from the start of a 60-90 s clip; the pipeline works
on full episodes, so every timestamp has to be shifted. The only common point
of reference are the subtitles: TVQA ships subtitles for the clips
(``data/interim/tbbt/tvqa_subtitles/``), we have our own for the episodes
(``data/subs/tbbt/``). They come from different authors and differ in speaker
prefixes, stage directions, line breaking, sometimes wording and timestamps.

Method:

1. Both files become word streams -- text normalized (HTML, stage directions,
   speaker prefixes, punctuation, case) and every word given a time
   interpolated inside its line. Normalization removes exactly what the two
   sets differ in most.
2. The streams are matched with ``difflib.SequenceMatcher``. Words rather than
   lines, so different line breaking stops mattering.
3. Matched blocks give anchors -- pairs of times. An anchor is exact when the
   word is the first of a line on both sides (no interpolation error);
   interpolated anchors are the fallback.
4. The offset of a clip is the median of the differences after discarding
   outliers (1-D RANSAC with half-window ``TOLERANCE``).

The scale is estimated once per episode as a check (:func:`episode_scale`), not
per clip -- empirically 1.00000 everywhere, so the mapping is a pure offset.

Independent check: TVQA cut the clips on whole seconds, so the offsets should
lie on a 1 s grid shifted by a constant ``delta``. :func:`second_grid` measures
the concentration on that grid (``R``, 0-1); a high ``R`` allows snapping.

Series-specific (hence the ``tbbt_`` prefix) and standard library only.
"""
import csv
import difflib
import math
import re
import statistics
from pathlib import Path

#: minimum length (in words) of a block counted as matched
MIN_BLOCK = 4
#: half-width of the anchor agreement window when estimating the offset [s]
TOLERANCE = 1.0
#: below this many exact anchors a clip goes to manual checking
MIN_ANCHORS = 5
#: concentration on the second grid from which offsets may be snapped
MIN_CONCENTRATION = 0.90
#: admissible residual against the grid when snapping [s]
MAX_GRID_RESIDUAL = 0.25
#: smallest offset jump inside a clip regarded as real [s]
MIN_JUMP = 1.5

_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
                   r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_TAGS = re.compile(r"<[^>]+>")
_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")
_SPEAKER = re.compile(r"^\s*[-–—]?\s*[A-Z][A-Z' .]{1,20}:\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9' ]+")


def _round(x, n=3):
    """Rounding without negative zero (-0.0 in a CSV reads like an error)."""
    v = round(x, n)
    return 0.0 if v == 0 else v


# ----------------------------------------------------------------- subtitles

def read_srt(path) -> list:
    """Reads an SRT file -> list of tuples (start, end, text) in seconds."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for block in re.split(r"\n\s*\n", text):
        rows = [l for l in block.split("\n") if l.strip()]
        if len(rows) < 2:
            continue
        i = 1 if re.fullmatch(r"\d+", rows[0].strip()) else 0
        if i >= len(rows):
            continue
        m = _TIME.search(rows[i])
        if not m:
            continue
        g = m.groups()
        start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
        end = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000
        lines.append((start, end, "\n".join(rows[i+1:])))
    return lines


def normalize(text: str) -> list:
    """Line text -> list of words comparable between subtitle sets."""
    t = _TAGS.sub(" ", text)
    t = _BRACKETS.sub(" ", t)
    t = " ".join(_SPEAKER.sub("", part) for part in t.split("\n"))
    t = _SPEAKER.sub("", t)
    t = t.lower().replace("’", "'")
    return _NON_ALNUM.sub(" ", t).split()


def word_stream(lines: list) -> list:
    """Lines -> [(word, time, line_no, is_first_word_of_line)]."""
    out = []
    for no, (start, end, text) in enumerate(lines):
        words = normalize(text)
        if not words:
            continue
        d = max(end - start, 1e-3)
        for i, w in enumerate(words):
            out.append((w, start + d*(i + 0.5)/len(words), no, i == 0))
    return out


# ----------------------------------------------------------------- anchoring

def anchors(ep_lines, ep_stream, clip_lines, clip_stream, min_block=MIN_BLOCK):
    """Returns (exact, interpolated) -- lists of pairs (clip_time, episode_time)."""
    a = [x[0] for x in ep_stream]
    b = [x[0] for x in clip_stream]
    matches = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    exact, interp = [], []
    for i, j, n in matches:
        if n < min_block:
            continue
        for k in range(n):
            e, c = ep_stream[i+k], clip_stream[j+k]
            interp.append((c[1], e[1]))
            if e[3] and c[3]:
                exact.append((clip_lines[c[2]][0], ep_lines[e[2]][0]))
    return exact, interp


def offset(pairs, tolerance=TOLERANCE):
    """Robust median of time differences -> (offset, spread, anchor_count).

    ``None`` when there are too few anchors. The spread is the range of the
    differences regarded as consistent -- how much the two subtitle sets differ in
    the timestamps themselves.
    """
    if len(pairs) < 3:
        return None
    diffs = sorted(y - x for x, y in pairs)
    best = max(([d for d in diffs if abs(d - k) <= tolerance] for k in diffs),
               key=len)
    b = statistics.median(best)
    consistent = [d for d in diffs if abs(d - b) <= tolerance]
    b = statistics.median(consistent)
    return b, max(consistent) - min(consistent), len(consistent)


def episode_scale(global_pairs):
    """Line ``t_ep = a*t + b`` fitted on the anchors of a whole episode.

    Only a check of the assumption that both axes run at the same speed. Returns
    ``(a, b)``.
    """
    n = len(global_pairs)
    sx = sum(x for x, _ in global_pairs)
    sy = sum(y for _, y in global_pairs)
    sxx = sum(x*x for x, _ in global_pairs)
    sxy = sum(x*y for x, y in global_pairs)
    denom = n*sxx - sx*sx
    a = (n*sxy - sx*sy) / denom if denom else 1.0
    return a, (sy - a*sx) / n


def detect_jump(exact_anchors, min_jump=MIN_JUMP, min_anchors_side=3):
    """Looks for a single point inside a clip at which the offset jumps.

    Our recording and the TVQA source may differ in the length of a transition
    (ours is about 3 s shorter at a chapter boundary), so a clip spanning such a
    boundary has two offsets. Splits are searched over the anchors ordered by clip
    time -- a single outlying anchor then cannot break the split -- and the one
    with the largest gap between medians relative to the spread wins. ``None`` when
    there is no jump.
    """
    d = sorted(exact_anchors)
    if len(d) < 2*min_anchors_side:
        return None
    diffs = [y - x for x, y in d]
    best = None
    for i in range(min_anchors_side, len(d) - min_anchors_side + 1):
        a, b = diffs[:i], diffs[i:]
        ma, mb = statistics.median(a), statistics.median(b)
        if abs(mb - ma) < min_jump:
            continue
        spread = (max(abs(x-ma) for x in a) + max(abs(x-mb) for x in b)) / 2
        quality = abs(mb - ma) - spread
        if best is None or quality > best[0]:
            best = (quality, i, ma, mb)
    if best is None or best[0] <= 0:
        return None
    _, i, ma, mb = best
    return {"offset_before": ma, "offset_after": mb, "jump": mb - ma,
            "t_before": d[i-1][0], "t_after": d[i][0],
            "anchors_before": i, "anchors_after": len(d) - i}


def second_grid(offsets: list) -> dict:
    """Concentration of the offsets on a 1 s grid (directional statistic).

    Returns ``delta`` (the difference between the starts of the TVQA recording and
    ours), the concentration ``R`` (1 = perfect) and the largest residual.
    """
    if not offsets:
        return {"delta": float("nan"), "R": 0.0, "residual_max": float("nan")}
    angles = [2*math.pi*(x % 1.0) for x in offsets]
    S = sum(math.sin(a) for a in angles) / len(angles)
    C = sum(math.cos(a) for a in angles) / len(angles)
    # delta brought into (-0.5; 0.5], so that a value right next to zero does
    # not land on the other side of the period as 0.999
    delta = ((math.atan2(S, C) / (2*math.pi)) + 0.5) % 1.0 - 0.5
    residuals = [abs(((x - delta + 0.5) % 1.0) - 0.5) for x in offsets]
    return {"delta": delta, "R": math.hypot(S, C), "residual_max": max(residuals)}


# ------------------------------------------------------------------- episode

def _clip_key(name: str):
    parts = name.split("_")
    return (parts[1], int(parts[-1]))


def map_episode(episode_subs, clips_dir, episode, tolerance=TOLERANCE,
                min_block=MIN_BLOCK, snap=True) -> dict:
    """Offset of every clip of an episode against the full recording.

    Returns ``{"episode", "scale", "grid", "clips": [row, ...]}``; a row holds the
    offset and the match quality measures. With ``snap`` on and a high enough grid
    concentration the offsets are snapped to the grid (column ``grid`` = yes).
    """
    ep_lines = read_srt(episode_subs)
    ep_stream = word_stream(ep_lines)
    files = sorted(Path(clips_dir).glob(f"{episode}_seg*_clip_*.srt"),
                   key=lambda p: _clip_key(p.stem))
    if not files:
        raise FileNotFoundError(f"No clip subtitles for {episode} in {clips_dir}")

    rows, global_pairs = [], []
    for file in files:
        clip_lines = read_srt(file)
        clip_stream = word_stream(clip_lines)
        exact, interp = anchors(ep_lines, ep_stream, clip_lines, clip_stream, min_block)
        result_i = offset(interp, tolerance)
        row = {"vid_name": file.stem, "episode": episode,
               "segment": file.stem.split("_")[1],
               "clip_no": int(file.stem.split("_")[-1]),
               "clip_words": len(clip_stream)}
        if result_i is None:
            row.update({"offset": None, "offset_raw": None,
                        "source": "none", "grid": "no", "anchors": 0,
                        "anchors_interp": len(interp), "spread": None,
                        "word_share": 0.0, "episode_start": None, "episode_end": None,
                        "jump": None, "jump_start": None, "jump_end": None,
                        "offset_before": None, "offset_after": None})
            rows.append(row)
            continue
        result_e = offset(exact, tolerance)
        if result_e is not None and result_e[2] >= 3:
            b, spread, n_anch = result_e
            source = "lines"
        else:
            b, spread, n_anch = result_i
            source = "interpolation"
        global_pairs += [(x + b, y) for x, y in interp if abs((y - x) - b) <= tolerance]
        jp = detect_jump(exact)
        if jp is None:
            row.update({"jump": None, "jump_start": None, "jump_end": None,
                        "offset_before": None, "offset_after": None})
        else:
            before = [y - x for x, y in exact if x <= jp["t_before"]]
            after = [y - x for x, y in exact if x >= jp["t_after"]]
            row.update({"jump": _round(jp["jump"]),
                        "jump_start": _round(jp["t_before"]), "jump_end": _round(jp["t_after"]),
                        "offset_before": _round(statistics.median(before)),
                        "offset_after": _round(statistics.median(after))})
        row.update({"offset": b, "offset_raw": _round(b),
                    "source": source, "grid": "no", "anchors": n_anch,
                    "anchors_interp": result_i[2], "spread": _round(spread),
                    "word_share": round(result_i[2] / max(len(clip_stream), 1), 3),
                    "_clip_end": clip_lines[-1][1] if clip_lines else 0.0})
        rows.append(row)

    a = episode_scale(global_pairs)[0] if len(global_pairs) > 10 else float("nan")
    ok = [r for r in rows if r["offset"] is not None]
    grid = second_grid([r["offset"] for r in ok])
    snapped = (snap and grid["R"] >= MIN_CONCENTRATION
               and grid["residual_max"] <= MAX_GRID_RESIDUAL)
    for r in ok:
        if snapped:
            r["offset"] = _round(round(r["offset"] - grid["delta"]) + grid["delta"])
            r["grid"] = "yes"
        else:
            r["offset"] = _round(r["offset"])
        r["episode_start"] = r["offset"]
        r["episode_end"] = _round(r["offset"] + r.pop("_clip_end"))
    for r in rows:
        r["episode_scale"] = _round(a, 6)
        r["grid_R"] = _round(grid["R"])
        r["grid_delta"] = _round(grid["delta"])
    return {"episode": episode, "scale": a, "grid": grid, "clips": rows}


def subtitle_path(subs_dir, episode: str, series: str = "tbbt") -> Path:
    """Episode subtitles, named like the video file: ``tbbt_s01e15.srt``."""
    return Path(subs_dir) / f"{series}_{episode}.srt"


def map_episodes(subs_dir, clips_dir, episodes, series: str = "tbbt", **kw) -> list:
    """:func:`map_episode` for a list of episodes."""
    out = []
    for ep in episodes:
        file = subtitle_path(subs_dir, ep, series)
        if not file.exists():
            raise FileNotFoundError(f"No episode subtitles: {file}")
        out.append(map_episode(file, clips_dir, ep, **kw))
    return out


def flat_rows(results: list) -> list:
    """Results for many episodes -> one list of rows for a CSV."""
    return [c for r in results for c in r["clips"]]


def apply_corrections(results: list, corrections: dict) -> int:
    """Manually verified offset correction for a whole episode [s].

    When the subtitles of an episode are shifted against the picture, the offset
    computed from them agrees with the subtitles but not with what is seen -- and
    that cannot be detected from the subtitles alone. ``corrections`` is
    ``{"s09e12": -2.0}``: that many seconds are added to every offset of the
    episode. Returns the number of corrected clips.
    """
    n = 0
    for r in results:
        d = corrections.get(r["episode"], 0.0)
        for c in r["clips"]:
            if d:
                for field in ("offset", "episode_start", "episode_end",
                              "offset_before", "offset_after"):
                    if c.get(field) is not None:
                        c[field] = _round(c[field] + d)
                n += 1
            c["correction"] = d          # the column exists in all rows
    return n


# --------------------------------------------------------------------- check

def check(results: list, max_spread=2.0, max_drift=0.5, max_overlap=2.0) -> dict:
    """Correctness checks of the mapping -> dict with notes and counters.

    Checks: the episode scale is close to 1; clips in a segment are ordered by
    their numbering; the subtitles of the next clip do not overlap the previous
    one; anchors are numerous enough and do not drift apart.
    """
    notes = []
    for r in results:
        ep, a = r["episode"], r["scale"]
        drift = abs(a - 1.0) * 1400.0        # drift over the episode length [s]
        if a != a or drift > max_drift:
            notes.append((ep, "-", f"scale {a:.6f}, drift {drift:.2f} s per episode"))
        ok = [c for c in r["clips"] if c["offset"] is not None]
        for c in r["clips"]:
            if c["offset"] is None:
                notes.append((ep, c["vid_name"], "no match"))
            elif c["anchors"] < MIN_ANCHORS:
                notes.append((ep, c["vid_name"], f"only {c['anchors']} anchors"))
            elif c.get("jump"):
                notes.append((ep, c["vid_name"],
                              f"offset jump {c['jump']:+.2f} s inside the clip "
                              f"(t_clip {c['jump_start']:.1f}-{c['jump_end']:.1f}) "
                              f"-> mapping by parts"))
            elif c["grid"] == "no" and c["spread"] > max_spread:
                notes.append((ep, c["vid_name"], f"anchor spread {c['spread']:.2f} s"))
        for seg in sorted({c["segment"] for c in ok}):
            cs = sorted((c for c in ok if c["segment"] == seg), key=lambda c: c["clip_no"])
            for p, n in zip(cs, cs[1:]):
                if n["episode_start"] < p["episode_start"]:
                    notes.append((ep, n["vid_name"], "order inconsistent with clip numbering"))
                elif (n["clip_no"] - p["clip_no"] == 1
                      and p["episode_end"] - n["episode_start"] > max_overlap):
                    notes.append((ep, n["vid_name"],
                                  f"subtitles overlap the previous clip by "
                                  f"{p['episode_end'] - n['episode_start']:.1f} s"))
    return {"notes": notes,
            "clips": sum(len(r["clips"]) for r in results),
            "mapped": sum(1 for r in results for c in r["clips"]
                          if c["offset"] is not None),
            "on_grid": sum(1 for r in results for c in r["clips"] if c["grid"] == "yes")}


def format_summary(results: list, checks: dict) -> str:
    """Table episode by episode + the verdict of the checks."""
    lines = []
    header = (f"{'episode':<9}{'clips':>7}{'scale':>10}{'grid_R':>10}{'resid':>8}"
              f"{'anchors':>8}{'spread':>9}{'gap':>9}{'coverage':>16}")
    lines += [header, "-" * len(header)]
    for r in results:
        ok = [c for c in r["clips"] if c["offset"] is not None]
        anch = sorted(c["anchors"] for c in ok)
        spr = sorted(c["spread"] for c in ok)
        s1 = [c for c in ok if c["segment"] == "seg01"]
        s2 = [c for c in ok if c["segment"] == "seg02"]
        gap = (min(c["episode_start"] for c in s2) - max(c["episode_end"] for c in s1)) \
            if (s1 and s2) else float("nan")
        coverage = (f"{min(c['episode_start'] for c in ok):.1f}-"
                    f"{max(c['episode_end'] for c in ok):.1f}")
        lines.append(f"{r['episode']:<9}{len(r['clips']):>7}{r['scale']:>10.5f}"
                     f"{r['grid']['R']:>10.3f}{r['grid']['residual_max']:>8.3f}"
                     f"{statistics.median(anch):>8.0f}{spr[len(spr)//2]:>9.2f}"
                     f"{gap:>9.1f}{coverage:>16}")
    lines.append("")
    lines.append(f"mapped {checks['mapped']} of {checks['clips']} clips; "
                 f"snapped to the second grid: {checks['on_grid']}")
    if not checks["notes"]:
        lines.append("OK: no check notes")
    else:
        lines.append(f"NOTES ({len(checks['notes'])}) - for manual verification:")
        for ep, clip, text in checks["notes"]:
            lines.append(f"  {ep}  {clip:<26}{text}")
    return "\n".join(lines)


def check_by_text(episode_subs, clips_dir, rows, window=4.0, min_words=5) -> dict:
    """A check independent of the anchors: does the text of a clip line occur in
    the episode subtitles within +-``window`` seconds of the mapped time.

    All lines are checked, not only the anchors. Returns the error distribution and
    the number of lines not found (usually differently worded subtitles, not a
    mapping error).
    """
    ep_lines = read_srt(episode_subs)
    errors, not_found, checked = [], 0, 0
    for r in rows:
        if r["offset"] is None:
            continue
        for start, _, text in read_srt(Path(clips_dir) / f"{r['vid_name']}.srt"):
            words = normalize(text)
            if len(words) < min_words:
                continue
            checked += 1
            target = start + r["offset"]
            frag = " ".join(words[:min_words])
            hits = [q for q in ep_lines if abs(q[0] - target) <= window
                    and frag in " ".join(normalize(q[2]))]
            if hits:
                errors.append(abs(hits[0][0] - target))
            else:
                not_found += 1
    e = sorted(errors)
    return {"checked": checked, "found": len(e), "not_found": not_found,
            "error_med": statistics.median(e) if e else float("nan"),
            "error_p95": e[int(0.95*(len(e)-1))] if e else float("nan"),
            "error_max": e[-1] if e else float("nan")}


# -------------------------------------------------------------------- saving

def save_csv(rows: list, path, columns=None) -> Path:
    path = Path(path)
    if not rows and not columns:
        # without this an empty list would wipe an earlier, correct file
        raise ValueError(f"nothing to save to {path.name} - empty list of rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns or list(rows[0].keys()),
                           delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def to_episode_scale(ts, offset_s) -> tuple:
    """``[start, end]`` on the clip scale -> ``(start, end)`` on the episode scale.

    Clipped at zero: the offset of the first clip is sometimes -0.001 s (grid
    rounding), and a negative timestamp makes no sense.
    """
    return (_round(max(ts[0] + offset_s, 0.0)),
            _round(max(ts[1] + offset_s, 0.0)))


def map_ts(ts, row) -> tuple:
    """``(start, end)`` on the clip scale -> ``(start, end, status)`` on the episode
    scale.

    For a clip with a detected jump the offset of the part the interval lies in is
    used. Status: ``ok`` (no jump), ``before`` / ``after`` (jump, interval clearly
    on one side), ``uncertain`` (interval crosses the jump window), ``none`` (clip
    without an offset).
    """
    b = row.get("offset")
    if b is None:
        return None, None, "none"
    jump = row.get("jump")
    if not jump:
        return (*to_episode_scale(ts, b), "ok")
    if ts[1] <= row["jump_start"]:
        return (*to_episode_scale(ts, row["offset_before"]), "before")
    if ts[0] >= row["jump_end"]:
        return (*to_episode_scale(ts, row["offset_after"]), "after")
    return (*to_episode_scale(ts, b), "uncertain")
