"""The loop every per-episode component cache shares.

All the heavy components produce one file per episode and are resumable per
episode. They also share the reasons a file may fail to appear -- no recording on
disk yet, an episode outside the split -- and none of those may abort a run,
because the corpus is annotated while the material is still being completed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from src.data import datasets
from src.utils.progress import progress


def ensure(
    dataset: str,
    episodes: dict[str, dict],
    target: Callable[[str], Path],
    compute: Callable[[str, Path, Path], str],
    label: str,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Guarantees the cache file of every episode; returns the ones that exist.

    ``compute(episode, video_path, target_path)`` is called for the missing ones and
    whatever it returns is logged. It writes to a temporary file renamed into place
    only on success, so the presence of a cache file means the episode is FINISHED --
    an extraction stopped halfway would otherwise leave a half-written file that
    every later run treats as complete.
    """
    missing = {ep: entry for ep, entry in sorted(episodes.items())
               if force or not target(ep).exists()}
    if missing:
        log(f"{dataset}: {label} for {len(missing)} episodes")

    done = []
    report = progress(log, total=len(missing))
    for episode, entry in sorted(episodes.items()):
        path = target(episode)
        if episode not in missing:
            done.append(episode)
            continue
        video = datasets.video_path(dataset, entry["video_file"])
        if not video.exists():
            log(f"  {episode}: no video file ({video.name}) - skipped")
            continue
        started = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        # the marker goes BEFORE the extension: numpy appends ".npz" to any name not
        # ending in it, so "x.npz.partial" would be written as "x.npz.partial.npz"
        # and the rename would find nothing
        partial = path.with_name(f"{path.stem}.partial{path.suffix}")
        partial.unlink(missing_ok=True)
        summary = compute(episode, video, partial)
        partial.replace(path)
        report(f"{episode}: {summary}  ({time.perf_counter() - started:.0f} s)")
        done.append(episode)
    return done
