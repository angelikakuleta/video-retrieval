"""Building the segment cache -- the single implementation behind both the
standalone script (``scripts/run_segmentation.py``) and the experiment runner
(``src.runners.stages``), so the logic exists exactly once.

The cache is split-agnostic: an episode's fragments do not depend on the
split it belongs to, so one CSV per (strategy, dataset) carries a ``split``
column and the split becomes a filter applied downstream.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.data import datasets
from src.data.ranges import load_ranges, load_timelines
from src.segmentation import segments as seg
from src.segmentation.timeline import Timeline
from src.utils.config import SegmentationConfig
from src.utils.progress import progress

#: the strategy of a collection whose recordings are already the fragments.
#: Not a value of `SegmentationConfig.strategy`: there is nothing to configure,
#: which is why a VATEX configuration carries no segmentation section at all.
WHOLE_CLIP = "whole_clip"


def ensure_segments(
    dataset: str,
    strategies: dict[str, SegmentationConfig],
    episodes: list[str] | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, list[dict]]:
    """Guarantees the cache for the given strategies; computes only what is missing.

    Returns the cache rows per strategy (all episodes present in the cache,
    not only the ones computed now). Strategies sharing a run also share one
    decoding pass per episode. Length-correction parameters must agree across
    the given strategies -- they define one comparable family of collections.
    """
    if WHOLE_CLIP in strategies:
        if len(strategies) > 1:
            raise ValueError(f"{WHOLE_CLIP!r} is the only strategy of its collection, "
                             f"got {sorted(strategies)}")
        return {WHOLE_CLIP: _whole_clips(dataset, episodes, force, log)}

    lengths = {(s.min_len_s, s.max_len_s) for s in strategies.values()}
    if len(lengths) > 1:
        raise ValueError("min/max fragment length differs between the strategies")
    (min_len, max_len), = lengths

    ranges = load_ranges(dataset)
    timelines = load_timelines(dataset, ranges)
    wanted = sorted(set(episodes or ranges) & set(ranges))

    cache: dict[str, list[dict]] = {}
    done: dict[str, set[str]] = {}
    for strategy in strategies:
        path = datasets.segments_csv(strategy, dataset)
        cache[strategy] = seg.load(path) if path.exists() else []
        done[strategy] = {r["episode"] for r in cache[strategy]}

    log(f"{dataset}: {len(wanted)} episodes, strategies: {', '.join(strategies)}")
    for episode in wanted:
        todo = [s for s in strategies if force or episode not in done[s]]
        if not todo:
            continue
        entry = ranges[episode]
        timeline = timelines[episode]
        started = time.perf_counter()
        spans_by_strategy = _compute_episode(dataset, entry, timeline, strategies,
                                             todo, min_len, max_len, log)
        counts = []
        for strategy, spans in spans_by_strategy.items():
            cache[strategy] = [r for r in cache[strategy] if r["episode"] != episode]
            cache[strategy] += seg.rows(episode, entry["split"], entry["video_file"],
                                        spans, timeline)
            done[strategy].add(episode)
            seg.save(cache[strategy], datasets.segments_csv(strategy, dataset))
            counts.append(f"{strategy}: {len(spans)}")
        if counts:
            log(f"  {episode}: {'; '.join(counts)}  ({time.perf_counter() - started:.1f} s)")

    for strategy in strategies:
        path = datasets.segments_csv(strategy, dataset)
        log(f"cache -> {path.relative_to(datasets.ROOT)} ({len(done[strategy])} episodes)")
    return cache


def _whole_clips(
    dataset: str,
    episodes: list[str] | None,
    force: bool,
    log: Callable[[str], None],
) -> list[dict]:
    """One fragment per recording, straight from its range.

    A clip IS the excerpt its annotation describes, so there is nothing to cut
    and no length to correct. Taking the shot path instead would open a decoder
    for each of a few thousand files to write a single row each, and the caller
    writes the CSV once at the end rather than after every one of them.
    """
    path = datasets.segments_csv(WHOLE_CLIP, dataset)
    cache = seg.load(path) if path.exists() else []
    done = {r["episode"] for r in cache}

    ranges = load_ranges(dataset)
    timelines = load_timelines(dataset, ranges)
    wanted = sorted(set(episodes or ranges) & set(ranges))
    todo = [ep for ep in wanted if force or ep not in done]
    log(f"{dataset}: {len(wanted)} recordings, strategy: {WHOLE_CLIP}")

    if todo:
        report = progress(log, total=len(todo))
        rebuilt = [r for r in cache if r["episode"] not in set(todo)]
        for episode in todo:
            entry = ranges[episode]
            rebuilt += seg.rows(episode, entry["split"], entry["video_file"],
                                entry["spans"], timelines[episode])
            report(f"{episode}: {len(entry['spans'])} fragment(s)")
        cache = rebuilt
        seg.save(cache, path)
    log(f"cache -> {path.relative_to(datasets.ROOT)} "
        f"({len({r['episode'] for r in cache})} recordings)")
    return cache


def _compute_episode(
    dataset: str,
    entry: dict,
    timeline: Timeline,
    strategies: dict[str, SegmentationConfig],
    todo: list[str],
    min_len: float,
    max_len: float,
    log: Callable[[str], None],
) -> dict[str, list[tuple[float, float]]]:
    """Fragments of one episode for the requested strategies (one decode pass)."""
    out: dict[str, list[tuple[float, float]]] = {}

    if "fixed_window" in todo:
        out["fixed_window"] = seg.fixed_windows(
            timeline, strategies["fixed_window"].window_s, min_len, max_len)

    shot_strategies = [s for s in todo if s != "fixed_window"]
    if not shot_strategies:
        return out
    video = datasets.video_path(dataset, entry["video_file"])
    if not video.exists():
        log(f"  {entry['video_file']}: no video file - shot strategies skipped")
        return out

    from src.segmentation import decode, histogram, transnet

    need_tiny = "shots_transnetv2" in shot_strategies
    need_small = "shots_histogram" in shot_strategies
    stream = decode.FrameStream(video, small=need_small, tiny=need_tiny)
    detector = None
    if need_small:
        threshold = strategies["shots_histogram"].histogram_threshold or histogram.THRESHOLD
        detector = histogram.HistogramDetector(stream.fps, threshold)
    tiny_frames = []
    for index, small, tiny in stream:
        if detector is not None:
            detector.feed(index, small)
        if need_tiny:
            tiny_frames.append(tiny)

    if detector is not None:
        out["shots_histogram"] = seg.from_boundaries(
            detector.boundaries, timeline, min_len, max_len)
    if need_tiny:
        predictions = transnet.frame_predictions(decode.tiny_buffer(tiny_frames))
        threshold = strategies["shots_transnetv2"].transnet_threshold or transnet.THRESHOLD
        cuts = transnet.boundaries_from_predictions(predictions, stream.fps, threshold)
        out["shots_transnetv2"] = seg.from_boundaries(
            cuts, timeline, min_len, max_len)
    return out
