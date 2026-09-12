"""Cache-or-compute stages of a run: black stretches -> segments -> frame
embeddings -> collection with a FAISS index.

Everything here is derived from the recordings and the annotations by a fixed
rule, so a run needs no preparation beyond its INPUT (see
notebooks/experiments/README.md). Whatever is missing is computed and cached on
the way, and only the episodes of the evaluated split are touched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.data import datasets, manifest
from src.data.ranges import (load_ranges, load_timelines, split_episodes,
                             vatex_range_files)
from src.evaluation.relevance import fragment_id
from src.features import frame_cache
from src.features.scene import build_clip_embedding
from src.segmentation import black
from src.segmentation.build import ensure_segments
from src.segmentation.timeline import Timeline
from src.utils.config import ExperimentConfig


#: patch features of every fragment, stored next to the index they belong to
PROMPT_FEATURES_FILE = "prompt_features.npy"


def scene_signal(config: ExperimentConfig, collection, encoder):
    """The scene signal of the configured base representation.

    Matching a query against a stored vector is the same operation whichever encoder
    produced it, so both contrastive representations and X-CLIP without prompting
    take that path. Only the prompted variant reads the patch features saved beside
    the index. Both X-CLIP variants share one index: the vision pass that fills it
    produces the vectors and the patch features together.
    """
    from src.retrieval.signals import SceneSignal, XClipSceneSignal

    scene = config.components.scene_embedding
    if not scene.prompting:
        return SceneSignal(collection, encoder)
    assert config.split is not None
    path = (datasets.index_dir(config.dataset, config.collection_strategy,
                               scene.model, config.split) / PROMPT_FEATURES_FILE)
    return XClipSceneSignal(collection, encoder, np.load(path).astype(np.float32))


def ensure_collection(
    config: ExperimentConfig,
    encoder_factory: Callable[[], Any],
    force: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[Any, dict, dict]:
    """The FAISS collection of the evaluated split, building it if needed.

    Returns ``(collection, inputs, context)``: the inputs name the files the
    collection depends on (for the run's metadata), the context carries the
    fragments, episodes and timelines the components need.
    """
    from src.indexing import faiss_index

    assert config.split is not None

    strategy = config.collection_strategy
    model = config.components.scene_embedding.model
    segments_path = datasets.segments_csv(strategy, config.dataset)
    index_path = datasets.index_dir(config.dataset, strategy, model, config.split)

    ranges = load_ranges(config.dataset)
    wanted = sorted(split_episodes(ranges, config.split))
    if not wanted:
        raise RuntimeError(f"no episode belongs to split {config.split!r} - annotate first")

    # 1. Black stretches (cache-or-compute). They decide which frames a component
    #    may see, so they are guaranteed before the first Timeline: a missing file
    #    would silently mean "no black frames".
    #    The recordings are named from the ranges rather than looked up by name:
    #    a VATEX clip is not called sXXeYY and does not live under data/processed,
    #    so the pattern that finds the series episodes finds none of them.
    black.ensure_black(config.dataset, episodes=wanted, log=log,
                       videos={ep: datasets.video_path(config.dataset,
                                                       ranges[ep]["video_file"])
                               for ep in wanted})

    # 2. Segments of those episodes (cache-or-compute).
    rows = ensure_segments(config.dataset, {strategy: config.segmentation},
                           episodes=wanted, log=log)[strategy]
    rows = [r for r in rows if r["split"] == config.split]
    if not rows:
        raise RuntimeError(f"no segments for split {config.split!r} - annotate first")
    segments_sha = manifest.file_entry(segments_path)["sha256"]

    # what the collection depends on. VATEX has no ranges file and needs none:
    # the query files ARE the ranges, so they are what the manifest records.
    inputs = {"segments": segments_path,
              "ranges": (vatex_range_files() if config.dataset == "vatex"
                         else datasets.ranges_csv(config.dataset))}

    # 3. A ready index is reused only when built from the SAME segments.
    complete = model != "xclip_b32" or (index_path / PROMPT_FEATURES_FILE).exists()
    if not force and complete and faiss_index.exists(index_path):
        collection = faiss_index.load(index_path)
        import json
        catalog = json.loads((index_path / faiss_index.CATALOG_FILE).read_text(encoding="utf-8"))
        if catalog.get("segments_sha256") == segments_sha:
            log(f"index reused -> {index_path.relative_to(datasets.ROOT)}"
                f" ({collection.size} fragments)")
            return collection, inputs, _context(config, ranges, rows)
        log("index outdated (segments changed) - rebuilding")

    context = _context(config, ranges, rows)
    episodes, timelines = context["episodes"], context["timelines"]
    if model in frame_cache.OPENCLIP_MODELS:
        # 4. Frame embeddings once per (model, episode), shared by all strategies.
        frame_cache.ensure_frame_embeddings(
            config.dataset, episodes, model, config.frames.step_s,
            encoder_factory=encoder_factory, log=log)
        # 5. Aggregation: fragment embedding = dedup + mean of its grid frames.
        matrix, vids = fragment_matrix(config.dataset, rows, model,
                                       config.frames.dedup_cosine, timelines, log=log)
    elif model == "xclip_b32":
        # X-CLIP embeds the fragment directly (64-frame windows, averaged),
        # so its collection depends on the segmentation and is built here. The
        # patch features travel with it: the query phase needs them to condition
        # the prompt module on the fragment being scored.
        matrix, vids, prompts = fragment_matrix_xclip(
            config.dataset, rows, episodes, encoder_factory(), timelines, log=log)
    else:
        raise ValueError(f"unknown scene model: {model!r}")
    collection = faiss_index.build_index(matrix, vids)
    faiss_index.save(collection, index_path, metadata={
        "dataset": config.dataset, "strategy": strategy, "model": model,
        "split": config.split, "segments_sha256": segments_sha,
        "step_s": config.frames.step_s, "dedup_cosine": config.frames.dedup_cosine,
    })
    if model == "xclip_b32":
        np.save(index_path / PROMPT_FEATURES_FILE, prompts.astype(np.float16))
    log(f"index built -> {index_path.relative_to(datasets.ROOT)}"
        f" ({collection.size} fragments)")
    return collection, inputs, context


def _context(config: ExperimentConfig, ranges: dict, rows: list[dict]) -> dict:
    """The fragments, episodes and timelines the components work against."""
    episodes = {ep: e for ep, e in split_episodes(ranges, config.split).items()
                if ep in {r["episode"] for r in rows}}
    # one axis, read once: the same object decides which frames every component
    # is allowed to see (src.segmentation.timeline)
    return {"rows": rows, "episodes": episodes,
            "timelines": load_timelines(config.dataset, ranges)}


def fragment_matrix_xclip(
    dataset: str,
    rows: list[dict],
    episodes: dict[str, dict],
    encoder,
    timelines: dict[str, Timeline] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """X-CLIP collection: one decode pass and one encoder pass per episode.

    Returns ``(matrix, vids, prompt_features)``. A fragment longer than one window
    gets the mean over its windows, for both outputs.
    """
    import time

    from src.features import xclip

    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)

    vectors, vids, prompts = [], [], []
    for episode in sorted(by_episode):
        video = datasets.video_path(dataset, episodes[episode]["video_file"])
        if not video.exists():
            log(f"  {episode}: no video file - fragments skipped")
            continue
        started = time.perf_counter()
        import cv2

        reader = cv2.VideoCapture(str(video))
        fps = reader.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        reader.release()
        if fps <= 0 or frame_count <= 0:
            log(f"  {episode}: could not read the video - fragments skipped")
            continue

        fragments = sorted(by_episode[episode], key=lambda r: r["start"])
        per_fragment, needed = xclip.fragment_windows_for_episode(
            fragments, fps, frame_count, (timelines or {}).get(episode))
        frames = xclip.collect_frames(video, needed)

        windows_flat, spans = [], []
        for windows in per_fragment:
            spans.append((len(windows_flat), len(windows_flat) + len(windows)))
            windows_flat += [[frames[i] for i in window] for window in windows]
        embeddings, patches = encoder.encode_windows_with_prompts(windows_flat)
        for row, (a, b) in zip(fragments, spans):
            mean = embeddings[a:b].mean(axis=0)
            norm = np.linalg.norm(mean)
            vectors.append((mean / norm).astype(np.float32) if norm > 0 else mean)
            prompts.append(patches[a:b].mean(axis=0).astype(np.float32))
            vids.append(fragment_id(row["episode"], row["segment_id"]))
        log(f"  {episode}: {len(fragments)} fragments, {len(windows_flat)} windows"
            f"  ({time.perf_counter() - started:.0f} s)")
    if not vectors:
        raise RuntimeError("no fragment got an X-CLIP embedding")
    return (np.stack(vectors).astype(np.float32), vids,
            np.stack(prompts).astype(np.float32))


def fragment_matrix(
    dataset: str,
    rows: list[dict],
    model: str,
    dedup_cosine: float,
    timelines: dict[str, Timeline] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, list[str]]:
    """Builds the (n_fragments, dim) matrix from the per-episode frame caches.

    The cache covers the whole file, so a fragment picks its frames by time and the
    timeline narrows the selection BEFORE deduplication. The order matters: a black
    frame reaching deduplication first would become the reference the next frame is
    compared against, and a static shot would then contribute two copies of the same
    picture instead of one.
    """
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)

    vectors, vids = [], []
    for episode in sorted(by_episode):
        try:
            times, embeddings = frame_cache.load_episode(model, dataset, episode)
        except FileNotFoundError:
            log(f"  {episode}: no frame cache - fragments skipped")
            continue
        timeline = (timelines or {}).get(episode)
        for row in sorted(by_episode[episode], key=lambda r: r["start"]):
            mask = (times >= row["start"]) & (times < row["end"])
            if timeline is not None:
                usable = mask & np.asarray(timeline.select(times), dtype=bool)
                # an all-black fragment keeps its frames: an empty
                # representation would be worse than a black one
                mask = usable if usable.any() else mask
            selected = embeddings[mask]
            if not len(selected):   # cannot happen for fragments >= 2 * grid step
                nearest = int(np.argmin(np.abs(times - row["start"])))
                selected = embeddings[nearest:nearest + 1]
            vectors.append(build_clip_embedding(selected, dedup_cosine))
            vids.append(fragment_id(row["episode"], row["segment_id"]))
    if not vectors:
        raise RuntimeError("no fragment got an embedding - is the frame cache built?")
    return np.stack(vectors).astype(np.float32), vids
