"""Frame embeddings on the common time grid, cached per episode.

The same 1.25 s grid feeds the single-frame components regardless of the
segmentation strategy, so embeddings are computed ONCE per (model, episode) and
every strategy's collection is built by aggregation alone.

The grid covers the WHOLE file, not just the corpus ranges: a later edit of the
intro or credits masks then never invalidates this most expensive artifact, since
fragments simply select frames by time.

One ``.npz`` per episode holds ``times`` (seconds) and ``embeddings``
(n, dim; L2-normalized float32).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data import datasets
from src.utils.progress import progress

#: the two contrastive base representations of chapter 5. ViT-B/32 carries the
#: ``openai`` weights on purpose: as the control variant of E2 it must be the
#: encoder X-CLIP base/32 was initialized from, otherwise the split into "size of
#: the encoder" and "type of representation" compares two unrelated models.
OPENCLIP_MODELS = {
    "openclip_vit_h14": ("ViT-H-14", "laion2b_s32b_b79k"),
    "openclip_vit_b32": ("ViT-B-32", "openai"),
}

#: grid frames decoded and encoded at once
BATCH = 32


def episode_npz(model: str, dataset: str, episode: str) -> Path:
    return datasets.frame_cache_dir(model, dataset) / f"{episode}.npz"


def load_episode(model: str, dataset: str, episode: str) -> tuple[np.ndarray, np.ndarray]:
    """``(times, embeddings)`` of one cached episode."""
    with np.load(episode_npz(model, dataset, episode)) as data:
        return data["times"], data["embeddings"]


def ensure_frame_embeddings(
    dataset: str,
    episodes: dict[str, dict],
    model: str,
    step: float,
    encoder_factory: Callable[[], object] | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Computes the missing per-episode caches for the given model and grid.

    The encoder is created lazily, so a run whose episodes are all cached loads no
    model at all.
    """
    if model == "xclip_b32":
        raise NotImplementedError("the X-CLIP base representation arrives with E2")
    if model not in OPENCLIP_MODELS:
        raise ValueError(f"unknown scene model: {model!r}")

    missing = {ep: entry for ep, entry in episodes.items()
               if not episode_npz(model, dataset, ep).exists()}
    if not missing:
        return

    import time

    from src.features.openclip import ClipEncoder
    from src.segmentation.frames import iter_batches

    name, weights = OPENCLIP_MODELS[model]
    log(f"{dataset}: frame embeddings for {len(missing)} episodes ({name}/{weights})")
    encoder = encoder_factory() if encoder_factory is not None \
        else ClipEncoder(model=name, weights=weights)
    report = progress(log, total=len(missing))
    for episode, entry in sorted(missing.items()):
        video = datasets.video_path(dataset, entry["video_file"])
        if not video.exists():
            log(f"  {episode}: no video file ({video.name}) - skipped")
            continue
        started = time.perf_counter()
        # in batches: an episode holds around a thousand grid frames, and keeping
        # them all decoded at once would cost gigabytes for no gain
        times: list[float] = []
        chunks = []
        for batch_times, batch_frames in iter_batches(video, step, BATCH):
            chunks.append(encoder.encode_images(batch_frames))
            times += batch_times
        embeddings = np.concatenate(chunks) if chunks else np.zeros((0, 1), np.float32)
        target = episode_npz(model, dataset, episode)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            times=np.asarray(times, dtype=np.float32),
            embeddings=embeddings.astype(np.float32))
        report(f"{episode}: {len(times)} frames"
               f"  ({time.perf_counter() - started:.0f} s)")
