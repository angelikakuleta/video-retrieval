"""Computational cost of the pipeline components (chapter 4).

Four quantities, normalized per HOUR OF MATERIAL rather than per fragment: the
segmentation strategies produce different fragment counts from the same
recording, so anything per-fragment would not be comparable.

    extraction        minutes per hour of material
    peak GPU memory   GB, device-wide, so the ONNX models are covered too
    index             MB per hour, from what actually lies on disk
    query handling    milliseconds per query, median and 95th percentile

Three things shape the numbers. The sample comes from the corpus, not from
noise, because a detector's cost depends on what it finds. Models are loaded and
released one at a time, since LLaVA alone wants 14 of the card's 16 GB and the
pipeline works that way too, so the peak reported is the peak reached in
practice. And the warm-up passes are discarded: kernel selection is paid once
per session, not once per hour of material.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data import datasets
from src.data.ranges import load_ranges, load_timelines, split_episodes
from src.segmentation.timeline import total as span_total
from src.utils import gpu

#: passes whose time is recorded
REPEATS = 15
#: passes thrown away first, so kernel selection is not counted as extraction
WARMUP = 3

#: source frames per window of the sequence models (X-CLIP and SlowFast share it)
WINDOW_FRAMES = 64
#: frame rate of the normalized recordings, for turning windows into seconds
FPS = 23.976


# -------------------------------------------------------------------- timing
def free_gpu() -> None:
    """Releases what the previous component left behind."""
    gpu.free()


#: how often the device-wide memory sampler looks at the card, in seconds
VRAM_POLL_S = 0.02
#: how many baseline samples are taken before the timed repeats
VRAM_BASELINE_SAMPLES = 12


class _DeviceMemorySampler:
    """Smallest free memory seen on the whole card while the component runs.

    ``torch.cuda.max_memory_allocated`` only sees PyTorch's own allocator, so it
    reports near zero for the ONNX Runtime models (RetinaFace, ArcFace,
    HSEmotion) -- their weights and workspace are invisible to it. What has to
    fit on the card is the DEVICE-wide figure, so that is what is measured here:
    free memory before the run minus the least free memory during it.

    The baseline is the LARGEST of several samples taken while idle, not the
    last one: a transient dip caused by something else on the card would
    otherwise be charged to the component. The scatter of those samples is
    reported alongside, so a noisy card is visible in the result instead of
    quietly inflating it. The measurement assumes exclusive use of the GPU.
    """

    def __init__(self) -> None:
        import torch
        self._torch = torch
        self.baseline = 0
        self.noise = 0
        self._lowest = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _free(self) -> int:
        return self._torch.cuda.mem_get_info()[0]

    def start(self) -> None:
        samples = []
        for _ in range(VRAM_BASELINE_SAMPLES):
            samples.append(self._free())
            time.sleep(VRAM_POLL_S)
        self.baseline = max(samples)
        self.noise = max(samples) - min(samples)
        self._lowest = self.baseline
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop.wait(VRAM_POLL_S):
            free = self._free()
            if free < self._lowest:
                self._lowest = free

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # the sampler can miss a peak that lives shorter than one poll, so take
        # the deeper of "seen while polling" and "still held now"
        self._lowest = min(self._lowest, self._free())
        return {"peak_vram_gb": round(max(0, self.baseline - self._lowest) / 1e9, 2),
                "vram_noise_gb": round(self.noise / 1e9, 2)}


def device_memory_sampler():
    """Sampler with its baseline taken NOW; ``None`` when there is no card.

    Start it while the card is still clean -- before the model is built, not
    after. PyTorch's caching allocator takes its pool from the driver on first
    use and then reuses it, so a baseline taken once the model is loaded sees
    free memory that no longer moves, and the device-wide figure comes out zero
    for exactly the components the allocator already accounts for.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    sampler = _DeviceMemorySampler()
    sampler.start()
    return sampler


def timed(run: Callable[[], Any], repeats: int = REPEATS,
          warmup: int = WARMUP, sampler=None) -> dict:
    """Runs ``run``; reports how long it took and how much of the card it needed.

    ``peak_vram_gb`` is measured device-wide, so it covers the ONNX Runtime
    models too and includes the CUDA context. ``torch_vram_gb`` is the PyTorch
    allocator's own peak, kept for comparison: where the two diverge sharply the
    component runs outside PyTorch.

    Pass a ``sampler`` from :func:`device_memory_sampler` started before the
    model was built; without one the weights are already on the card when the
    baseline is taken and only the activations get counted.
    """
    import torch

    cuda = torch.cuda.is_available()
    if sampler is None and cuda:
        sampler = _DeviceMemorySampler()
        sampler.start()

    for _ in range(warmup):
        run()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        run()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)

    memory = sampler.stop() if sampler is not None else {"peak_vram_gb": None,
                                                         "vram_noise_gb": None}
    torch_peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else None
    median = float(np.median(times))
    # how much the repeats disagree with each other, relative to the median: the
    # honest answer to "would more repeats help". Wide spread means the machine
    # is noisy; narrow spread means the number is as good as this method gets.
    spread = float(max(times) - min(times)) / median if median else 0.0
    return {"median_s": median,
            "p95_s": float(np.percentile(times, 95)),
            "min_s": float(min(times)),
            "spread": round(spread, 3),
            "repeats": repeats,
            "torch_vram_gb": round(torch_peak, 2) if torch_peak is not None else None,
            **memory}


# ---------------------- how much material the numbers are normalized against
def corpus_hours(dataset: str, split: str) -> float:
    """Hours of material in the corpus of one split, on the content axis."""
    ranges = load_ranges(dataset)
    episodes = split_episodes(ranges, split)
    timelines = load_timelines(dataset, ranges)
    seconds = 0.0
    for episode in episodes:
        timeline = timelines.get(episode)
        if timeline is None:
            seconds += sum(b - a for a, b in ranges[episode]["spans"])
            continue
        # the content axis: ranges minus the transition masks, which is the
        # material a fragment length is measured in
        seconds += sum(span_total(piece) for piece in timeline.pieces)
    return seconds / 3600.0


def units_per_hour(step_s: float = 1.25) -> dict[str, float]:
    """How many units of each kind one hour of material contains.

    A frame comes from the sampling grid, a window from the 64 source frames the
    sequence models read. Both follow from the material, not from the segmentation.
    """
    return {"frame": 3600.0 / step_s,
            "window": 3600.0 / (WINDOW_FRAMES / FPS)}


def faces_per_hour(dataset: str, split: str) -> float:
    """Detected faces per hour, the unit the crop-level components are paid in."""
    from src.features import faces

    ranges = load_ranges(dataset)
    episodes = split_episodes(ranges, split)
    found = 0
    for episode in episodes:
        path = faces.episode_npz(dataset, episode)
        if path.exists():
            found += len(faces.load_episode(dataset, episode)["crop"])
    hours = corpus_hours(dataset, split)
    return found / hours if hours else 0.0


def per_hour(seconds_per_unit: float, units: float) -> float:
    """Seconds per unit -> minutes per hour of material."""
    return seconds_per_unit * units / 60.0


# ---------------------------------------- the sample the measurement runs on
def sample_frames(dataset: str, split: str, count: int,
                  step_s: float = 1.25) -> list:
    """``count`` grid frames of the split, spread over its episodes."""
    from src.segmentation.frames import iter_frames

    ranges = load_ranges(dataset)
    episodes = sorted(split_episodes(ranges, split))
    if not episodes:
        return []
    per_episode = max(1, -(-count // len(episodes)))
    frames = []
    for episode in episodes:
        video = datasets.video_path(dataset, ranges[episode]["video_file"])
        if not video.exists():
            continue
        # a wide step spreads the sample over the whole episode instead of
        # taking a thousand frames of one scene
        wide = max(step_s, 20.0)
        taken = 0
        for _, frame in iter_frames(video, wide):
            frames.append(frame)
            taken += 1
            if taken >= per_episode:
                break
        if len(frames) >= count:
            break
    return frames[:count]


def sample_crops(dataset: str, split: str, count: int) -> np.ndarray:
    """``count`` aligned face crops from the buffer, or an empty array."""
    from src.features import faces

    episodes = sorted(split_episodes(load_ranges(dataset), split))
    available = [e for e in episodes if faces.episode_npz(dataset, e).exists()]
    if not available:
        return np.zeros((0, faces.CROP_SIZE, faces.CROP_SIZE, 3), np.uint8)
    # spread over the episodes, like the frame sample: one episode's faces are
    # the same few actors in the same light, and that is not the whole split
    per_episode = max(1, -(-count // len(available)))
    collected = []
    for episode in available:
        crops = faces.load_episode(dataset, episode)["crop"]
        collected.append(crops[:per_episode])
        if sum(len(c) for c in collected) >= count:
            break
    if not collected:
        return np.zeros((0, faces.CROP_SIZE, faces.CROP_SIZE, 3), np.uint8)
    return np.concatenate(collected)[:count]


def sample_windows(frames: list, count: int, per_window: int = 8) -> list[list]:
    """Windows of ``per_window`` frames, built from the sampled frames."""
    arrays = [np.asarray(f) for f in frames]
    if not arrays:
        return []
    return [[arrays[(i * per_window + k) % len(arrays)] for k in range(per_window)]
            for i in range(count)]


# ---------------------------------------------------------------- index size
def directory_mb(path: Path) -> float:
    """Megabytes a cache directory occupies; a missing one is zero."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def index_sizes(dataset: str, split: str, strategy: str,
                encoder: str = "openclip_vit_h14") -> list[dict]:
    """What each component leaves on disk, per hour of material.

    Everything the component writes counts, not only the vector index: several keep
    their representation in a catalog instead.
    """
    from src.features import (
        captions,
        expressions,
        faces,
        identity,
        motion,
        objects,
        regions,
    )

    root = datasets.ROOT / "data" / "cache"
    name = datasets.dataset_dir(dataset)
    hours = corpus_hours(dataset, split)
    entries = [
        ("OpenCLIP ViT-H/14", [root / "embeddings" / "openclip_vit_h14" / name,
                               datasets.index_dir(dataset, strategy, "openclip_vit_h14", split)]),
        ("OpenCLIP ViT-B/32", [root / "embeddings" / "openclip_vit_b32" / name,
                               datasets.index_dir(dataset, strategy, "openclip_vit_b32", split)]),
        ("X-CLIP", [datasets.index_dir(dataset, strategy, "xclip_b32", split)]),
        ("SlowFast", [motion.cache_dir(strategy, dataset)]),
        ("BLIP", [captions.cache_dir("blip", dataset),
                  root / "captions" / "blip" / encoder / name]),
        ("LLaVA-1.5", [captions.cache_dir("llava_1_5_7b", dataset),
                       root / "captions" / "llava_1_5_7b" / encoder / name]),
        ("YOLO11", [objects.cache_dir("yolo11", dataset)]),
        ("YOLOE-11", [objects.cache_dir("yoloe_promptfree", dataset)]),
        ("RetinaFace (crop buffer)", [faces.cache_dir(dataset)]),
        ("ArcFace", [identity.cache_dir(dataset)]),
        ("HSEmotion", [expressions.cache_dir(dataset)]),
        ("face regions", [regions.cache_dir(encoder, dataset)]),
    ]
    out = []
    for label, paths in entries:
        megabytes = sum(directory_mb(p) for p in paths)
        out.append({"component": label,
                    # three decimals: the smallest components sit around 0.1 MB/h
                    # and a single decimal would flatten them all to the same value
                    "mb": round(megabytes, 3),
                    "mb_per_hour": round(megabytes / hours, 3) if hours else None})
    return out


# --------------------------------------------------------------- query phase
def query_times(config_file: Path | str, split: str,
                repeats: int = REPEATS, log: Callable[[str], None] = print) -> dict:
    """Per-signal and total query handling time on the real collection.

    Every signal is timed on the split's queries one at a time, after a warm-up. The
    fusion entry covers what happens after the signals: standardization, the
    weighted sum and the ranking.
    """
    from src.data.queries import load_queries
    from src.features import encoders
    from src.retrieval.fusion import combine, standardize
    from src.retrieval.pipeline import Pipeline
    from src.runners import components as comp
    from src.runners import stages
    from src.utils.config import load_experiment

    config = load_experiment(config_file, split)
    factory = encoders.lazy(config.components.scene_embedding.model)
    collection, _, context = stages.ensure_collection(config, factory, log=log)
    signals = comp.build_signals(config, collection, context["rows"],
                                 context["episodes"], context["timelines"],
                                 factory, log=log)
    queries = load_queries(config.dataset, split)
    pipeline = Pipeline(collection, signals, weights=config.scoring.weights.manual)

    # the enabled components travel WITH the measurement: the notebook labels the
    # rows from this record, so a run stays readable after its configuration file
    # is renamed, reworked or removed
    enabled = {name: {key: value for key, value in fields.items() if key != "enabled"}
               for name, fields in config.components.model_dump().items()
               if fields.get("enabled")}
    out = {"config": str(config_file), "collection": collection.size,
           "queries": len(queries), "components": enabled, "signals": []}

    # The phrase layer is part of handling a query, not of building the index:
    # spaCy parses the query, WordNet decides what each phrase is and the text
    # encoder embeds every form, once per query, before any closed-vocabulary
    # signal can start. Pipeline.signal_matrices counts it inside its per-query
    # timing for exactly that reason; here it gets its own row so the table can
    # say what it costs, and the signals below are then timed on phrases already
    # extracted rather than on the empty path phrases=None takes them down.
    #
    # The extractor comes from the pipeline rather than being rebuilt here: the
    # rule for when phrases are needed at all belongs in one place.
    extract = pipeline._phrase_extractor()
    if extract is None:
        out["phrases"] = None
        log(f"  {'frazy zapytania':<18}{'-':>8}  (zaden sygnal ich nie czyta)")
        prepared: list = [None] * len(queries)
    else:
        for query in queries[:WARMUP]:
            extract(query)
        times = []
        prepared = []
        for query in queries:
            started = time.perf_counter()
            found = extract(query)
            times.append((time.perf_counter() - started) * 1000.0)
            prepared.append([found])
        out["phrases"] = {"median_ms": round(float(np.median(times)), 3),
                          "p95_ms": round(float(np.percentile(times, 95)), 3)}
        log(f"  {'frazy zapytania':<18}{out['phrases']['median_ms']:>8.2f} ms"
            f"  p95 {out['phrases']['p95_ms']:>8.2f} ms")

    for signal in signals:
        for query, one in zip(queries[:WARMUP], prepared[:WARMUP]):
            signal.raw_values([query], one)
        times = []
        for query, one in zip(queries, prepared):
            started = time.perf_counter()
            signal.raw_values([query], one)
            times.append((time.perf_counter() - started) * 1000.0)
        out["signals"].append({
            "signal": signal.name,
            "median_ms": round(float(np.median(times)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3)})
        log(f"  {signal.name:<18}{out['signals'][-1]['median_ms']:>8.2f} ms"
            f"  p95 {out['signals'][-1]['p95_ms']:>8.2f} ms")

    # the fusion layer on its own: standardization, weighting, ranking
    raw = [signal.raw_values(queries[:1], prepared[0]) for signal in signals]
    active = np.ones((len(signals), 1), dtype=bool)
    times = []
    for _ in range(WARMUP):
        combine([standardize(r) for r in raw], active, pipeline.base)
    for _ in range(max(repeats, 20)):
        started = time.perf_counter()
        scores = combine([standardize(r) for r in raw], active, pipeline.base)
        pipeline.top(scores[0], 10)
        times.append((time.perf_counter() - started) * 1000.0)
    out["fusion"] = {"median_ms": round(float(np.median(times)), 3),
                     "p95_ms": round(float(np.percentile(times, 95)), 3)}
    log(f"  {'laczenie+ranking':<18}{out['fusion']['median_ms']:>8.2f} ms")
    return out


# ---------------------------------------------------------------- extraction
#: Every component, the unit it is paid in, and how to build its measurement.
#: Each ``run`` walks its sample in the SAME batch the pipeline uses, so the peak
#: memory is the one production reaches; accuracy comes from covering more distinct
#: units and from REPEATS, never from an inflated batch.
#: The order is the one chapter 6 lists them in, cheapest models last so that an
#: interrupted session still leaves the expensive ones measured.
COMPONENTS = ("openclip_vit_h14", "openclip_vit_b32", "xclip", "slowfast",
              "blip", "llava", "yolo11", "yoloe", "retinaface", "arcface",
              "regions", "hsemotion")

#: paid per face CROP rather than per frame or window, so they are skipped
#: whenever the crop buffer of the dataset is empty: there is nothing for them
#: to walk and a zero would not be the answer.
CROP_COMPONENTS = ("arcface", "regions", "hsemotion")


def base_representation() -> str | None:
    """The base representation E2 settled on, from ``configs/frozen.yaml``.

    The region embeddings are computed with the visual encoder of the base
    representation (:mod:`src.features.regions`), and which encoder that is, is a
    VERDICT. Writing the name here would make this module a second place where
    the base is decided, and the two would drift the first time E2 was revisited.

    The ``openclip_*`` rows are not this: they measure the two E2 CANDIDATES on
    frames, which is the question E2 itself asks and which has an answer before
    any verdict exists.
    """
    from src.utils.frozen import load_frozen

    return load_frozen().scene_embedding


def _openclip(model: str, frames: list, count: int = 256):
    from src.features import encoders
    from src.features.frame_cache import BATCH

    encoder = encoders.build(model)
    chunk = frames[:count]

    def run():
        for start in range(0, len(chunk), BATCH):
            encoder.encode_images(chunk[start:start + BATCH])

    return encoder, run, len(chunk), "frame"


def _xclip(frames: list, windows: int = 64):
    from src.features.xclip import XClipEncoder

    encoder = XClipEncoder()
    batch = sample_windows(frames, windows)
    return encoder, (lambda: encoder.encode_windows_with_prompts(batch)), \
        len(batch), "window"


def _slowfast(frames: list, windows: int = 32):
    from src.features.motion import FAST_FRAMES, SlowFast

    model = SlowFast()
    batch = sample_windows(frames, windows, per_window=FAST_FRAMES)
    return model, (lambda: model.predict(batch)), len(batch), "window"


def _caption(generator: str, frames: list, count: int):
    from src.features.captions import BATCH, Captioner

    model = Captioner(generator)
    chunk = frames[:count]
    batch = BATCH[generator]

    def run():
        for start in range(0, len(chunk), batch):
            model.describe(chunk[start:start + batch])

    return model, run, len(chunk), "frame"


def _objects(detector: str, frames: list, count: int = 128):
    from src.features import objects

    model = objects._model(detector)
    chunk = [np.asarray(f) for f in frames[:count]]

    def run():
        for start in range(0, len(chunk), objects.BATCH):
            model.predict(chunk[start:start + objects.BATCH], verbose=False,
                          conf=objects.CONFIDENCE)

    return model, run, len(chunk), "frame"


def _retinaface(frames: list, count: int = 128):
    from src.features.faces import FaceDetector

    detector = FaceDetector()
    chunk = [np.asarray(f) for f in frames[:count]]

    def run():
        for image in chunk:
            detector.detect(image)

    return detector, run, len(chunk), "frame"


def _arcface(crops: np.ndarray, count: int = 512):
    from src.features.identity import BATCH, ArcFace

    model = ArcFace()
    chunk = crops[:count]

    def run():
        for start in range(0, len(chunk), BATCH):
            model.encode(chunk[start:start + BATCH])

    return model, run, len(chunk), "crop"


def _regions(crops: np.ndarray, count: int = 512):
    """The base encoder over the face crops: the open-vocabulary half of E5.

    This component has no model of its own, and that is exactly why it needs its
    own row. The encoder is the one measured two rows up, but the INPUT is not:
    there it walks the grid frames of the recording, here every accepted face
    crop of the corpus, and the two counts per hour are different numbers. The
    cost of the region signal is therefore not contained in the encoder's row and
    never was.

    The batch is ``regions.BATCH``, the call is ``encode_images`` on a list of
    crops -- what ``ensure_regions`` does, so the peak memory is production's.
    """
    from src.features import encoders
    from src.features.regions import BATCH

    encoder = encoders.build(base_representation())
    chunk = list(crops[:count])

    def run():
        for start in range(0, len(chunk), BATCH):
            encoder.encode_images(chunk[start:start + BATCH])

    return encoder, run, len(chunk), "crop"


def _hsemotion(crops: np.ndarray, count: int = 512):
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    from src.features.expressions import BATCH, MODEL

    model = HSEmotionRecognizer(model_name=MODEL)
    chunk = list(crops[:count])

    def run():
        for start in range(0, len(chunk), BATCH):
            model.predict_multi_emotions(chunk[start:start + BATCH], logits=False)

    return model, run, len(chunk), "crop"


def extraction_costs(dataset: str, split: str, frames: list, crops: np.ndarray,
                     only: Sequence[str] | None = None,
                     caption_frames: int = 8,
                     log: Callable[[str], None] = print) -> list[dict]:
    """Extraction cost of every component, in minutes per hour of material.

    Each model is built, measured and dropped before the next one is created.
    """
    units = units_per_hour()
    units["crop"] = faces_per_hour(dataset, split)
    wanted = set(only) if only else set(COMPONENTS)

    builders = {
        "openclip_vit_h14": ("OpenCLIP ViT-H/14", lambda: _openclip("openclip_vit_h14", frames)),
        "openclip_vit_b32": ("OpenCLIP ViT-B/32", lambda: _openclip("openclip_vit_b32", frames)),
        "xclip": ("X-CLIP base/32", lambda: _xclip(frames)),
        "slowfast": ("SlowFast R50", lambda: _slowfast(frames)),
        "blip": ("BLIP large", lambda: _caption("blip", frames, caption_frames)),
        "llava": ("LLaVA-1.5-7B", lambda: _caption("llava_1_5_7b", frames, caption_frames)),
        "yolo11": ("YOLO11 large", lambda: _objects("yolo11", frames)),
        "yoloe": ("YOLOE-11 large", lambda: _objects("yoloe_promptfree", frames)),
        "retinaface": ("RetinaFace R50", lambda: _retinaface(frames)),
        "arcface": ("ArcFace R100", lambda: _arcface(crops)),
        # the same name the index section of this payload gives it: one
        # component, one name. The Polish row label of the thesis table lives in
        # the notebook, which looks this entry up by `key` and never by name.
        "regions": ("face regions", lambda: _regions(crops)),
        "hsemotion": ("HSEmotion B0", lambda: _hsemotion(crops)),
    }

    out = []
    for key in COMPONENTS:
        if key not in wanted:
            continue
        label, build = builders[key]
        if key in CROP_COMPONENTS and not len(crops):
            log(f"{label}: no face crops cached - skipped")
            continue
        if key == "regions" and base_representation() is None:
            # a skipped row, not a crash: this module reports what it could
            # measure, and the verdict is not its to make up
            log(f"{label}: configs/frozen.yaml carries no scene_embedding verdict "
                "yet - skipped")
            continue
        free_gpu()
        log(f"{label}: loading")
        # baseline while the card is still clean: the weights are part of what
        # the component needs, so they have to fall inside the measured window
        sampler = device_memory_sampler()
        started = time.perf_counter()
        model, run, batch, unit = build()
        load_s = time.perf_counter() - started

        result = timed(run, sampler=sampler)
        seconds_per_unit = result["median_s"] / batch
        out.append({
            "component": label, "key": key, "unit": unit, "batch": batch,
            "load_s": round(load_s, 2),
            "seconds_per_unit": round(seconds_per_unit, 5),
            "min_per_hour": round(per_hour(seconds_per_unit, units[unit]), 2),
            "peak_vram_gb": result["peak_vram_gb"],
            "torch_vram_gb": result["torch_vram_gb"],
            "vram_noise_gb": result["vram_noise_gb"],
            "repeats": result["repeats"],
            "spread": result["spread"],
        })
        log(f"  {seconds_per_unit * 1000:.1f} ms/{unit}"
            f"   {out[-1]['min_per_hour']:.1f} min/h"
            f"   peak {out[-1]['peak_vram_gb']} GB"
            f"   rozrzut {100 * result['spread']:.0f}%")
        del model, run
        free_gpu()
    return out
