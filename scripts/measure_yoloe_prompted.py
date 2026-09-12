# -*- coding: utf-8 -*-
"""Peak GPU memory and per-frame time of the PROMPTABLE YOLOE, as E4-D runs it.

    python scripts/measure_yoloe_prompted.py
    python scripts/measure_yoloe_prompted.py --dataset tbbt --frames 128

The cost table has a row for the query-time detection stage whose memory cell is
the only one still empty. The extraction pass measures the PROMPT-FREE YOLOE
(one closed class list, no text side); the stage E4-D runs is a different object:
it holds the detector AND the MobileCLIP-BLT text encoder that turns the query
phrases into class embeddings, and the encoder is loaded lazily on the first
prompt. Measuring the prompt-free variant and writing its number into the E4-D
row would understate the stage by whatever the encoder costs.

Method is the one used for every other row of that table (src/measurement/cost):
the device-wide sampler starts BEFORE the model is built, so the weights count,
and the peak covers the CUDA context as well.

Writes results/measurements/yoloe_prompted_<date>_<time>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.queries import load_queries                      # noqa: E402
from src.measurement import cost                               # noqa: E402
from src.retrieval.query_detection import (PromptedDetector,   # noqa: E402
                                           prompts_of, texts_of)
from src.utils.notebook import save_measurement                # noqa: E402

#: how many distinct query phrases the detector is asked for at once. E4-D asks
#: per query, and a query carries a handful of object phrases; the text encoder
#: runs once per distinct tuple, so this is what decides its share of the peak.
PROMPTS = 8


def query_texts(dataset: str, split: str, wanted: int) -> list[str]:
    """Distinct object phrases taken from real queries, not invented ones.

    A made-up prompt list would measure the same weights but a different text
    encoder input, and the encoder is half of what this script is here to see.
    """
    texts: list[str] = []
    for query in load_queries(dataset, split):
        found = prompts_of(query)
        if not found:
            continue
        for text in texts_of(found)[0]:
            if text not in texts:
                texts.append(text)
        if len(texts) >= wanted:
            break
    return texts[:wanted]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="office", choices=["tbbt", "office"])
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--frames", type=int, default=128,
                        help="grid frames the detector is run on per repeat")
    parser.add_argument("--prompts", type=int, default=PROMPTS,
                        help="distinct query phrases asked for at once")
    args = parser.parse_args()

    started = time.perf_counter()
    frames = cost.sample_frames(args.dataset, args.split, args.frames)
    if not frames:
        raise SystemExit(f"no frames for {args.dataset}/{args.split}")
    texts = query_texts(args.dataset, args.split, args.prompts)
    if not texts:
        raise SystemExit("no object phrases in the queries of this split")
    print(f"{args.dataset} / {args.split}: {len(frames)} frames, "
          f"{len(texts)} prompts")
    print(f"prompts: {', '.join(texts)}\n")

    # the baseline has to be taken before the model reaches the card, otherwise
    # the weights are already there when the sampler starts and only the
    # activations get counted (see cost.device_memory_sampler)
    sampler = cost.device_memory_sampler()
    detector = PromptedDetector()
    result = cost.timed(lambda: detector.detect(frames, texts), sampler=sampler)

    per_frame_ms = 1000 * result["median_s"] / len(frames)
    data = {
        "dataset": args.dataset,
        "split": args.split,
        "weights": detector.weights,
        "frames": len(frames),
        "prompts": texts,
        "batch": detector.batch,
        "confidence": detector.confidence,
        "seconds_per_frame": round(result["median_s"] / len(frames), 5),
        "ms_per_frame": round(per_frame_ms, 2),
        "elapsed_min": round((time.perf_counter() - started) / 60, 1),
        **{key: result[key] for key in ("median_s", "p95_s", "min_s", "spread",
                                        "repeats", "peak_vram_gb",
                                        "torch_vram_gb", "vram_noise_gb")},
    }

    print(f"peak GPU memory: {data['peak_vram_gb']} GB "
          f"(PyTorch allocator {data['torch_vram_gb']} GB, "
          f"noise {data['vram_noise_gb']} GB)")
    print(f"time: {per_frame_ms:.2f} ms/frame, spread {data['spread']:.1%}, "
          f"{data['repeats']} repeats")
    save_measurement("yoloe_prompted", data)


if __name__ == "__main__":
    main()
