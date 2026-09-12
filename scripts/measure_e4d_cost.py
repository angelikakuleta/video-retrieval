"""Cost of the query-time detection stage, E4-D (section 08, chapter 5).

    python scripts/measure_e4d_cost.py --split dev --dataset office

Measured on every query of the split, not on a sample: the table of chapter 5
reports a p95, and the p95 of a sample is not the p95 of the query set.

The loop here is the NATURAL one -- query by query, each opening the recordings
its own candidates lie in. The stage itself runs the inverted loop, episode by
episode, because decoding is the bottleneck and the inverted order pays it once;
but an inverted loop has no per-query time to take a median of, and the row of
the cost table is a per-query number. So the two differ on purpose, and this
script measures the SLOWER of the two. That is the honest direction: it is the
cost a single query would carry if asked on its own.

Four components, timed apart: extracting the phrases, decoding the frames,
detecting, and fusing. Warm-up as in measure_cost.py -- the first query pays for
the weights, the CUDA context and the text encoder, and none of that belongs in
the median.

Writes through save_measurement("cost", ...) under the key ``e4d``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

WARMUP = 2


def summarize(values: list[float]) -> dict:
    """Median, mean, p95 and the observed range, in milliseconds.

    The range is here because of what the probe showed: decoding runs from 161 to
    682 seconds a query while detection stays inside 6,2 to 8,2. One summary
    number would hide that the two components of this stage behave nothing alike,
    and the table reports them apart for exactly that reason. Min and max cost
    nothing to carry and are the only honest way to state a spread the p95 cannot
    describe on a sample.
    """
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_ms": round(statistics.median(ordered) * 1000, 2),
        "mean_ms": round(statistics.fmean(ordered) * 1000, 2),
        "p95_ms": round(float(np.percentile(ordered, 95)) * 1000, 2),
        "min_ms": round(ordered[0] * 1000, 2),
        "max_ms": round(ordered[-1] * 1000, 2),
        "total_s": round(sum(ordered), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--dataset", default="office")
    parser.add_argument("--config", default="configs/e2a_office.yaml",
                        help="the BASE configuration whose ranking is re-ranked")
    parser.add_argument("--limit", type=int, default=None,
                        help="measure this many queries drawn at random without "
                             "replacement, seeded from the configuration - a timing "
                             "probe, NOT the measurement: a p95 over a sample is "
                             "not the p95")
    args = parser.parse_args()

    # Gate 1 refused --split test while the configuration was still open: the text
    # path of YOLOE had no business touching test queries before the verdicts were
    # frozen. The verdicts are in configs/frozen.yaml, so the gate has served its
    # purpose and both splits are allowed. Nothing here decides anything -- it is a
    # clock measurement, and it reads the same queries the test runs already read.

    from PIL import Image

    from src.data import datasets
    from src.data.queries import load_queries
    from src.data.ranges import load_ranges
    from src.evaluation.relevance import relevance_sets
    from src.features import encoders
    from src.features.xclip import collect_frames
    from src.retrieval import query_detection as qd
    from src.retrieval.fusion import combine
    from src.retrieval.pipeline import Pipeline
    from src.runners import components, stages
    from src.segmentation import segments as seg
    from src.utils import gpu
    from src.utils.config import load_experiment
    from src.utils.notebook import save_measurement

    config = load_experiment(args.config, args.split)
    factory = encoders.lazy(config.components.scene_embedding.model)
    collection, _, context = stages.ensure_collection(config, factory)
    signals = components.build_signals(config, collection, context["rows"],
                                       context["episodes"], context["timelines"],
                                       factory)
    queries = load_queries(config.dataset, args.split)
    rows = [r for r in seg.load(datasets.segments_csv(config.collection_strategy,
                                                      config.dataset))
            if r["split"] == args.split]
    relevance = relevance_sets(queries, rows)
    in_collection = set(collection.vids)
    evaluated = [q for q in queries if relevance[q.desc_id] & in_collection]
    population = len(evaluated)
    sampled = False
    if args.limit and args.limit < len(evaluated):
        # Drawn at random without replacement, not off the head of the list: the
        # list is grouped by episode, so the first n queries would all point at
        # one or two recordings, and episodes differ in length and in how far
        # apart their candidates lie. Seeded from the configuration, so the same
        # --limit twice measures the same queries.
        #
        # The picks keep their ORIGINAL relative order. The full measurement walks
        # the queries grouped by episode and the file cache is warm for the second
        # query of a recording; preserving the order keeps that structure in the
        # sample instead of replacing it with a locality the real loop never has.
        # Thinning still costs some of it, so a sample runs slightly slower per
        # query than the whole set -- the same direction as everything else here.
        import random

        picked = sorted(random.Random(config.seed).sample(
            range(len(evaluated)), args.limit))
        evaluated = [evaluated[i] for i in picked]
        sampled = True
    print(f"{config.dataset}/{args.split}: {len(evaluated)} queries, "
          f"{collection.size} fragments"
          + (f" (random sample of {args.limit}, seed {config.seed})" if sampled else ""))

    pipeline = Pipeline(collection, signals)
    matrices, active = pipeline.signal_matrices(evaluated)
    scores = combine(matrices, active, pipeline.base)

    # one big model at a time: the base encoder has to leave the card before the
    # detector loads, exactly as in a run
    del pipeline, signals, matrices
    gpu.free()

    ranges = load_ranges(config.dataset)
    row_of = {vid: i for i, vid in enumerate(collection.vids)}
    episode_of_column = {}
    for row in rows:
        from src.evaluation.relevance import fragment_id
        column = row_of.get(fragment_id(row["episode"], row["segment_id"]))
        if column is not None:
            episode_of_column[int(column)] = row["episode"]

    detector = qd.PromptedDetector()
    frames_cache: dict[str, tuple[float, int, dict[int, list[int]]]] = {}

    def geometry(episode: str):
        """fps, frame count and the usable frame numbers per fragment."""
        if episode not in frames_cache:
            path = datasets.video_path(config.dataset, ranges[episode]["video_file"])
            fps, count = qd.probe(path)
            frames_cache[episode] = (fps, count, qd.episode_frames(
                rows, episode, row_of, context["timelines"].get(episode), count, fps))
        return frames_cache[episode]

    parts = {"phrases": [], "decoding": [], "detection": [], "fusion": [], "total": []}
    for position, query in enumerate(evaluated):
        started = time.perf_counter()

        mark = time.perf_counter()
        prompts = qd.prompts_of(query)
        phrases_s = time.perf_counter() - mark

        candidates = qd.candidates_of(scores[position])
        values = np.full(len(candidates), np.nan)
        decoding_s = detection_s = 0.0

        if prompts:
            texts, owners = qd.texts_of(prompts)
            wanted: dict[str, list[int]] = {}
            for column in candidates:
                episode = episode_of_column.get(int(column))
                if episode is not None:
                    wanted.setdefault(episode, []).append(int(column))

            for episode, columns in wanted.items():
                fps, count, per_column = geometry(episode)
                path = datasets.video_path(config.dataset,
                                           ranges[episode]["video_file"])
                needed = sorted({n for column in columns
                                 for n in per_column.get(column, [])})

                mark = time.perf_counter()
                decoded = collect_frames(path, needed, height=None)
                decoding_s += time.perf_counter() - mark

                for column in columns:
                    numbers = [n for n in per_column.get(column, []) if n in decoded]
                    at = int(np.flatnonzero(candidates == column)[0])
                    if not numbers:
                        continue
                    mark = time.perf_counter()
                    found = detector.detect(
                        [Image.fromarray(decoded[n]) for n in numbers], texts)
                    detection_s += time.perf_counter() - mark
                    values[at] = qd.value_of(
                        qd.per_phrase_maxima(found, owners, len(prompts)))

            mark = time.perf_counter()
            fused = qd.fuse(values, scores[position][candidates])
            qd.reorder(np.argsort(-scores[position], kind="stable"), candidates, fused)
            fusion_s = time.perf_counter() - mark
        else:
            fusion_s = 0.0

        total_s = time.perf_counter() - started
        if position >= WARMUP:          # the first queries pay for the weights
            parts["phrases"].append(phrases_s)
            parts["decoding"].append(decoding_s)
            parts["detection"].append(detection_s)
            parts["fusion"].append(fusion_s)
            parts["total"].append(total_s)
        print(f"[{position + 1}/{len(evaluated)}] {total_s:6.1f} s  "
              f"(decode {decoding_s:5.1f}, detect {detection_s:5.1f}, "
              f"{len(prompts)} prompt(s))", flush=True)

    payload = {name: summarize(values) for name, values in parts.items()}
    payload["queries_total"] = len(evaluated)
    payload["warmup_skipped"] = WARMUP
    payload["dataset"] = config.dataset
    payload["split"] = args.split
    payload["loop"] = ("natural: one query at a time, each opening the recordings its "
                       "own candidates lie in. The stage itself runs the inverted "
                       "loop and is faster; this is the cost of a query asked alone.")
    payload["partial"] = sampled

    # A sample is saved, but never as if it were the census. Precision of a
    # quantile follows the NUMBER of observations, not the fraction of the
    # population they cover, so a sample carries the same median as a census of
    # the same size -- and the same weak p95. Roughly 5% of the observations lie
    # above the p95, so the estimate rests on that many; the count travels with
    # the payload and the table has to be read with it.
    if sampled:
        above = sum(1 for value in parts["total"]
                    if value * 1000.0 >= payload["total"]["p95_ms"])
        payload["sample"] = {
            "measured": len(evaluated),
            "population": population,
            "seed": config.seed,
            "draw": "random without replacement, original order kept",
            "p95_rests_on": above,
        }

    print("\ncomponent        n   median    mean     p95    total")
    for name in ("phrases", "decoding", "detection", "fusion", "total"):
        entry = payload[name]
        if entry.get("n"):
            print(f"{name:12} {entry['n']:5} {entry['median_ms']:8.1f} "
                  f"{entry['mean_ms']:8.1f} {entry['p95_ms']:8.1f} "
                  f"{entry['total_s']:8.1f} s")

    if sampled:
        print(f"\nSAMPLE: {len(evaluated)} of {population} queries, seed "
              f"{config.seed}. The median holds at this size. The p95 rests on "
              f"{payload['sample']['p95_rests_on']} observation(s) and is NOT the "
              f"p95 of the query set - about 200 queries are needed before the "
              f"tail carries ten.")
    save_measurement("cost", {"e4d": payload})


if __name__ == "__main__":
    main()
