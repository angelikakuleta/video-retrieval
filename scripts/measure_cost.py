"""Measures the computational cost of the pipeline components.

    python scripts/measure_cost.py --dataset office --split dev
    python scripts/measure_cost.py --only llava --skip-query

The query phase runs on the FULL pipeline unless --config says otherwise, so the
first run also computes whatever that pipeline still misses -- the SlowFast motion
cache, most likely. That is extraction, not measurement, and it happens once.

Extraction in minutes per hour of material, peak GPU memory, index size in MB per
hour, query handling as median and 95th percentile. Warmed-up models, the same
number of repetitions everywhere.

Extraction is measured on a SAMPLE spread over the episodes of the split and the
per-unit median is scaled to an hour of material, so the run takes about half an
hour rather than the days a full pass would need. LLaVA decides that budget on its
own -- lowering --caption-frames is what makes the measurement quick and rough.

**Close everything else that uses the GPU first** - LLaVA alone wants 14 GB and a
browser holding half a gigabyte changes the reported peak.

Writes results/measurements/cost_<date>_<time>.json, read by
notebooks/experiments/cost.ipynb. Cheapest model last, so an interrupted session
still leaves the expensive ones done.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.measurement import cost
from src.utils.config import load_experiment
from src.utils.notebook import save_measurement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="office",
                        choices=["tbbt", "office"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--strategy", default=None,
                        help="segmentation strategy of the measured collection "
                             "(default: the one in --config)")
    parser.add_argument("--config", default=None,
                        help="configuration whose query phase is measured "
                             "(default: the FULL pipeline, configs/full_<dataset>.yaml)")
    parser.add_argument("--frames", type=int, default=320,
                        help="grid frames sampled from the corpus for the models")
    parser.add_argument("--caption-frames", type=int, default=64,
                        help="frames per pass of the caption generators; LLaVA "
                             "runs one at a time and dominates the running time, "
                             "so this is the knob that decides how long the "
                             "measurement takes")
    parser.add_argument("--crops", type=int, default=512,
                        help="face crops sampled for the crop-level components")
    parser.add_argument("--only", nargs="*", default=None,
                        help="measure only these components: "
                             + ", ".join(cost.COMPONENTS))
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    args = parser.parse_args()

    if args.only and (unknown := set(args.only) - set(cost.COMPONENTS)):
        raise SystemExit(f"unknown component(s): {sorted(unknown)}")

    # the FULL pipeline by default: the query column of the thesis table needs a
    # time for EVERY component, and only a configuration that enables them has one
    config = args.config or f"configs/full_{args.dataset}.yaml"
    # the collection on disk belongs to a strategy, and after E1 that is no longer
    # the default one -- take it from the configuration instead of a stale constant
    strategy = args.strategy or load_experiment(config).collection_strategy
    started = time.perf_counter()

    hours = cost.corpus_hours(args.dataset, args.split)
    print(f"{args.dataset} / {args.split}: {hours:.2f} h of corpus")
    print(f"units per hour: {cost.units_per_hour()}, "
          f"faces/h: {cost.faces_per_hour(args.dataset, args.split):.0f}\n")

    data = {"dataset": args.dataset, "split": args.split,
            "strategy": strategy, "corpus_hours": round(hours, 3),
            "repeats": cost.REPEATS, "warmup": cost.WARMUP}

    if not args.skip_extraction:
        print("=== extraction ===")
        frames = cost.sample_frames(args.dataset, args.split, args.frames)
        crops = cost.sample_crops(args.dataset, args.split, args.crops)
        print(f"sample: {len(frames)} frames, {len(crops)} face crops\n")
        data["extraction"] = cost.extraction_costs(
            args.dataset, args.split, frames, crops, only=args.only,
            caption_frames=args.caption_frames)
        del frames, crops
        cost.free_gpu()

    print("\n=== index size on disk ===")
    data["index"] = cost.index_sizes(args.dataset, args.split, strategy)
    for entry in data["index"]:
        if entry["mb"]:
            print(f"  {entry['component']:<30}{entry['mb']:>9.1f} MB"
                  f"{entry['mb_per_hour']:>9.1f} MB/h")

    if not args.skip_query:
        print("\n=== query phase ===")
        data["query"] = cost.query_times(config, args.split)

    data["elapsed_min"] = round((time.perf_counter() - started) / 60, 1)
    save_measurement("cost", data)
    print(f"\ndone in {data['elapsed_min']:.1f} min")


if __name__ == "__main__":
    main()
