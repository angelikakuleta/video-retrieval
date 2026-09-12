"""Writes the samples of phrase-class pairs a person judges (section 07).

    python scripts/make_threshold_samples.py --split dev

Development material only, and the assertion below is what makes that a property
of the code rather than a promise.

Why by hand at all: the threshold cannot be calibrated against the Kinetics class
of a clip. The description and the label often name the same action in different
words, and the clip is often labelled with an action the description never
mentions -- where the phrase IS the class name, character for character, the
label still agrees only about two thirds of the time. That number is the ceiling
of the automatic path, which is why the truth is now established by a person.

Writes four files to data/interim/threshold/ -- not under a dataset directory,
because the threshold belongs to no collection:

    threshold_sample_<vocabulary>.csv    the pairs to judge
    threshold_population.json            every occurrence they were drawn from

The sample files carry no truth column and no clip label, because a judgment
that can see the answer is not a judgment. What happens to them afterwards is
described in the header of notebooks/experiments/threshold.ipynb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.measurement import text_bridge


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev",
                        help="the development part, and nothing else")
    parser.add_argument("--encoder", default=text_bridge.ENCODER)
    parser.add_argument("--measure", default="top1", choices=["top1", "z"],
                        help="the confidence the pairs are ranked by")
    parser.add_argument("--out", default=str(text_bridge.SAMPLE_DIR))
    args = parser.parse_args()

    # gate 1, in the code rather than in a promise
    if args.split != "dev":
        raise SystemExit(
            f"--split {args.split!r} refused: the threshold is calibrated on the "
            "development part only. The test split stays unreachable until the "
            "configuration is frozen, and this calibration is what freezes it.")

    out = text_bridge.build_threshold_samples(
        part=args.split, encoder=args.encoder, measure=args.measure, out_dir=args.out)

    print(f"\n{'sample':32} {'occurr':>6}  {'pairs':>6}   {'judge':>6}   "
          "exact + the ten intervals")
    for name, entry in out["samples"].items():
        drawn, held = entry["sampled"], entry["strata"]
        if "all" in drawn:
            bands = f"all {drawn['all']}"
        else:
            bins = [f"b{i:02d}" for i in range(1, text_bridge.SAMPLE_BINS + 1)]
            bands = (f"{drawn.get('exact', 0)} of {held.get('exact', 0)}  |  "
                     + " ".join(str(drawn.get(b, 0)) for b in bins))
        print(f"{name:32} {entry['n_occurrences']:6}  {entry['n_pairs']:6}   "
              f"{entry['n_sample']:6}   {bands}")
    if out["datasets_missing"]:
        print(f"\nNOT in any sample (no development query file on disk): "
              f"{', '.join(out['datasets_missing'])}")
    print(f"\npopulation -> {out['population_file']}")
    print("next (rules: tools/judge/README.md):")
    print("  python tools/judge/judge_pairs.py --samples data/interim/threshold/threshold_sample_*.csv --output data/interim/threshold/threshold_judgments.csv")
    print("then: python scripts/measure_threshold.py --split dev")


if __name__ == "__main__":
    main()
