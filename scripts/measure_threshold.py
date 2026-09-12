"""Measures the phrase-matching threshold tau on the VATEX development clips.

    python scripts/measure_threshold.py --split dev

Development material only, and the assertion below is what makes that a property
of the code rather than a promise. The test split stays out of reach until the
configuration is frozen (gate 1), and tau is one of the things that has to be
frozen first: nothing that scores against a closed vocabulary may run before it.

Text in, one number out. The ground truth used HERE is the Kinetics class of a
clip: for a clip whose class is one of the 400, the description ought to name
that class. That truth UNDERSTATES precision, and by a measured amount -- where
the phrase is the class name character for character, the label agrees only
about two thirds of the time -- so this path no longer sets the threshold. It is
kept, and reported, because the thesis shows both and explains the difference.

The threshold itself is read off pairs judged by hand, which this script only
reads back -- the procedure that produces them is described in the header of
notebooks/experiments/threshold.ipynb. No ranking is computed and no Recall is
read either way, so both paths are allowed to run while the test split is still
untouchable.

Writes results/measurements/threshold_<date>_<time>.json through
save_measurement, and nothing else. The tables are printed by
notebooks/experiments/threshold.ipynb -- scripts compute, notebooks read, and a
second renderer of the same numbers is a second thing to keep true. What keeps a
threshold out of a configuration is configs/frozen.yaml, written by hand and
committed, and the check in src/runners/run.py that refuses a run without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.measurement import text_bridge
from src.utils.notebook import save_measurement

#: judged with tools/judge/judge_pairs.py and moved here by the author. Read only
#: -- no code in this repository writes to data/annotations/ (CLAUDE.md). Not
#: under a dataset directory: the threshold belongs to no collection.
JUDGMENTS = ROOT / "data" / "annotations" / "threshold_judgments.csv"


def _number(value, digits=3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _share(value, digits=1) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev",
                        help="the development part, and nothing else")
    parser.add_argument("--encoder", default=text_bridge.ENCODER)
    parser.add_argument("--datasets", nargs="+", default=["tbbt", "office", "vatex"],
                        help="development datasets the activation table covers")
    args = parser.parse_args()

    # gate 1, in the code rather than in a promise: the test split is out of
    # reach until the configuration is frozen, and tau is what freezes it
    if args.split != "dev":
        raise SystemExit(
            f"--split {args.split!r} refused: the threshold is measured on the "
            "development part only. The test split stays unreachable until the "
            "configuration is frozen, and this measurement is what freezes it.")

    if not JUDGMENTS.exists():
        print(f"no judgments at {JUDGMENTS} - nothing to read a threshold off yet; "
              "the activation table and the rejected phrases stay unanswered")

    # `measure` reads the judged thresholds itself, BEFORE it computes the
    # activation table, because that table is computed at them. Adding the block
    # here afterwards is what left both tables empty.
    data = text_bridge.measure(part=args.split, encoder=args.encoder,
                               datasets=tuple(args.datasets), judgments=JUDGMENTS)

    save_measurement("threshold", data)

    old = data[data["chosen_measure"]]
    print(f"\nold path (agreement with the clip label, an UNDERSTATEMENT): "
          f"tau({data['chosen_measure']}) = {old['tau']} on "
          f"{data['material']['descriptions']} descriptions")

    judged = data.get("judged")
    if judged:
        print("\nthreshold per vocabulary, from the judged pairs - THIS is the result")
        print(f"{'vocabulary':28}{'threshold':>11}{'precision':>11}"
              f"{'phrases':>10}{'queries':>10}{'unresolved':>12}")
        for name, entry in judged.items():
            print(f"{name:28}{_number(entry['threshold'], 4):>11}"
                  f"{_number(entry['precision_at_threshold'], 3):>11}"
                  f"{_share(entry['coverage_phrases']):>10}"
                  f"{_share(entry['coverage_queries']):>10}"
                  f"{entry['n_unresolved'] if entry['n_unresolved'] is not None else '-':>12}")
            if entry.get("reason"):
                print(f"{'':28}{entry['reason']}")
    print("\ntables: notebooks/experiments/threshold.ipynb")


if __name__ == "__main__":
    main()
