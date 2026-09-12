"""Which heads the phrase rules pick out of the development queries.

    python scripts/measure_phrase_heads.py

Diagnostic, not a decision: it documents in chapter 7 how far the object and
expression rules reach on series material, and the rules are frozen before the
threshold is measured, so nothing here may be used to change them.

Development queries of both series only. Writes
results/measurements/phrase_heads_<date>_<time>.json, whose keys read
``phrase_heads.<kind>.<dataset>``. When scripts/measure_threshold.py arrives it
calls the same function and folds the payload into the threshold measurement
under that same key, so this script becomes the standalone way of asking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.measurement.text_bridge import KINDS, SERIES, TOP_HEADS, phrase_heads
from src.utils.notebook import save_measurement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(SERIES),
                        help="series to count over (default: both)")
    parser.add_argument("--top", type=int, default=TOP_HEADS,
                        help="heads listed per kind and dataset")
    # the phrase step runs on development queries only (gate 1): the test part
    # stays out of reach until the configuration is frozen
    args = parser.parse_args()

    payload = phrase_heads(datasets=tuple(args.datasets), kinds=KINDS,
                           split="dev", top=args.top)
    for kind in KINDS:
        for dataset, entry in payload[kind].items():
            shown = ", ".join(f"{head} ({count})" for head, count in entry["heads"][:8])
            share = entry["queries_with_phrase"] / entry["queries"] if entry["queries"] else 0
            print(f"\n{kind} / {dataset}: {entry['queries_with_phrase']} of "
                  f"{entry['queries']} queries ({100 * share:.1f}%), "
                  f"{entry['phrases']} phrases")
            print(f"  {shown}")
    save_measurement("phrase_heads", payload)


if __name__ == "__main__":
    main()
