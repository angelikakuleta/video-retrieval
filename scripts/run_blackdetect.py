"""Finds the stretches of (almost) black picture in every episode.

    python scripts/run_blackdetect.py tbbt office

Output: ``data/cache/black/<dataset>_black.csv``, read through
:class:`src.segmentation.timeline.Timeline` by every component. Optional: the
experiment runner guarantees the same artifact for the episodes it needs. All
logic lives in :mod:`src.segmentation.black`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import black


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="dataset names (tbbt, office)")
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="only these episodes (e.g. s03e02)")
    parser.add_argument("--force", action="store_true",
                        help="recompute episodes already present in the file")
    parser.add_argument("--pix-th", type=float, default=black.PIX_TH)
    parser.add_argument("--pic-th", type=float, default=black.PIC_TH)
    parser.add_argument("--min-duration", type=float, default=black.MIN_DURATION)
    args = parser.parse_args()

    for dataset in args.datasets:
        black.ensure_black(dataset, episodes=args.episodes, force=args.force,
                           pix_th=args.pix_th, pic_th=args.pic_th,
                           min_duration=args.min_duration)


if __name__ == "__main__":
    main()
