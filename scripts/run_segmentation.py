"""Builds the segment cache for experiment E1.

    python scripts/run_segmentation.py configs/e1b_office.yaml configs/e1c_office.yaml

Configurations of one dataset share a single decoding pass per episode, so
passing both shot detectors together halves the work. Resumable per episode;
--force recomputes. The experiment runner calls the same function, so running
this first is optional.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.build import ensure_segments
from src.utils.config import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="YAML configuration files")
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="only these episodes (e.g. s02e03)")
    parser.add_argument("--force", action="store_true",
                        help="recompute episodes already present in the cache")
    args = parser.parse_args()

    missing = [f for f in args.configs if not Path(f).exists()]
    if missing:
        raise SystemExit("configuration file not found: " + ", ".join(missing))

    grouped = defaultdict(dict)
    for file in args.configs:
        config = load_experiment(file)
        if config.segmentation is None:
            raise SystemExit(f"{file}: the {config.dataset} dataset is not segmented")
        grouped[config.dataset][config.segmentation.strategy] = config.segmentation

    for dataset, strategies in grouped.items():
        ensure_segments(dataset, strategies,
                        episodes=args.episodes, force=args.force)


if __name__ == "__main__":
    main()
