"""Runs one experiment variant on one split.

    python scripts/run_experiment.py configs/e1a_office.yaml --split dev

The split is a runtime parameter, not part of the variant's identity: the same
untouched file serves the development phase and, after the freeze, the test
phase. Whatever is missing -- black stretches, segments, embeddings, component
features, indexes -- is computed and cached on the way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runners.run import execute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML configuration file of the variant")
    parser.add_argument("--split", required=True, choices=["dev", "test"],
                        help="evaluated split; test only after the config freeze")
    parser.add_argument("--force-index", action="store_true",
                        help="rebuild the FAISS index even when up to date")
    args = parser.parse_args()
    if not Path(args.config).exists():
        raise SystemExit(f"configuration file not found: {args.config}")
    # Gate 1 (section 02), as a property of the code rather than a promise. The
    # test split stays out of reach until every decision of the development
    # phase is written into configs/frozen.yaml -- a run on it before that would
    # be a run whose composition could still change afterwards, which is the one
    # thing the test phase must never be.
    if args.split == "test":
        from src.utils import frozen as frozen_module

        frozen_module.require(frozen_module.STAGE_TWO,
                              "a run on the test split (gate 1)")

    execute(args.config, args.split, force_index=args.force_index)


if __name__ == "__main__":
    main()
