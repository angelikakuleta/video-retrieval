"""Creates and inspects the requirement-tag file of a series.

    python scripts/prepare_data/query_tags.py office --init --check --split dev

--init writes ``<series>_query_tags_new.csv`` next to the file you fill in: a
MERGE, not a fresh skeleton. Filled cells are carried over, missing rows added,
rows no longer accepted dropped, and nothing you typed is overwritten - so it is
safe to copy over the working name. The notebook does the same.

Context columns (episode, split) follow the register; the query text does not.
Typos are corrected here and the query files are built from this file, so a
wording already in it wins and a diverging register row is only reported. Tags
marked on an annotation in the annotator are pre-filled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation import registry, tags
from src.data import datasets


def register_path(dataset: str) -> Path:
    name = datasets.dataset_dir(dataset)
    return ROOT / "data" / "interim" / name / f"{name}_annotations.csv"


def tags_path(dataset: str) -> Path:
    name = datasets.dataset_dir(dataset)
    return tags.path_for(name, ROOT / "data" / "annotations" / name)


def report_annotator(register_rows: list[dict]) -> None:
    """Prints what the annotator's labels bring in, tag by tag."""
    summary = tags.annotator_report(register_rows)
    marked = {tag: count for tag, count in summary["marked"].items() if count}
    if marked:
        print("tags marked in the annotator:")
        for tag, count in marked.items():
            episodes = summary["episodes"][tag]
            print(f"  {tag:<18}{count:>4} annotations in {len(episodes)} episodes"
                  f" ({', '.join(episodes)})")
        print("  episodes above count as reviewed for their tag; their remaining"
              " annotations get 0")
    else:
        print("no requirement tag marked in the annotator yet")
    if summary["unknown"]:
        print("  labels outside the vocabulary (ignored): "
              + ", ".join(summary["unknown"]))


def report(rows: dict[int, dict], split: str | None) -> None:
    """Prints how much of the file is filled in."""
    stats = tags.coverage(rows, split)
    scope = split or "all splits"
    print(f"coverage ({scope}): {stats['filled']}/{stats['total']} queries filled")
    if not stats["per_episode"]:
        return
    print(f"  {'episode':<10}{'filled':>8}{'total':>8}")
    for episode, entry in stats["per_episode"].items():
        print(f"  {episode:<10}{entry['filled']:>8}{entry['total']:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["tbbt", "office"],
                        help="series whose tag file is handled")
    parser.add_argument("--init", action="store_true",
                        help="add the missing rows, keeping every filled cell")
    parser.add_argument("--check", action="store_true",
                        help="validate the values and report the coverage")
    parser.add_argument("--split", choices=["dev", "test"], default=None,
                        help="limit the report to one split")
    args = parser.parse_args()
    if not (args.init or args.check):
        parser.error("nothing to do: pass --init, --check, or both")

    source = register_path(args.dataset)
    if not source.exists():
        raise SystemExit(f"no register: {source} - run the annotation notebook first")
    target = tags_path(args.dataset)
    fresh_target = target.with_name(target.stem + "_new.csv")

    register_rows = registry.load(source)
    accepted = registry.accepted(register_rows)
    existing = tags.load(target)

    if args.init:
        fresh, counts = tags.skeleton(
            accepted, existing, tags.from_register(accepted))
        tags.save(fresh, fresh_target)
        report_annotator(accepted)
        print(f"tag file -> {fresh_target.relative_to(ROOT)}")
        print(f"  copy it over {target.name} by hand once you are done")
        print(f"  rows: {len(fresh)}  (added {counts['added']}, kept {counts['kept']},"
              f" dropped {counts['dropped']})")
        print(f"  cells taken from the annotator: {counts['from_annotator']}")
        if counts["retexted"]:
            print(f"  NOTE: the register wording differs for {len(counts['retexted'])}"
                  " rows - this file keeps yours, check they still mean the same:")
            print("    " + ", ".join(str(i) for i in counts["retexted"][:20]))
        existing = fresh

    if args.check:
        problems = tags.validate(existing)
        for problem in problems:
            print(f"  INVALID {problem}")
        if problems:
            print(f"{len(problems)} invalid cells - fix them before rebuilding the queries")
        report(existing, args.split)
        if problems:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
