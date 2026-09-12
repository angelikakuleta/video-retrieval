"""The judgment pool: checking how complete the annotations are (chapter 6.7.6).

An annotation says which fragment answers a query. It does not say that no OTHER
fragment answers it, and for a whole episode searched instead of the short clip
the annotation was written against, that difference is real: a system can return
a correct fragment and be scored wrong because nobody wrote that one down.

So a sample of queries is taken, the fragments the main configurations actually
returned for them are pooled, a person judges those, and the measures are
recomputed on the completed annotation. What the control produces is not a better
Recall -- it is the size of the gap between the two, and whether the ORDER of the
contributions survives closing it.

Three properties make it a control rather than a second experiment, and all three
are written down before anything is measured:

* the sample is 10% of the test queries per series, drawn with a fixed seed;
* the set of configurations is fixed, and it is the MAIN ones -- no X-CLIP, no
  segmentation variants, no E4-D. A pool fed by every variant would measure the
  union of their mistakes rather than the annotation;
* the file the judge sees carries no configuration column and its rows are
  shuffled, so no answer can be traced back to the variant that produced it.

VATEX has no pool. Its clips are ten seconds long and its annotation is the
description of the clip itself, so the gap this control measures cannot open.

Nothing here writes to ``data/annotations/``: the pool goes to the working
directory, the judgments come back by hand.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from src.utils.experiments import (ADDED_SIGNALS, BASE, FULL, SERIES,
                                   SWAPPED_SIGNALS, no_label, swap_label)

#: share of the test queries drawn per series, fixed before the measurement
POOL_SHARE = 0.10

#: and the seed of that draw, so the pool is the same one every time
POOL_SEED = 1234

#: The configurations whose top-10 the pool is made of: the MAIN ones. Derived
#: from the facts table rather than retyped, so a signal added to the pipeline
#: cannot quietly fall out of the pool -- but the resulting set is fixed before
#: the measurement and pinned by a test, which is what "written down" means here.
#:
#: Deliberately absent: E2-Ap and E2-B/E2-Bp (X-CLIP controls), E1-B and E1-C
#: (segmentation variants) and E4-D. The first two answer questions about
#: representations and about splitting a recording, not about what an annotation
#: missed; E4-D scores a different collection per query, so its top-10 is not
#: comparable with the others.
POOL_LABELS: tuple[str, ...] = (
    BASE, "E2-C", "E3-B", "E3-C", "E4-B", "E4-C", "E5-B", "E5-C", "E6-B", FULL,
    *(no_label(signal) for signal in ADDED_SIGNALS),
    *(swap_label(signal) for signal in SWAPPED_SIGNALS),
)

#: what the judge is given. No configuration column: which variant returned a
#: fragment must not be visible, or the judging could favour one of them.
#: ``relevant`` is written EMPTY and is the column the judge fills, so judging is
#: editing this file rather than transcribing it into another one.
POOL_COLUMNS = ("desc_id", "desc", "fragment", "video_file", "start", "end",
                "relevant")

#: and what has to be there when it comes back. The judged file keeps the other
#: columns -- nothing reads them, and they are what makes a row reviewable.
JUDGMENT_COLUMNS = ("desc_id", "fragment", "relevant")
ANSWERS = ("yes", "no")

#: the pool is written under this suffix and the judged file drops it: the
#: answers land in data/annotations/<dataset>/<dataset>_pool_judgments.csv, which
#: is the same name without "_new". A build therefore cannot overwrite answers,
#: and which file is which is visible from the name alone.
POOL_SUFFIX = "_new"


def pool_file(dataset: str, split: str = "test",
              suffix: str = POOL_SUFFIX) -> Path:
    """Where the pool of one dataset is written, for the judge to fill in.

    Named after the file the answers become: dropping ``_new`` and moving the
    file to ``data/annotations/<dataset>/`` is the whole of step three. One
    rename, nothing to retype, and no chance of the two files diverging in the
    only column that matters.

    The suffix also keeps a rebuild off the judge's work: the set of
    configurations grows as the test phase arrives, so a pool built today is not
    the pool built last week, and the build must not land on top of answers.
    """
    from src.data import datasets

    name = datasets.dataset_dir(dataset)
    stem = (f"{name}_pool_judgments" if split == "test"
            else f"{name}_pool_judgments_{split}")
    return datasets.ROOT / "data" / "interim" / name / "work" / f"{stem}{suffix}.csv"


def judgments_file(dataset: str) -> Path:
    """Where the judged pool comes back. Read only -- nothing here writes it."""
    from src.data import datasets

    name = datasets.dataset_dir(dataset)
    return datasets.ROOT / "data" / "annotations" / name / f"{name}_pool_judgments.csv"


def sample_queries(queries, share: float = POOL_SHARE,
                   seed: int = POOL_SEED) -> list[int]:
    """The ``desc_id`` of the queries the pool covers, drawn once and for all.

    Drawn from the sorted identifiers, so the sample depends on the query set and
    the seed and on nothing else -- not on the order a file happened to be
    written in.
    """
    ids = sorted({int(query.desc_id) for query in queries})
    wanted = min(len(ids), max(1, round(share * len(ids))))
    return sorted(random.Random(seed).sample(ids, wanted))


def _fragments_of(dataset: str, strategy: str, split: str) -> dict[str, dict]:
    """``fragment id -> {video_file, start, end}`` for one dataset and strategy."""
    from src.data import datasets
    from src.evaluation.relevance import fragment_id
    from src.segmentation import segments

    path = datasets.segments_csv(strategy, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"no segment cache for {dataset!r} under {strategy!r}: {path}. "
            "The pool needs the times of the fragments it lists.")
    return {fragment_id(row["episode"], row["segment_id"]): row
            for row in segments.load(path) if row["split"] == split}


def _already_relevant(run: dict) -> dict[int, set[str]]:
    """The relevance the run itself used, read from its own ``relevance.csv``.

    Taken from the run rather than recomputed: what the pool must exclude is
    exactly what was already counted as correct when the ranking was scored.
    """
    out: dict[int, set[str]] = defaultdict(set)
    path = Path(run["dir"]) / "relevance.csv"
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            out[int(row["desc_id"])].add(row["fragment_id"])
    return out


def build_pool(split: str = "test", share: float = POOL_SHARE, seed: int = POOL_SEED,
               datasets_: tuple[str, ...] = SERIES, runs: dict | None = None,
               suffix: str = POOL_SUFFIX, log=print) -> dict:
    """Writes the pool of every series dataset and says what went into it.

    ``runs`` is ``{label: {dataset: run}}`` as :func:`runs.load_runs` returns it;
    loaded here when not given. A configuration with no run is named and skipped
    rather than fatal -- the test phase arrives one variant at a time, and a pool
    built on nine of eighteen is still a pool, as long as the report says so.
    """
    from src.data.queries import load_queries
    from src.evaluation import runs as runs_module

    if runs is None:
        runs = runs_module.load_runs(split, log=log)

    out: dict[str, dict] = {}
    for dataset in datasets_:
        queries = load_queries(dataset, split)
        by_id = {int(query.desc_id): query for query in queries}
        sampled = sample_queries(queries, share, seed)

        present = [label for label in POOL_LABELS if dataset in runs.get(label, {})]
        missing = [label for label in POOL_LABELS if label not in present]
        if not present:
            log(f"{dataset}: none of the {len(POOL_LABELS)} pool configurations has a "
                f"run on split {split!r} - no pool written")
            out[dataset] = {"queries": len(sampled), "labels": [], "missing": missing,
                            "rows": 0, "file": None}
            continue

        # the NAMES, not the blocks: a set of segmentation blocks is a set of
        # dicts, which Python refuses to build, so a second configuration with a
        # run would raise here
        blocks = [runs[label][dataset]["config"].get("segmentation")
                  for label in present]
        names = {block.get("strategy") for block in blocks
                 if isinstance(block, dict)} or {"whole_clip"}
        if len(names) > 1:
            raise ValueError(
                f"{dataset}: the pool configurations do not share one segmentation "
                f"strategy ({sorted(names)}) - their fragments are not the same "
                "objects and their top-10 cannot be pooled")
        fragments = _fragments_of(dataset, names.pop(), split)

        relevant: dict[int, set[str]] = defaultdict(set)
        candidates: dict[int, set[str]] = defaultdict(set)
        for label in present:
            run = runs[label][dataset]
            for desc_id, found in _already_relevant(run).items():
                relevant[desc_id] |= found
            for record in run["per_query"]:
                desc_id = int(record["desc_id"])
                if desc_id in by_id and desc_id in set(sampled):
                    candidates[desc_id] |= {entry["fragment"] for entry in record["top"]}

        rows, unknown = [], set()
        for desc_id in sampled:
            for fragment in sorted(candidates.get(desc_id, ())):
                if fragment in relevant.get(desc_id, ()):
                    continue                 # already counted correct
                entry = fragments.get(fragment)
                if entry is None:
                    unknown.add(fragment)
                    continue
                rows.append({"desc_id": desc_id, "desc": by_id[desc_id].desc,
                             "fragment": fragment, "video_file": entry["video_file"],
                             "start": f"{entry['start']:.2f}",
                             "end": f"{entry['end']:.2f}", "relevant": ""})
        # shuffled, so the order carries no trace of which configuration
        # returned what, nor of how highly it ranked it
        random.Random(seed).shuffle(rows)

        path = pool_file(dataset, split, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(POOL_COLUMNS), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

        covered = {row["desc_id"] for row in rows}
        empty = [desc_id for desc_id in sampled if desc_id not in covered]
        out[dataset] = {"queries": len(sampled), "sampled": list(sampled),
                        "labels": present, "missing": missing,
                        "rows": len(rows), "file": str(path),
                        "without_candidates": empty,
                        "unknown_fragments": sorted(unknown)}
        log(f"{dataset}: {len(rows)} fragments to judge for {len(sampled)} queries "
            f"from {len(present)} configuration(s) -> {path.name}")
        if missing:
            log(f"{dataset}: no run for {', '.join(missing)}")
        if empty:
            log(f"{dataset}: {len(empty)} sampled query(ies) contributed no row - "
                "their whole top-10 is already in the annotation, so they drop out "
                f"of the recomputation: {empty}")
        if unknown:
            log(f"{dataset}: {len(unknown)} fragment(s) of the rankings are not in the "
                "segment cache and were left out")
    return out


def load_judgments(source) -> list[dict]:
    """The judged pool, from a CSV path or from anything already iterable."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return []
        with open(path, encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=";")]
    return [dict(row) for row in source]


def pool_ids(judgments) -> set[int]:
    """The queries the pool covered, judged either way.

    The recomputation runs on THESE and no others. A query outside the pool has
    the annotation it always had, so mixing it in would average a completed
    measure with an incomplete one and report neither.
    """
    return {int(row["desc_id"]) for row in load_judgments(judgments)
            if str(row.get("desc_id", "")).strip()}


#: how deep a run's ranking is stored (``run.TOP_SAVED``). The recomputation
#: reads the ranking back from ``per_query.jsonl``, so it can answer Recall@k
#: only down to this depth -- and it refuses rather than under-report beyond it.
DEPTH = 10


def recomputed_hits(run: dict, judgments, k: int = DEPTH) -> dict[int, tuple[int, int]]:
    """``{desc_id: (before, after)}`` for every POOLED query of the run.

    The per-query half of :func:`recompute`. A CONTRIBUTION recomputed over the
    pool is a difference between two configurations, so it has to be read at the
    unit of inference -- and a pair of collection averages cannot be grouped into
    episodes after the fact. Hence the query-level answer lives here and both
    readings are built on it.

    "After" is read off the stored top-``k`` of the ranking rather than
    recomputed from scores: a fragment the pool judged correct raises the query
    to a hit exactly when it is already inside that list. The run stores
    :data:`DEPTH` positions, which is why a deeper ``k`` is refused instead of
    silently answered with a shorter list.
    """
    if k > DEPTH:
        raise ValueError(
            f"a run stores {DEPTH} positions of its ranking, so Recall@{k} cannot "
            "be recomputed from it; rerun the configuration to score deeper")
    wanted = pool_ids(judgments)
    augmented = augment_relevance(judgments, _already_relevant(run))

    out: dict[int, tuple[int, int]] = {}
    for record in run["per_query"]:
        desc_id = int(record["desc_id"])
        if desc_id not in wanted:
            continue
        was = int(record[f"recall@{k}"] > 0)
        top = {entry["fragment"] for entry in record["top"][:k]}
        now = int(bool(top & augmented.get(desc_id, set())))
        if now < was:
            raise ValueError(
                f"query {desc_id}: the stored recall@{k} says hit and the stored "
                "top does not. The judgments belong to a different run than the "
                "one passed in, or the ranking was written by another version.")
        out[desc_id] = (was, now)
    return out


def recompute(run: dict, judgments, k: int = DEPTH) -> dict:
    """Recall@k of one run before and after the pool, over the POOLED queries.

    The whole point of the control is the difference between the two, so both
    sides are computed the same way on the same queries. A query outside the
    pool keeps the annotation it always had; averaging it in would mix a
    completed measure with an incomplete one and report neither, so it is left
    out (see :func:`pool_ids`).

    Returns ``{"n", "before", "after", "gained", "added"}`` -- ``gained`` is how
    many queries the pool turned from a miss into a hit, ``added`` how many
    fragments the judgments contributed to those queries.

    ``before`` and ``after`` are LEVELS of one and the same configuration, so
    both are means over queries and so is the gap between them: what this control
    measures is how far the annotation moved that configuration, not how two
    configurations compare. A difference BETWEEN configurations over the same
    pool is a different question and is read per episode -- it starts from
    :func:`recomputed_hits`.
    """
    hits = recomputed_hits(run, judgments, k)
    original = _already_relevant(run)
    augmented = augment_relevance(judgments, original)
    counted = len(hits)
    before = sum(was for was, _ in hits.values())
    after = sum(now for _, now in hits.values())
    return {"n": counted,
            "before": (before / counted) if counted else None,
            "after": (after / counted) if counted else None,
            "gained": after - before,
            "added": sum(len(augmented.get(desc_id, set())
                             - original.get(desc_id, set())) for desc_id in hits)}


def augment_relevance(judgments, relevance: dict[int, set[str]] | None = None
                      ) -> dict[int, set[str]]:
    """The relevance sets with the fragments judged correct added to them.

    Returns a NEW mapping; the one passed in is left alone, because the same
    relevance is used to score the runs both before and after and the difference
    between the two is the whole result.

    ``no`` is not an erasure. A fragment the pool rejected was never in the
    annotation to begin with, and rejecting one that IS in it would be a
    correction of the annotation -- a different act, done by hand, in the
    annotation itself.
    """
    out: dict[int, set[str]] = defaultdict(set)
    for desc_id, found in (relevance or {}).items():
        out[int(desc_id)] |= set(found)
    for row in load_judgments(judgments):
        if str(row.get("relevant", "")).strip().lower() == "yes":
            out[int(row["desc_id"])].add(str(row["fragment"]).strip())
    return dict(out)
