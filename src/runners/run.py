"""One experiment run: configuration + split -> an immutable run directory.

The runner only loads the configuration, builds the pipeline out of ready
components and measures -- no component logic lives here. Everything needed
to reproduce or audit the run lands in its directory:

    results/runs/<experiment>_<dataset>_<split>_YYYYMMDD_HHMMSS/
        config.resolved.yaml   the merged configuration (split included)
        metadata.json          package versions, input hashes, seed
        relevance.csv          the Rel(q) sets used by this run
        per_query.jsonl        ranking and metrics per query
        metrics.json           aggregates (overall, per episode, per tag)
        signals.npz            standardized signals, for the weight sensitivity
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

from src.data import datasets, manifest
from src.data.queries import episode_of, load_queries
from src.evaluation.metrics import PER_QUERY_KEYS, query_metrics, summarize
from src.evaluation.relevance import relevance_sets
from src.retrieval.fusion import combine
from src.segmentation import segments as seg
from src.utils.config import ExperimentConfig, dump_resolved, load_experiment

TOP_SAVED = 10
VERSIONED_PACKAGES = ("torch", "open_clip_torch", "faiss-cpu", "numpy",
                      "opencv-python", "pydantic", "transnetv2-pytorch")


def _require_matching(config, config_file: Path | str) -> None:
    """Refuses a run whose closed-vocabulary signals have no threshold to use.

    The schema lets the block be absent, because run_features.py and
    measure_cost.py read the same files and neither produces a number the
    threshold could bend. A RUN does produce such numbers, so the check belongs
    here: a missing block would otherwise mean scoring on a threshold nobody
    measured.
    """
    if config.needs_matching and config.matching is None:
        raise ValueError(
            f"{config_file}: this configuration scores against a closed "
            "vocabulary (objects, hsemotion expressions or motion) and needs a "
            "matching block with measure and threshold. Every file that needs "
            "one is generated and carries it, so this is either a hand-written "
            "file that should not hold such a signal at all, or a generated one "
            "from before configs/frozen.yaml held the measure: fill in "
            "configs/frozen.yaml and run scripts/make_configs.py --execute.")


def execute(config_file: Path | str, split: str, force_index: bool = False) -> Path:
    """Runs one experiment and returns the path of its run directory."""
    config = load_experiment(config_file, split)
    _require_matching(config, config_file)
    _seed_everything(config.seed)

    run_dir = _new_run_dir(config)
    print(f"run -> {run_dir.relative_to(datasets.ROOT)}")

    # --- collection, components and encoder (heavy imports stay in the run) --
    from src.features import encoders
    from src.retrieval.pipeline import Pipeline
    from src.runners import components, stages

    print("components: " + ", ".join(config.components.enabled_signals()))
    encoder_factory = encoders.lazy(config.components.scene_embedding.model)
    collection, inputs, context = stages.ensure_collection(config, encoder_factory,
                                                           force=force_index)
    signals = components.build_signals(
        config, collection, context["rows"], context["episodes"],
        context["timelines"], encoder_factory, force=force_index)

    # --- queries and relevance sets -----------------------------------------
    queries = load_queries(config.dataset, split)
    strategy = config.collection_strategy
    rows = seg.load(datasets.segments_csv(strategy, config.dataset))
    rows = [r for r in rows if r["split"] == split]
    relevance = relevance_sets(queries, rows)
    in_collection = set(collection.vids)
    evaluated = [q for q in queries if relevance[q.desc_id] & in_collection]
    skipped = [q.desc_id for q in queries if not (relevance[q.desc_id] & in_collection)]
    if skipped:
        print(f"queries without a relevant fragment in the collection: {len(skipped)}")
    if not evaluated:
        # every metric would be an empty average and the run would look finished
        raise RuntimeError(
            f"no query of split {split!r} has a relevant fragment in the collection "
            f"({len(queries)} queries, {collection.size} fragments) - the run would "
            "report nothing at all")

    # --- ranking and metrics ------------------------------------------------
    pipeline = Pipeline(collection, signals, weights=config.scoring.weights.manual)
    timings: list[float] = []
    matrices, active = pipeline.signal_matrices(evaluated, timings)
    scores = combine(matrices, active, pipeline.base)

    # E4-D re-ranks the fifty best of the base with a detector asked the query's
    # own phrases (section 08). A scoring STAGE, so it changes the order and
    # nothing else: no row in signals.npz, no weight in the fusion above.
    orders, detection_records = _query_time_detection(
        config, scores, collection, evaluated, context, relevance)

    per_query, groups = [], defaultdict(list)
    in_k400 = _k400_membership(config)
    for position, (query, row) in enumerate(zip(evaluated, scores)):
        order = orders[position]
        ranking = [collection.vids[j] for j in order]
        metrics = query_metrics(ranking, relevance[query.desc_id])
        record = {
            "desc_id": query.desc_id,
            "episode": episode_of(query),
            "n_relevant": len(relevance[query.desc_id]),
            **metrics,
            "top": [{"fragment": collection.vids[j], "score": round(float(row[j]), 4)}
                    for j in order[:TOP_SAVED]],
        }
        if detection_records is not None:
            # the one field outside PER_QUERY_KEYS, and the reason is that E4-D
            # is not a signal: the activation table has nowhere else to read it
            record["query_time_detection"] = detection_records[position]
        per_query.append(record)
        # one group per recording is a useful cut for eighteen episodes and a
        # few thousand rows of noise for a collection of single-clip recordings
        if config.dataset != "vatex":
            groups["episode:" + record["episode"]].append(metrics)
        if "requirements" in config.evaluation.report_by:
            for tag in (query.requirements or ["unknown"]):
                groups["requirements:" + tag].append(metrics)
        if "complexity" in config.evaluation.report_by:
            groups["complexity:" + (query.complexity or "unknown")].append(metrics)
        if in_k400 is not None:
            known = in_k400.get(query.vid_name)
            label = "unknown" if known is None else ("yes" if known else "no")
            groups["kinetics_vocab:" + label].append(metrics)

    metrics_out = {
        "overall": summarize([{k: r[k] for k in PER_QUERY_KEYS} for r in per_query]),
        "queries_total": len(queries),
        "queries_skipped_no_relevant": len(skipped),
        "collection_size": collection.size,
        # E4-D: the time of the BASE alone. Naming it so is the whole point --
        # the stage costs seconds per query, not milliseconds, and a reader
        # taking this for its cost would be out by three orders of magnitude
        # (section 08; the stage is measured by scripts/measure_e4d_cost.py).
        ("query_time_ms_scene_only" if _has_query_time_detection(config)
         else "query_time_ms"): _query_time(timings, len(evaluated)),
        "groups": {name: summarize(items) for name, items in sorted(groups.items())},
    }

    # --- artifacts ----------------------------------------------------------
    dump_resolved(config, run_dir / "config.resolved.yaml")
    _write_metadata(run_dir, config, config_file, inputs, skipped)
    with open(run_dir / "relevance.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("desc_id;fragment_id\n")
        for query in queries:
            for fragment in sorted(relevance[query.desc_id]):
                f.write(f"{query.desc_id};{fragment}\n")
    with open(run_dir / "per_query.jsonl", "w", encoding="utf-8") as f:
        for record in per_query:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_signals(run_dir, pipeline, matrices, active, evaluated, collection)
    _write_matching(run_dir, signals)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = metrics_out["overall"]
    print(f"queries: {overall.get('query_count', 0)}  "
          + "  ".join(f"R@{k}: {overall[f'recall@{k}']:.3f}" for k in (1, 5, 10)
                      if f"recall@{k}" in overall))
    print(f"done -> {run_dir.relative_to(datasets.ROOT)}")
    return run_dir


def _has_query_time_detection(config) -> bool:
    detection = config.scoring.query_time_detection
    return detection is not None and detection.enabled


def _query_time_detection(config, scores, collection, evaluated, context, relevance):
    """Runs the E4-D stage when the configuration asks for it.

    Returns ``(orders, records)``: the ranking of every query, and the per-query
    record of the stage or ``None`` when it did not run. Without the stage the
    orders are the base ranking, so the caller has one code path either way.
    """
    detection = config.scoring.query_time_detection
    if detection is None or not detection.enabled:
        return [row.argsort()[::-1] for row in scores], None

    from src.data.ranges import load_ranges
    from src.retrieval import query_detection as qd
    from src.utils import gpu

    # one big model at a time: the base encoder has to leave the card before the
    # detector loads, and by here the ranking of the base is already computed
    gpu.free()
    ranges = load_ranges(config.dataset)
    row_of = {vid: i for i, vid in enumerate(collection.vids)}
    orders, records, _ = qd.rescore(
        scores, collection.vids, evaluated, context["rows"], row_of,
        context["timelines"],
        lambda episode: datasets.video_path(config.dataset,
                                            ranges[episode]["video_file"]),
        relevant=[relevance[query.desc_id] for query in evaluated],
        k=detection.candidates_top_n)
    return orders, records


def _k400_membership(config) -> dict[str, bool] | None:
    """Whether each clip's class is one SlowFast knows; ``None`` when not asked.

    The cut that says whether the motion signal had a name to reach at all.

    The split file is produced per part by the leak filter. On DEV it may
    legitimately be absent -- the clips are still downloading and the file is
    rewritten after every batch -- so a missing one warns and drops the groups.
    On TEST it may not: those rows are a table of the thesis, and a run that
    quietly finished without them would be indistinguishable from one that has
    them until somebody opened metrics.json.
    """
    if "kinetics_vocab" not in config.evaluation.report_by:
        return None
    if config.dataset != "vatex":
        print("report_by: kinetics_vocab applies to VATEX only - groups skipped")
        return None
    from src.utils import vatex

    split = config.split or "test"
    try:
        return vatex.k400_membership(split)
    except FileNotFoundError as error:
        if split == "test":
            raise
        print(f"kinetics_vocab groups skipped: {error}")
        return None


def _write_matching(run_dir: Path, signals: list) -> None:
    """Every phrase a closed-vocabulary signal judged, accepted or not.

    Its own file rather than a field of per_query.jsonl: the record is per
    PHRASE, not per query, and PER_QUERY_KEYS stays as the thesis lists it. This
    is what the "what was rejected" table of the threshold measurement reads, so
    the rejected phrases have to be here too, marked.
    """
    records = [record for signal in signals
               for record in getattr(signal, "matching_log", {}).values()]
    if not records:
        return
    records.sort(key=lambda r: (r["desc_id"], r["signal"], r["phrase"]))
    with open(run_dir / "matching.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _query_time(timings: list[float], expected: int) -> dict:
    """Query handling time in milliseconds: median, mean and 95th percentile.

    This is the cost of the query phase alone -- the indexes are already built,
    so what it measures is encoding the query and scoring the collection. It is
    the number the thesis reports as the overhead of a base representation whose
    final match depends on the fragment.
    """
    if not timings or len(timings) != expected:
        return {}
    import numpy as np

    values = np.asarray(timings, dtype=np.float64)
    return {"queries": int(values.size),
            "median": round(float(np.median(values)), 3),
            "mean": round(float(values.mean()), 3),
            "p95": round(float(np.percentile(values, 95)), 3)}


def _write_signals(run_dir: Path, pipeline, matrices, active, queries, collection) -> None:
    """Saves the standardized signals of the run, ready to be re-weighed.

    The sensitivity analysis walks a grid of weights over the very same values,
    so keeping them turns each grid point into a weighted sum instead of another
    pass over the recordings. Stored as float16: the values are z-scores of
    order one and the file is read to re-rank, never to reproduce a metric to
    the last digit.
    """
    import numpy as np

    np.savez_compressed(
        run_dir / "signals.npz",
        names=np.array([s.name for s in pipeline.signals]),
        desc_ids=np.array([q.desc_id for q in queries], dtype=np.int64),
        fragments=np.array(collection.vids),
        active=active,
        values=np.stack(matrices).astype(np.float16),
    )


def _new_run_dir(config: ExperimentConfig) -> Path:
    """``<experiment>_<dataset>_<split>_YYYYMMDD_HHMMSS``.

    The variant comes first, so a directory listing groups the runs of one
    experiment together; the timestamp closes the name with second resolution, so
    a repeated run gets its own directory and the runs of one variant still sort
    chronologically. The numbered fallback stays for two runs started within the
    same second.
    """
    # same shape as the measurement files (src.utils.notebook.save_measurement):
    # one timestamp format across the repository, date and time separated
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (f"{config.experiment}_{datasets.dataset_dir(config.dataset)}"
            f"_{config.split}_{stamp}")
    for n in range(1000):
        candidate = datasets.runs_dir() / (base if n == 0 else f"{base}_{n:03d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("could not allocate a run directory")


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _write_metadata(run_dir: Path, config: ExperimentConfig, config_file: Path | str,
                    inputs: dict, skipped: list) -> None:
    import importlib.metadata as pkg

    versions = {}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = pkg.version(name)
        except pkg.PackageNotFoundError:
            versions[name] = None

    # an input may be one file or several: VATEX reads its ranges from the
    # query files, of which there is one per part
    files = [path for value in inputs.values()
             for path in (value if isinstance(value, list) else [value])]
    files.append(datasets.queries_jsonl(config.dataset, config.split or "test"))
    metadata = {
        "run_id": run_dir.name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "config_file": str(config_file),
        "experiment": config.experiment,
        "dataset": config.dataset,
        "split": config.split,
        "seed": config.seed,
        "packages": versions,
        "inputs": manifest.manifest(files),
        "queries_skipped_no_relevant": skipped,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
