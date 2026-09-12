"""The judgment pool and the weight sensitivity: the rules, on synthetic data.

Neither can be exercised on real material yet -- both read runs of the test
split, and there are none. What a test can pin is what was written down before
the measurement: the share, the seed, the set of configurations, the shape of the
grid, and the two things about the sensitivity analysis that are easy to get
quietly wrong -- the undefined point at w = 1 and "full without k" at non-uniform
weights.
"""

from pathlib import Path

import numpy as np
import pytest

from src.evaluation import pool, sensitivity
from src.utils import experiments as exp


# ------------------------------------------------------------------ the pool
def test_the_share_and_the_seed_are_the_ones_written_down():
    assert pool.POOL_SHARE == 0.10
    assert pool.POOL_SEED == 1234


def test_the_pool_configurations_are_the_main_ones_and_only_those():
    """Eighteen labels, fixed before the measurement.

    Written out here rather than derived, because the point of the list is that
    it cannot drift: if `experiments.py` grows a signal, this test fails and a
    person decides whether the pool changes, instead of it changing by itself.
    """
    assert set(pool.POOL_LABELS) == {
        "E2-A", "E2-C", "E3-B", "E3-C", "E4-B", "E4-C", "E5-B", "E5-C", "E6-B",
        "full",
        "full_no_caption", "full_no_objects", "full_no_motion",
        "full_no_face_regions", "full_no_identity",
        "full_swap_caption", "full_swap_objects", "full_swap_face_regions",
    }
    assert len(pool.POOL_LABELS) == 18


def test_the_pool_leaves_out_the_controls_and_the_query_time_variant():
    """X-CLIP, the segmentation variants and E4-D answer other questions."""
    for label in ("E2-Ap", "E2-B", "E2-Bp", "E1-A", "E1-B", "E1-C", "E4-D", "E5-Bp"):
        assert label not in pool.POOL_LABELS


def test_the_judge_never_sees_which_configuration_returned_a_fragment():
    assert pool.POOL_COLUMNS == ("desc_id", "desc", "fragment", "video_file",
                                 "start", "end", "relevant")
    assert not {"experiment", "label", "config", "rank", "score"} & set(pool.POOL_COLUMNS)
    assert pool.JUDGMENT_COLUMNS == ("desc_id", "fragment", "relevant")
    # every column the recomputation reads is already in the file the judge got,
    # so judging is filling one column in place and never transcribing rows
    assert set(pool.JUDGMENT_COLUMNS) <= set(pool.POOL_COLUMNS)


def test_the_judged_file_is_the_pool_without_the_suffix():
    """Step three of the control is a rename, so the names have to line up."""
    built = pool.pool_file("tbbt", "test").name
    answers = pool.judgments_file("tbbt").name
    assert built == answers.replace(".csv", f"{pool.POOL_SUFFIX}.csv")
    assert pool.pool_file("tbbt").parent != pool.judgments_file("tbbt").parent


def test_vatex_has_no_pool():
    """Ten-second clips annotated by their own description: no gap to measure."""
    assert "vatex" not in pool.build_pool.__defaults__[3]
    assert set(pool.build_pool.__defaults__[3]) == set(exp.SERIES)


class _Query:
    def __init__(self, desc_id):
        self.desc_id = desc_id
        self.desc = f"query {desc_id}"


def test_the_sample_is_a_tenth_and_the_same_tenth_every_time():
    queries = [_Query(i) for i in range(600)]
    first = pool.sample_queries(queries)
    assert len(first) == 60
    assert first == pool.sample_queries(queries)
    # drawn from the identifiers, so the order of the file cannot move it
    assert first == pool.sample_queries(list(reversed(queries)))


def test_a_query_set_smaller_than_ten_still_yields_one():
    assert len(pool.sample_queries([_Query(1), _Query(2)])) == 1


def test_a_fragment_already_counted_correct_is_not_judged_again():
    """`augment_relevance` adds; it is the additions the pool exists to find."""
    relevance = {7: {"s01e01_003"}}
    judged = [{"desc_id": "7", "fragment": "s01e01_009", "relevant": "yes"},
              {"desc_id": "7", "fragment": "s01e01_011", "relevant": "no"}]
    out = pool.augment_relevance(judged, relevance)
    assert out[7] == {"s01e01_003", "s01e01_009"}
    assert relevance == {7: {"s01e01_003"}}      # the original is left alone


def test_a_rejected_fragment_never_removes_one_from_the_annotation():
    """`no` says the pool found nothing new, not that the annotation was wrong."""
    relevance = {7: {"s01e01_003"}}
    judged = [{"desc_id": "7", "fragment": "s01e01_003", "relevant": "no"}]
    assert pool.augment_relevance(judged, relevance)[7] == {"s01e01_003"}


def test_the_recomputation_is_restricted_to_the_queries_the_pool_covered():
    judged = [{"desc_id": "7", "fragment": "a", "relevant": "yes"},
              {"desc_id": "9", "fragment": "b", "relevant": "no"}]
    assert pool.pool_ids(judged) == {7, 9}


def test_no_judgments_yet_is_a_result_and_not_a_crash():
    assert pool.load_judgments("nothing/here.csv") == []
    assert pool.augment_relevance([], {3: {"x"}}) == {3: {"x"}}


def pooled_run(run_dir, desc_ids, strategy="shots_histogram"):
    """A run stub with a configuration and a ranking, as build_pool reads one."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "relevance.csv").write_text("desc_id;fragment_id" + chr(10),
                                           encoding="utf-8-sig")
    return {"dir": run_dir,
            "config": {"segmentation": {"strategy": strategy,
                                        "histogram_threshold": 0.5}},
            "per_query": [{"desc_id": desc_id,
                           "top": [{"fragment": "s01e01_001"}]}
                          for desc_id in desc_ids]}


def test_the_pool_reads_the_strategy_of_several_configurations_at_once(
        tmp_path, monkeypatch):
    """Two configurations, one strategy -- and the pool is written.

    The strategies used to be collected as a set of segmentation BLOCKS, which
    is a set of dicts and therefore a TypeError; with one run in the mapping the
    comprehension never ran and nothing noticed. Two is the smallest number that
    is a pool at all.
    """
    from src.data import datasets
    from src.data.queries import load_queries

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    folder = tmp_path / "data" / "annotations" / "tbbt"
    folder.mkdir(parents=True)
    lines = [f'{{"desc_id": {i}, "desc": "q{i}", "vid_name": "tbbt_s01e01", '
             f'"ts": [0, 1], "source": "tbbt"}}' for i in range(20)]
    (folder / "tbbt_queries_test.jsonl").write_text(chr(10).join(lines),
                                                    encoding="utf-8")
    cache = tmp_path / "data" / "cache" / "segmentation" / "shots_histogram"
    cache.mkdir(parents=True)
    (cache / "tbbt_segments.csv").write_text(
        "episode;segment_id;split;video_file;start;end;duration;file_duration" + chr(10)
        + "s01e01;1;test;s01e01.mp4;0.0;10.0;10.0;10.0" + chr(10), encoding="utf-8-sig")

    ids = [q.desc_id for q in load_queries("tbbt", "test")]
    runs = {"E2-A": {"tbbt": pooled_run(tmp_path / "a", ids)},
            "full": {"tbbt": pooled_run(tmp_path / "b", ids)}}
    report = pool.build_pool("test", datasets_=("tbbt",), runs=runs, log=lambda _: None)
    assert report["tbbt"]["labels"] == ["E2-A", "full"]
    assert report["tbbt"]["rows"] > 0
    assert Path(report["tbbt"]["file"]).exists()


def judged_run(tmp_path, records, relevance):
    """A run stub with its own relevance.csv, as recompute reads one."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    with open(run_dir / "relevance.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("desc_id;fragment_id\n")
        for desc_id, fragments in relevance.items():
            for fragment in sorted(fragments):
                f.write(f"{desc_id};{fragment}\n")
    return {"dir": run_dir, "per_query": records}


def record(desc_id, top, hit):
    """One per_query record: ten stored positions and the recall the run scored."""
    return {"desc_id": desc_id, "recall@10": float(hit),
            "top": [{"fragment": name} for name in top]}


def test_a_judged_fragment_inside_the_top_turns_a_miss_into_a_hit(tmp_path):
    run = judged_run(tmp_path, [record(7, ["a", "b", "c"], hit=0)], {7: {"z"}})
    judged = [{"desc_id": "7", "fragment": "b", "relevant": "yes"}]
    out = pool.recompute(run, judged)
    assert (out["before"], out["after"], out["gained"], out["added"]) == (0.0, 1.0, 1, 1)


def test_a_rejected_fragment_changes_nothing(tmp_path):
    run = judged_run(tmp_path, [record(7, ["a", "b"], hit=0)], {7: {"z"}})
    judged = [{"desc_id": "7", "fragment": "b", "relevant": "no"}]
    out = pool.recompute(run, judged)
    assert (out["before"], out["after"], out["gained"]) == (0.0, 0.0, 0)


def test_only_the_queries_of_the_pool_enter_the_average(tmp_path):
    """A query outside the pool keeps the annotation it always had.

    Averaging it in would mix a completed measure with an incomplete one and
    report neither, so the denominator is the pool and nothing else.
    """
    run = judged_run(tmp_path,
                     [record(7, ["a"], hit=0), record(8, ["q"], hit=1)],
                     {7: {"z"}, 8: {"q"}})
    out = pool.recompute(run, [{"desc_id": "7", "fragment": "a", "relevant": "yes"}])
    assert out["n"] == 1                      # query 8 was never judged
    assert (out["before"], out["after"]) == (0.0, 1.0)


def test_a_run_that_stored_a_hit_its_top_does_not_show_is_refused(tmp_path):
    """The judgments belong to a different run, and a silent answer would hide it."""
    run = judged_run(tmp_path, [record(7, ["a", "b"], hit=1)], {7: {"elsewhere"}})
    with pytest.raises(ValueError, match="different run"):
        pool.recompute(run, [{"desc_id": "7", "fragment": "a", "relevant": "no"}])


def test_recall_deeper_than_the_stored_ranking_is_refused(tmp_path):
    """A run writes ten positions, so Recall@50 cannot be read back from it."""
    run = judged_run(tmp_path, [record(7, ["a"], hit=0)], {7: {"z"}})
    with pytest.raises(ValueError, match="stores 10 positions"):
        pool.recompute(run, [{"desc_id": "7", "fragment": "a", "relevant": "yes"}], k=50)


def test_the_stored_depth_is_the_one_the_runner_writes():
    """pool.DEPTH mirrors run.TOP_SAVED; the two drifting apart is the hazard."""
    from src.runners.run import TOP_SAVED

    assert pool.DEPTH == TOP_SAVED


def test_no_judgments_yet_gives_an_empty_recomputation(tmp_path):
    run = judged_run(tmp_path, [record(7, ["a"], hit=0)], {7: {"z"}})
    out = pool.recompute(run, [])
    assert out == {"n": 0, "before": None, "after": None, "gained": 0, "added": 0}


# ----------------------------------------------------------- the weight grid
def test_the_grid_is_the_eleven_points_plus_the_uniform_weight():
    """Twelve points for a series, eleven for VATEX - 72 and 55 reweightings."""
    series = sensitivity.grid_for(6)
    assert len(series) == 12                  # 1/6 is not on the grid
    assert round(1 / 6, 6) in series
    assert 6 * len(series) == 72

    vatex = sensitivity.grid_for(5)
    assert len(vatex) == 11                   # 1/5 = 0,2 already is
    assert 5 * len(vatex) == 55


def test_the_rest_of_the_weight_is_shared_equally():
    base = sensitivity.weight_vector(4, 1, 0.4)
    assert base[1] == pytest.approx(0.4)
    assert list(base[[0, 2, 3]]) == pytest.approx([0.2, 0.2, 0.2])
    assert base.sum() == pytest.approx(1.0)


# --------------------------------- a synthetic collection with a known order
def collection():
    """Three signals over four queries and five fragments.

    `scene_embedding` is active everywhere and points at fragment 0; `strong`
    points at the same one and is GATED - silent for the last query; `weak`
    points somewhere else. So the order of the contributions is known before
    anything is computed: strong above weak.
    """
    names = ["scene_embedding", "strong", "weak"]
    active = np.array([[True] * 4, [True, True, True, False], [True] * 4])
    values = np.zeros((3, 4, 5))
    for query in range(4):
        values[0, query] = [0.6, 0.1, 0.0, 0.0, 0.0]      # base: fragment 0
        values[1, query] = [1.0, 0.0, 0.0, 0.0, 0.0]      # strong: fragment 0
        values[2, query] = [0.0, 0.0, 0.0, 0.0, 0.9]      # weak: fragment 4
    fragments = [f"f{i}" for i in range(5)]
    return {"names": names, "desc_ids": [1, 2, 3, 4], "fragments": fragments,
            "active": active, "values": values}


RELEVANCE = {i: {"f0"} for i in (1, 2, 3, 4)}


def test_the_undefined_point_is_marked_not_counted_as_zero():
    """At w = 1 a gated signal leaves the query it is silent for with no weight.

    `query_weights` then hands that query a row of zeros over the whole
    collection and its ranking is an artefact of the fragment order. The point
    has to leave the curve and the range, not enter them as a low number.
    """
    swept = sensitivity.sweep(collection(), RELEVANCE, k=1)

    assert swept["strong"]["undefined_points"] == [1.0]
    assert 1.0 not in [w for w, _ in swept["strong"]["curve"]]
    # the signals that are active everywhere keep their w = 1
    for name in ("scene_embedding", "weak"):
        assert swept[name]["undefined_points"] == []
        assert 1.0 in [w for w, _ in swept[name]["curve"]]


def test_the_range_ignores_the_undefined_point():
    swept = sensitivity.sweep(collection(), RELEVANCE, k=1)
    reached = [value for _, value in swept["strong"]["curve"]]
    assert swept["strong"]["range"] == pytest.approx(max(reached) - min(reached))


def test_the_edge_control_reproduces_removing_the_signal():
    """w = 0 of a row is the run without that signal, reached from the other side.

    Computed here from the same matrices, which is what makes it a control of
    the STORE rather than of the arithmetic: a disagreement against the real
    `full_no_*` run means signals.npz is not what the run scored.
    """
    signals = collection()
    swept = sensitivity.sweep(signals, RELEVANCE, k=1)

    without = {}
    for row, name in enumerate(signals["names"]):
        if name == exp.BASE_SIGNAL:
            continue
        mask = signals["active"].copy()
        mask[row] = False
        scores = sensitivity.scores_at(signals["values"], mask, None)
        without[name] = sensitivity.recall_of(scores, signals["fragments"],
                                              [RELEVANCE[i] for i in signals["desc_ids"]],
                                              k=1)
    base_only = sensitivity.scores_at(
        signals["values"],
        np.array([signals["active"][0], *[np.zeros(4, bool)] * 2]), None)
    without[exp.BASE_SIGNAL] = sensitivity.recall_of(
        base_only, signals["fragments"], [RELEVANCE[i] for i in signals["desc_ids"]], k=1)

    checks = sensitivity.edge_checks(swept, without)
    for name, entry in checks.items():
        assert entry["close"] is True, (name, entry)
        assert abs(entry["difference"]) <= sensitivity.EDGE_TOLERANCE
    assert checks["scene_embedding"]["edge"] == 1.0     # the base row is checked at w=1
    assert checks["strong"]["edge"] == 0.0


def test_a_missing_reference_run_is_reported_rather_than_passed():
    swept = sensitivity.sweep(collection(), RELEVANCE, k=1)
    checks = sensitivity.edge_checks(swept, {"strong": None})
    assert checks["strong"]["close"] is None
    assert checks["strong"]["difference"] is None


def test_full_without_k_keeps_the_weights_of_the_grid_point():
    """Removing a signal must change one thing, not two.

    At w = 0,8 on the base row the others hold 0,1 each. Dropping `weak` has to
    leave `strong` at its proportional share of what is left (0,8 and 0,1
    renormalized), not send both back to uniform. The check is direct: the mask
    with `weak` zeroed has to give the same weights as `query_weights` of the
    two-signal problem at those base values.
    """
    from src.retrieval.fusion import query_weights

    signals = collection()
    base = sensitivity.weight_vector(3, 0, 0.8)
    mask = signals["active"].copy()
    mask[2] = False

    weights = query_weights(mask, base)
    assert weights[2].tolist() == [0.0] * 4                 # weak is gone
    assert weights[0][0] == pytest.approx(0.8 / 0.9)        # base keeps its share
    assert weights[1][0] == pytest.approx(0.1 / 0.9)        # and so does strong
    # uniform weights would have given 0,5 and 0,5 - two changes, not one
    assert weights[0][0] != pytest.approx(0.5)


def test_the_order_of_the_contributions_is_read_tie_tolerantly():
    reference = {"strong": 0.4, "weak": 0.1}
    assert sensitivity.order_is_kept(reference, {"strong": 0.3, "weak": 0.2})
    assert sensitivity.order_is_kept(reference, {"strong": 0.2, "weak": 0.2})  # a tie
    assert not sensitivity.order_is_kept(reference, {"strong": 0.1, "weak": 0.4})
    # one signal's position, not the whole order
    assert sensitivity.order_is_kept(reference, {"strong": 0.1, "weak": 0.4},
                                     signal="strong") is False


def test_the_known_order_survives_the_grid():
    """`strong` contributes more than `weak` by construction, at every point."""
    signals = collection()
    uniform = sensitivity.marginal_contributions(
        signals["values"], signals["active"], None, signals["fragments"],
        [RELEVANCE[i] for i in signals["desc_ids"]], signals["names"], k=1)
    assert uniform["strong"] >= uniform["weak"]

    swept = sensitivity.sweep(signals, RELEVANCE, k=1)
    assert swept["strong"]["order_kept"] is True
    assert swept["scene_embedding"]["order_kept"] is True
