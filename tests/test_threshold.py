"""The threshold measurement: how tau is read off, and what may not drift.

Everything here is synthetic. The measurement's own numbers depend on clips that
are still downloading; what a test can pin is the RULE that turns confidences
into a threshold, and the promise that the frozen control lists stay frozen.
"""

import math

import pytest

from src.measurement import text_bridge as tb
from src.utils import experiments as exp


def scored(pairs, measure="top1"):
    """Descriptions with a match: ``(confidence, hit)`` each."""
    return [{"clip": f"c{i}", "label": "x", "n_phrases": 1,
             measure: {"confidence": c, "hit": hit, "class": "x" if hit else "y",
                       "others": []}}
            for i, (c, hit) in enumerate(pairs)]


# ----------------------------------------------------------------------- tau
def test_tau_is_the_lowest_confidence_the_precision_still_survives():
    """The LARGEST prefix at 0.9, not the largest at 1.0.

    Nine hits and two misses: ten descriptions are 9/10 = 0.9 and still qualify,
    eleven are 9/11 and do not. So the threshold is the tenth confidence and the
    first miss is inside -- 0.9 is what was asked for, not perfection.
    """
    rows = scored([(0.90, True), (0.80, True), (0.70, True), (0.60, True),
                   (0.50, True), (0.45, True), (0.40, True), (0.35, True),
                   (0.30, True), (0.25, False), (0.20, False)])
    result = tb.tau_from(rows, "top1")

    assert result["tau"] == 0.25
    assert math.isclose(result["precision_at_tau"], 0.9)
    assert result["n_above_tau"] == 10
    assert math.isclose(result["coverage"], 10 / 11)


def test_one_miss_inside_the_prefix_is_tolerated_at_nine_tenths():
    rows = scored([(0.9, True), (0.8, True), (0.7, True), (0.6, True), (0.5, True),
                   (0.4, True), (0.3, True), (0.2, True), (0.15, True), (0.1, False)])
    result = tb.tau_from(rows, "top1")
    assert result["tau"] == 0.1                  # 9 of 10 is exactly 0.9
    assert math.isclose(result["precision_at_tau"], 0.9)


def test_descriptions_of_equal_confidence_are_both_in_or_both_out():
    """The pipeline accepts every match at or above a VALUE, not a rank."""
    rows = scored([(0.8, True), (0.5, True), (0.5, False), (0.5, True)])
    result = tb.tau_from(rows, "top1")
    assert result["tau"] == 0.8                  # taking 0.5 would take all three
    assert result["n_above_tau"] == 1


def test_no_threshold_when_nothing_reaches_the_precision():
    rows = scored([(0.9, False), (0.8, False), (0.7, True)])
    result = tb.tau_from(rows, "top1")
    assert result["tau"] is None
    assert "precision" in result["reason"]


def test_a_description_without_an_action_phrase_still_counts_in_the_denominator():
    """Coverage is over the material, not over the descriptions that matched."""
    rows = scored([(0.9, True)]) + [{"clip": "c9", "label": "x", "n_phrases": 0,
                                     "top1": None}]
    result = tb.tau_from(rows, "top1")
    assert result["n_descriptions"] == 2 and result["n_matched"] == 1
    assert math.isclose(result["coverage"], 0.5)


# ------------------------------------------------------------- which measure
def both(top1_pairs, z_pairs):
    rows = []
    for i, ((c1, h1), (c2, h2)) in enumerate(zip(top1_pairs, z_pairs)):
        rows.append({"clip": f"c{i}", "label": "x", "n_phrases": 1,
                     "top1": {"confidence": c1, "hit": h1, "class": "x", "others": []},
                     "z": {"confidence": c2, "hit": h2, "class": "x", "others": []}})
    return rows


def test_an_interval_covering_zero_hands_it_to_top1():
    """The rule was fixed before the numbers: undecided means the absolute measure.

    `z` is standardized over the names of a vocabulary, so the same value means
    something else for eight emotion names than for four hundred actions; `top1`
    is a similarity and carries across.
    """
    rows = both([(0.9, True)] * 10, [(0.9, True)] * 10)
    taus = {m: tb.tau_from(rows, m) for m in ("top1", "z")}
    choice = tb.choose_measure(rows, taus)
    assert choice["chosen"] == "top1"
    assert choice["interval"]["low"] <= 0.0 <= choice["interval"]["high"]


def test_the_wider_reach_wins_when_the_interval_excludes_zero():
    # z matches every description, top1 only the first half
    n = 40
    rows = both([(0.9, True)] * (n // 2) + [(0.0, False)] * (n // 2),
                [(0.9, True)] * n)
    taus = {m: tb.tau_from(rows, m) for m in ("top1", "z")}
    choice = tb.choose_measure(rows, taus)
    assert choice["chosen"] == "z"
    assert choice["interval"]["high"] < 0.0          # top1 - z is negative


def test_a_measure_without_a_threshold_cannot_win():
    rows = both([(0.9, True)] * 10, [(0.9, False)] * 10)
    taus = {m: tb.tau_from(rows, m) for m in ("top1", "z")}
    assert taus["z"]["tau"] is None
    assert tb.choose_measure(rows, taus)["chosen"] == "top1"


# --------------------------------------- the diagnostics that decide nothing
def test_the_per_phrase_ceiling_explains_why_the_unit_is_a_description():
    """A clip has one label, so only one phrase of a description can be right."""
    rows = [{"clip": "c0", "label": "x", "n_phrases": 3,
             "top1": {"confidence": 0.9, "hit": True, "class": "x",
                      "others": [0.5, 0.4]}}]
    found = tb.diagnostics(rows, "top1", 0.9)
    assert found["n_phrases"] == 3
    # one description, three phrases: at most one of the three can be right, so
    # per-phrase precision cannot pass 1/3 whatever the threshold does
    assert math.isclose(found["per_phrase_ceiling"], 1 / 3)

    # and the bound does not move when the one correct phrase is missed
    rows[0]["top1"]["hit"] = False
    assert math.isclose(tb.diagnostics(rows, "top1", 0.9)["per_phrase_ceiling"], 1 / 3)


def test_the_second_diagnostic_counts_descriptions_with_another_phrase_above_tau():
    rows = [{"clip": "c0", "label": "x", "n_phrases": 2,
             "top1": {"confidence": 0.9, "hit": True, "class": "x", "others": [0.8]}},
            {"clip": "c1", "label": "x", "n_phrases": 2,
             "top1": {"confidence": 0.9, "hit": True, "class": "x", "others": [0.1]}}]
    assert tb.diagnostics(rows, "top1", 0.5)["second_phrase_passes"] == 0.5


# ---------------------------------------------------------- the frozen lists
def test_the_precision_level_is_the_one_written_down_beforehand():
    """0,9 was fixed before any measurement and is not subject to tuning."""
    assert tb.PRECISION == 0.9
    assert tb.SAMPLE_SEED == 1234


def test_the_threshold_rests_on_one_instrument_per_vocabulary():
    """The judged sample, and nothing beside it.

    An earlier design measured one threshold against Kinetics and carried it to
    the others, checking the carry against hand-written lists of fifteen phrases
    with a counterpart and fifteen without. The supervisor replaced that with a
    judged sample per vocabulary, and the lists then duplicated the measurement
    with a worse instrument -- hand-picked phrases sit either side of an empty
    gap in the confidence distribution, so the threshold read off them lands on
    the edge of the gap rather than where precision falls.
    """
    for gone in ("CONTROL_LISTS", "CONTROL_MIN_WITH", "CONTROL_MAX_WITHOUT",
                 "MANDATORY_BELOW", "HSEMOTION_WITH", "COCO_WITH", "YOLOE_WITH",
                 "transfer_control", "kinetics_observation", "check_control_lists"):
        assert not hasattr(tb, gone), f"{gone} came back"
    assert set(tb.SAMPLE_SOURCES) == set(tb.VOCABULARY_KIND)


# --------------------------------------------- the two names of the baseline
def test_the_baseline_has_one_name_in_e2_and_another_everywhere_else():
    """E2 compares representations, so its row says which; E3-E6 call it BAZA."""
    assert exp.display(exp.BASE) == "E2-A: OC (ViT-H/14)"
    assert exp.display(exp.BASE, base_as_name=False) == "BAZA"
    # every other label has one name
    assert exp.display("E4-B", base_as_name=False) == exp.display("E4-B")


# ----------------------------------------------------- the gate, in the code
def test_the_script_refuses_any_split_but_dev(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_threshold.py"
    spec = importlib.util.spec_from_file_location("measure_threshold_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr("sys.argv", ["measure_threshold.py", "--split", "test"])
    with pytest.raises(SystemExit, match="development part only"):
        module.main()


# --------------------------- the judged sample and the threshold read off it
def occurrences(rows):
    """``(phrase, class, confidence)`` triples as occurrence records."""
    return [{"source": "vatex", "desc_id": f"d{i}", "text": f"a sentence about {p}",
             "phrase": p, "matched_class": c, "confidence": v}
            for i, (p, c, v) in enumerate(rows)]


def population(rows, n_texts):
    """The population entry `threshold_from_judgments` reads, from those rows."""
    _, counts = tb.draw_sample(occurrences(rows))
    return {"vocab": {"n_texts": n_texts, **counts,
                      "occurrences": [[r["source"], r["desc_id"], r["phrase"],
                                       r["matched_class"], r["confidence"]]
                                      for r in occurrences(rows)]}}


def judged(rows, answers):
    return [{"vocabulary": "vocab", "desc_id": f"d{i}", "phrase": p,
             "matched_class": c, "correct": a}
            for i, ((p, c, _), a) in enumerate(zip(rows, answers))]


def test_the_sample_carries_no_truth_and_no_clip_label():
    """A judgment that can see the answer is not a judgment."""
    assert tb.SAMPLE_COLUMNS == ("vocabulary", "source", "desc_id", "text",
                                 "phrase", "matched_class", "confidence")
    assert not {"correct", "truth", "label", "class_k600", "hit"} & set(tb.SAMPLE_COLUMNS)
    assert tb.JUDGMENT_COLUMNS == ("vocabulary", "desc_id", "phrase",
                                   "matched_class", "correct")
    # two answers, not three: a pair the judge cannot settle is skipped, and a
    # skip writes nothing at all
    assert tb.ANSWERS == ("yes", "no")


def test_the_same_phrase_and_class_are_judged_once():
    """A repeated phrase always draws the same class, so it is one question."""
    rows = occurrences([("mug", "cup", 0.8), ("mug", "cup", 0.8),
                        ("desk", "dining table", 0.7), ("mug", "cup", 0.8)])
    sample, counts = tb.draw_sample(rows)
    assert counts["n_occurrences"] == 4 and counts["n_pairs"] == 2
    assert len(sample) == 2


def test_a_sample_smaller_than_the_quota_is_judged_whole():
    """HSEmotion has about forty pairs, and every one of them is judged."""
    rows = occurrences([(f"p{i}", "Anger", i / 100) for i in range(40)])
    sample, counts = tb.draw_sample(rows)
    assert len(sample) == 40
    assert counts["strata"] == {"all": 40} and counts["sampled"] == {"all": 40}
    # nothing to re-weight when everything is judged
    assert counts["edges"] == []


def test_the_draw_takes_forty_literal_pairs_and_spreads_sixty_over_the_range():
    """Confidence exactly 1.0 is a mass point, not an interval.

    It answers one question - is a literal match correct - and forty pairs settle
    that. The rest of the range is what locates the threshold, so it takes the
    larger half, spread evenly over ten intervals rather than piled at the top.
    """
    rows = occurrences([(f"p{i}", "c", 0.5 + i / 600) for i in range(300)]
                       + [(f"x{i}", "c", 1.0) for i in range(200)])
    sample, counts = tb.draw_sample(rows)
    assert counts["sampled"]["exact"] == 40
    bins = {name: n for name, n in counts["sampled"].items() if name != "exact"}
    assert sum(bins.values()) == 60
    assert set(bins) == {f"b{i:02d}" for i in range(1, 11)}
    assert set(bins.values()) == {6}                  # ten intervals, six each
    assert len(sample) == 100
    assert counts["strata"]["exact"] == 200


def test_a_literal_match_joins_the_mass_despite_float_noise():
    """Identical embeddings come back as 0.99999994 as often as as 1.0.

    A phrase's forms include its bare head, so any phrase whose head is a class
    name verbatim is literally that name - and with the 4585 YOLOE names that is
    most of them. Without a tolerance, float noise alone would decide which of
    those identical pairs joined the mass.
    """
    edges = tb.bin_edges([1.0, 0.99999994, 0.9999, 0.5])
    assert tb.stratum_of(1.0, edges) == "exact"
    assert tb.stratum_of(0.99999994, edges) == "exact"
    # a near miss is not a literal match; it belongs to the range that locates
    # the threshold, at the very top of it
    assert tb.stratum_of(0.9999, edges) == "b10"
    assert tb.stratum_of(0.5, edges) == "b01"
    # and the mass itself is out of the binned range
    assert edges[-1] < tb.EXACT


def test_an_interval_that_runs_short_gives_its_share_to_the_others():
    """The sample keeps its size wherever the distribution is thin."""
    assert tb.allocate([20] * 10, 60) == [6] * 10
    assert sum(tb.allocate([1, 0, 50, 50, 50, 50, 50, 50, 50, 50], 60)) == 60
    # nothing is drawn from an interval that holds nothing
    assert tb.allocate([0, 5, 5], 60) == [0, 5, 5]


def test_the_draw_is_the_same_every_time():
    rows = occurrences([(f"p{i}", "c", i / 400) for i in range(400)])
    first, _ = tb.draw_sample(rows)
    second, _ = tb.draw_sample(rows)
    assert [r["phrase"] for r in first] == [r["phrase"] for r in second]


def test_the_rows_are_not_in_confidence_order():
    """In confidence order the rhythm of the answers becomes a pattern."""
    rows = occurrences([(f"p{i}", "c", i / 400) for i in range(400)])
    sample, _ = tb.draw_sample(rows)
    values = [r["confidence"] for r in sample]
    assert values != sorted(values) and values != sorted(values, reverse=True)


def test_no_judgments_yet_is_a_result_and_not_a_crash():
    """The path has to be exercisable before a single pair has been judged."""
    out = tb.threshold_from_judgments([], "vocab", population={})
    assert out["threshold"] is None and "no judgments" in out["reason"]
    assert out["n_judged"] == 0 and out["coverage_phrases"] is None


def test_the_threshold_is_the_lowest_confidence_precision_survives():
    """Hand-computed. Ten pairs at 0.0 ... 0.9, wrong at 0.5 and at 0.1.

    From the top down: the prefix ending at 0.6 holds four pairs, all correct
    (4/4). Adding 0.5 makes it 4/5 = 0.8 and every longer prefix stays under the
    level - 5/6, 6/7, 7/8, 7/9, 8/10. So 0.6 is the only value whose prefix
    survives, and it is therefore also the lowest.

    Two wrong pairs and not one: with ten pairs a single miss leaves the whole
    set at exactly 0.9, which QUALIFIES, and the threshold would legitimately
    fall to the bottom of the range.
    """
    rows = [(f"p{i}", "c", i / 10) for i in range(10)]
    answers = ["yes"] * 10
    answers[5] = answers[1] = "no"         # the pairs at 0.5 and at 0.1
    out = tb.threshold_from_judgments(judged(rows, answers), "vocab",
                                      population=population(rows, n_texts=10))
    assert out["threshold"] == pytest.approx(0.6)
    assert out["precision_at_threshold"] == pytest.approx(1.0)
    assert out["n_above"] == 4


def test_a_skipped_pair_is_counted_against_the_sample_it_is_missing_from():
    """A skip writes no row, so the rows that came back cannot measure it.

    Ten pairs were drawn and nine came back. Dividing by the nine would report
    no unresolved pairs at all - the skip took the tenth out of that denominator
    too. Against the sample it is one in ten.
    """
    rows = [(f"p{i}", "c", i / 10) for i in range(10)]
    answers = ["yes"] * 10
    out = tb.threshold_from_judgments(judged(rows, answers)[:9], "vocab",
                                      population=population(rows, n_texts=10))
    assert out["n_judged"] == 9 and out["n_sample"] == 10
    assert out["n_unresolved"] == 1
    assert out["unresolved_share"] == pytest.approx(0.1)
    # more than one in twenty: the protocol says report it, not click through it
    assert out["unresolved_high"] is True
    # nine usable pairs, all correct, so the threshold falls to the lowest
    assert out["threshold"] == pytest.approx(0.0)
    assert out["precision_at_threshold"] == pytest.approx(1.0)
    assert out["n_above"] == 9


def test_the_unresolved_share_is_not_diluted_by_the_answers_that_came_back():
    """The number the supervisor asked to see: 100 drawn, 60 judged, 40 at 40%."""
    # 200 pairs across the range plus 60 literal ones, so the draw is a full
    # 40 + 60 and the sample really is a hundred
    rows = ([(f"p{i}", "c", 0.4 + i / 400) for i in range(200)]
            + [(f"x{i}", "c", 1.0) for i in range(60)])
    drawn = population(rows, n_texts=200)
    assert sum(drawn["vocab"]["sampled"].values()) == 100
    sample = {(r["phrase"], r["matched_class"]) for r in tb.draw_sample(
        occurrences(rows))[0]}
    answered = [{"vocabulary": "vocab", "desc_id": "d", "phrase": phrase,
                 "matched_class": klass, "correct": "yes"}
                for phrase, klass in sorted(sample)[:60]]
    out = tb.threshold_from_judgments(answered, "vocab", population=drawn)
    assert out["n_sample"] == 100
    assert out["n_yes"] + out["n_no"] == 60
    assert out["n_unresolved"] == 40
    assert out["unresolved_share"] == pytest.approx(0.4)
    assert out["unresolved_high"] is True


def test_an_answer_outside_the_two_is_not_silently_dropped():
    """`unsure` no longer exists; a file still carrying it is a file to look at."""
    rows = [(f"p{i}", "c", i / 10) for i in range(10)]
    answers = ["yes"] * 10
    answers[9] = "unsure"
    out = tb.threshold_from_judgments(judged(rows, answers), "vocab",
                                      population=population(rows, n_texts=10))
    assert out["n_unknown"] == 1
    assert out["n_yes"] + out["n_no"] == 9
    assert out["n_unresolved"] == 1


def test_precision_is_re_weighted_back_to_the_population():
    """Counting the judgments flat would describe the sample, not the vocabulary.

    The edges run 0.3 to 0.9 in ten steps of 0.06, so 0.9 and 0.88 fall in the
    top interval `b10` and 0.35 and 0.3 in the bottom one `b01`. `b10` holds 10
    pairs of which 2 were judged (weight 5); `b01` holds 90 of which 2 were
    judged (weight 45). Both `b10` pairs are correct and one of the two `b01`
    pairs is. Unweighted that is 3/4 = 0.75; weighted it is
    (5 + 5 + 45) / (5 + 5 + 45 + 45) = 55/100 = 0.55.
    """
    entry = {"n_texts": 4,
             "edges": [0.3 + i * 0.06 for i in range(11)],
             "strata": {"b10": 10, "b01": 90},
             "occurrences": [["vatex", "d0", "a", "c", 0.9],
                             ["vatex", "d1", "b", "c", 0.88],
                             ["vatex", "d2", "e", "c", 0.35],
                             ["vatex", "d3", "f", "c", 0.3]]}
    rows = [{"vocabulary": "vocab", "desc_id": f"d{i}", "phrase": phrase,
             "matched_class": "c", "correct": answer}
            for i, (phrase, answer) in enumerate(
                zip("abef", ["yes", "yes", "yes", "no"]))]
    out = tb.threshold_from_judgments(rows, "vocab", population={"vocab": entry},
                                      precision=0.5)
    assert out["strata"]["b10"]["weight"] == pytest.approx(5.0)
    assert out["strata"]["b01"]["weight"] == pytest.approx(45.0)
    assert out["threshold"] == pytest.approx(0.3)
    assert out["precision_at_threshold"] == pytest.approx(0.55)


def test_coverage_is_counted_on_occurrences_not_on_distinct_pairs():
    """The question is how many phrases and queries the signal speaks for.

    Five occurrences of three distinct pairs, over four texts out of ten asked.
    At 0.7 three occurrences pass, from two texts: 3/5 of the phrases and 2/10
    of the queries - not 2/3, which is the share of the pair TYPES.
    """
    entry = {"n_texts": 10, "edges": [], "strata": {"all": 3},
             "occurrences": [["vatex", "d0", "mug", "cup", 0.9],
                             ["vatex", "d0", "mug", "cup", 0.9],
                             ["vatex", "d1", "desk", "dining table", 0.8],
                             ["vatex", "d2", "lamp", "vase", 0.4],
                             ["vatex", "d3", "lamp", "vase", 0.4]]}
    assert tb.coverage_at(entry, 0.7) == {
        "coverage_phrases": pytest.approx(0.6),
        "coverage_queries": pytest.approx(0.2),
        "n_occurrences": 5, "n_texts": 10}


def test_a_thin_accepted_set_is_flagged_rather_than_believed():
    """Under 35 judged pairs above it, the interval around 0.9 says nothing."""
    rows = [(f"p{i}", "c", i / 10) for i in range(10)]
    out = tb.threshold_from_judgments(judged(rows, ["yes"] * 10), "vocab",
                                      population=population(rows, n_texts=10))
    assert out["n_above"] == 10 and out["enough_above"] is False
    assert tb.MIN_ABOVE == 35


def test_a_vocabulary_whose_precision_never_reaches_the_level_says_so():
    """Alternating from the top down, no prefix ever passes one half."""
    rows = [(f"p{i}", "c", i / 10) for i in range(10)]
    out = tb.threshold_from_judgments(judged(rows, ["yes", "no"] * 5), "vocab",
                                      population=population(rows, n_texts=10))
    assert out["threshold"] is None
    assert "peaks at 0.500" in out["reason"]


def test_a_judgment_of_a_pair_outside_the_sample_is_counted_apart():
    """The confidence comes from the population; a pair with none cannot join."""
    rows = [(f"p{i}", "c", i / 10) for i in range(10)]
    extra = judged(rows, ["yes"] * 10) + [
        {"vocabulary": "vocab", "desc_id": "d99", "phrase": "ghost",
         "matched_class": "c", "correct": "yes"}]
    out = tb.threshold_from_judgments(extra, "vocab",
                                      population=population(rows, n_texts=10))
    assert out["n_unknown"] == 1 and out["n_above"] == 10


def test_the_literal_subset_measures_the_ceiling_of_the_old_path():
    """Where the phrase IS the class name, the match is right by construction.

    Three literal matches, two agreeing with the clip label, plus one
    non-literal match that is ignored: 2/3 - the ceiling the supervisor's second
    condition rests on, and the reason no threshold at 0.9 was reachable there.
    """
    rows = [
        {"top1": {"confidence": 0.9, "class": "situp", "hit": True,
                  "forms": ["situp", "sit"], "others": []}},
        {"top1": {"confidence": 0.8, "class": "dancing ballet", "hit": True,
                  "forms": ["dancing ballet"], "others": []}},
        {"top1": {"confidence": 0.7, "class": "yoga", "hit": False,
                  "forms": ["yoga"], "others": []}},
        {"top1": {"confidence": 0.6, "class": "surfing water", "hit": True,
                  "forms": ["ride wave"], "others": []}},
    ]
    out = tb.literal_agreement(rows, "top1")
    assert out["n"] == 3
    assert out["agreement"] == pytest.approx(2 / 3, abs=1e-3)


def test_every_vocabulary_is_sampled_with_the_kind_it_answers():
    assert set(tb.SAMPLE_SOURCES) == set(tb.VOCABULARY_KIND)
    for name, (kind, sources) in tb.SAMPLE_SOURCES.items():
        assert kind == tb.VOCABULARY_KIND[name]
        assert "vatex" in sources
    # only the expression vocabulary reaches beyond VATEX: clip descriptions
    # barely mention facial expression, and that material will not grow
    assert tb.SAMPLE_SOURCES["expressions_hsemotion"][1] == ("vatex",) + tb.SERIES
    for name in ("objects_yolo11", "objects_yoloe_promptfree", tb.KINETICS):
        assert tb.SAMPLE_SOURCES[name][1] == ("vatex",)

# ---------------- the activation table, and the thresholds it is computed at
#: two sentences whose phrases are known: the first yields object and action
#: phrases, the second expression ones. Both outputs are what the rules actually
#: produce, checked by running them.
ACTIVATION_TEXTS = [("A man in a blue shirt puts a coffee mug on the desk", ()),
                    ("A sad woman laughs at her friend", ())]

#: the forms the stub encoder places on the first class; every other form lands
#: near it but below any threshold worth having
KNOWN = ("mug", "laugh")


class StubEncoder:
    """Encodes a form as one of two vectors, so a confidence is decided here.

    Two dimensions and two classes: a form of KNOWN sits exactly on class 0 and
    scores 1.0, anything else sits at 0.4 of the way there. No model is loaded
    and no similarity depends on what a real encoder happens to think.
    """

    def encode_texts(self, texts):
        import numpy as np

        return np.asarray([[1.0, 0.0] if any(k in t for k in KNOWN) else [0.4, 0.0]
                           for t in texts], dtype=np.float32)


def synthetic_vocabularies():
    """Every vocabulary the activation table walks, on one 2 x 2 basis."""
    import numpy as np

    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return {name: (["first", "second"], matrix) for name in tb.VOCABULARY_KIND}


def run_activation(monkeypatch, thresholds):
    monkeypatch.setattr(tb, "dev_texts", lambda dataset, material: ACTIVATION_TEXTS)
    return tb.activation(["tbbt"], [], synthetic_vocabularies(), StubEncoder(),
                         "top1", thresholds, log=lambda *_: None)


def test_a_threshold_per_vocabulary_fills_the_rows_the_table_is_made_of(monkeypatch):
    """The point of the whole path: numbers, not dashes.

    Every row of `tab:aktywacja-dev` whose signal scores against a closed
    vocabulary is computed at that vocabulary's own threshold. Given one, the row
    answers; the phrases that fell short land in `tab:frazy-odrzucone`.
    """
    shares, rejected = run_activation(monkeypatch,
                                      {name: 0.5 for name in tb.VOCABULARY_KIND})

    gated = [row for row, (_, vocab) in tb.ACTIVATION_ROWS.items() if vocab is not None]
    assert gated                      # the rows the defect emptied
    for row in gated:
        assert shares[row]["tbbt"] is not None, row
        assert 0.0 <= shares[row]["tbbt"] <= 1.0

    # the object phrase "coffee mug"/"mug" clears 0.5 in the first sentence and
    # nothing does in the second, so exactly half the queries switch the signal on
    assert shares["objects_coco"]["tbbt"] == pytest.approx(0.5)
    # "blue shirt" and "desk" reached no name: they are what the second table lists
    assert rejected["objects_yolo11"]
    assert {"shirt", "desk"} <= {head for head, _ in rejected["objects_yolo11"]}


def test_without_a_threshold_the_same_rows_stay_unanswered(monkeypatch):
    """The state the measurement was in, pinned so it cannot come back quietly.

    `measure` used to hand `activation` the old path's tau for every vocabulary,
    and that tau is None -- so every gated row came back as a dash and the
    rejected phrases were never counted. A dash is the right answer to a question
    nobody put; the defect was that nobody put it.
    """
    shares, rejected = run_activation(monkeypatch,
                                      {name: None for name in tb.VOCABULARY_KIND})

    for row, (_, vocab) in tb.ACTIVATION_ROWS.items():
        if vocab is not None:
            assert shares[row]["tbbt"] is None, row
    assert not any(rejected.values())


def test_the_two_open_vocabulary_rows_do_not_depend_on_a_threshold(monkeypatch):
    """E5 measures a coverage difference, so these rows answer either way.

    They are the two that were filled in while the rest were empty, and they have
    to stay identical: the query-time detector and the face regions ask no closed
    vocabulary and have no threshold to clear.
    """
    with_thresholds, _ = run_activation(monkeypatch,
                                        {name: 0.5 for name in tb.VOCABULARY_KIND})
    without, _ = run_activation(monkeypatch,
                                {name: None for name in tb.VOCABULARY_KIND})

    open_rows = [row for row, (_, vocab) in tb.ACTIVATION_ROWS.items() if vocab is None]
    assert set(open_rows) == {"query_time_detection", "face_regions"}
    for row in open_rows:
        assert with_thresholds[row] == without[row]
        assert with_thresholds[row]["tbbt"] is not None


# ---------------- and that `measure` reads them BEFORE it computes the table
def scored_both(pairs):
    """Descriptions carrying both measures, as `measure` needs them."""
    return [{"clip": f"c{i}", "label": "x", "n_phrases": 1,
             "top1": {"confidence": c, "hit": hit, "class": "x" if hit else "y",
                      "others": [], "forms": ("x",)},
             "z": {"confidence": c, "hit": hit, "class": "x" if hit else "y",
                   "others": [], "forms": ("x",)}}
            for i, (c, hit) in enumerate(pairs)]


def test_measure_computes_the_table_at_the_judged_thresholds(monkeypatch, tmp_path):
    """The ordering defect itself: what reaches `activation` is what was judged.

    Everything expensive is stubbed -- no encoder, no vocabulary cache, no
    scoring pass -- because the thing under test is which numbers arrive where,
    not what the numbers are.
    """
    import numpy as np

    from src.features import encoders, text_vocab

    wanted = {"objects_yolo11": 0.11, "objects_yoloe_promptfree": 0.22,
              "expressions_hsemotion": 0.33, tb.KINETICS: 0.44}

    monkeypatch.setattr(tb, "dev_material",
                        lambda part, encoder, **kw: [{"clip": "c0", "text": "t",
                                                      "label": "x"}])
    monkeypatch.setattr(text_vocab, "load",
                        lambda encoder, name: (["a", "b"],
                                               np.zeros((2, 2), dtype=np.float32)))
    monkeypatch.setattr(encoders, "build", lambda name: StubEncoder())
    monkeypatch.setattr(tb, "score_descriptions",
                        lambda *a, **kw: scored_both([(0.9, True), (0.8, False)]))
    monkeypatch.setattr(tb, "threshold_from_judgments",
                        lambda judgments, name, **kw: {"threshold": wanted[name],
                                                       "vocabulary": name,
                                                       "reason": None})

    seen = {}

    def spy(datasets, material, vocabularies, encoder_object, measure_name,
            thresholds, log=print):
        seen.update(thresholds)
        return {row: {"tbbt": 0.0} for row in tb.ACTIVATION_ROWS}, {}

    monkeypatch.setattr(tb, "activation", spy)

    data = tb.measure(part="dev", datasets=("tbbt",), judgments=tmp_path / "none.csv",
                      log=lambda *_: None)

    assert seen == wanted                     # not the old path's tau, and not None
    assert {name: entry["threshold"] for name, entry in data["judged"].items()} == wanted


def test_the_judged_block_is_part_of_the_payload_not_bolted_on_by_the_caller():
    """`measure` returns it, so the script has nothing left to add afterwards.

    Computing the block after the call is one step too late for the activation
    table, and the two then go out of step.
    """
    import inspect

    from scripts import measure_threshold

    source = inspect.getsource(measure_threshold.main)
    assert "judgments=JUDGMENTS" in source
    assert 'data["judged"] =' not in source
    assert "judgments" in inspect.signature(tb.measure).parameters
