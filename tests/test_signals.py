"""Component signals: what a fragment is worth, and when a signal speaks at all.

Everything here runs on synthetic buffers. The point is the scoring rule of
chapter 4 -- the mean over the class names the query matched of the best evidence
any item of the fragment gives -- and the three ways a fragment can have nothing
to say, which are not the same thing.
"""

import numpy as np
import pytest

from src.retrieval.phrases import Phrase
from src.retrieval.signals import (
    aggregate_max,
    CaptionSignal,
    ExpressionSignal,
    FaceRegionSignal,
    IdentitySignal,
    MotionSignal,
    ObjectSignal,
    resolve_characters,
)
from src.utils.config import Matching
from src.utils.queries import Query


class StubEncoder:
    """Returns the vector a text was registered with."""

    def __init__(self, table):
        self.table = table

    def encode_texts(self, texts):
        return np.stack([self.table[t] for t in texts]).astype(np.float32)


def query(text="a query", identities=()):
    return Query(desc_id=1, desc=text, vid_name="tbbt_s01e01", ts=(0.0, 5.0),
                 source="tbbt", identities=list(identities))


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def phrase(*forms):
    return Phrase(tuple(forms), forms[-1], " ".join(forms))


def given(**kinds):
    """The phrase lists one query arrives with."""
    return [{"object": [], "expression": [], "action": [], **kinds}]


#: two class names; class 1 is the one the test queries match
VOCABULARY = np.stack([unit(0, 1), unit(1, 0)])
NAMES = ["other", "mug"]
STRICT = Matching(measure="top1", threshold=0.9)


# ------------------------------------------------------------ best item wins

def test_a_fragment_takes_the_best_of_its_items_not_their_mean():
    values = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    best = aggregate_max(values, np.array([0, 0, 1]), size=2)
    assert best[0, 0] == 1.0        # the best of the two, not 0.5
    assert best[0, 1] == 0.5


def test_a_fragment_with_no_items_reports_no_value():
    best = aggregate_max(np.array([[1.0]], dtype=np.float32), np.array([0]), size=3)
    assert best[0, 0] == 1.0
    assert np.isnan(best[0, 1]) and np.isnan(best[0, 2])


def test_caption_signal_scores_the_best_matching_description():
    encoder = StubEncoder({"a dog": unit(1, 0)})
    captions = np.stack([unit(0, 1), unit(1, 0), unit(0, 1)])   # item 1 matches
    signal = CaptionSignal(encoder, captions, np.array([0, 0, 1]), size=3)
    values = signal.raw_values([query("a dog")])
    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(0.0)
    assert np.isnan(values[0, 2])                                # no caption at all


def test_an_item_signal_with_nothing_cached_reports_no_value_anywhere():
    signal = CaptionSignal(StubEncoder({}), np.zeros((0, 2), np.float32),
                           np.zeros((0,), np.int32), size=2)
    assert np.all(np.isnan(signal.raw_values([query("anything")])))


# ------------------------------------------------------------------- objects

def object_signal(class_of_item, score_of_item, fragment_of, size=3,
                  matching=STRICT, table=None):
    encoder = StubEncoder(table or {"mug": unit(1, 0), "cup": unit(1, 0),
                                    "chair": unit(0, 1)})
    return ObjectSignal(encoder, VOCABULARY, NAMES, "objects_yolo11", matching, size,
                        class_of_item=np.asarray(class_of_item, np.int32),
                        score_of_item=np.asarray(score_of_item, np.float32),
                        fragment_of=np.asarray(fragment_of, np.int32))


def test_no_detection_ranks_above_a_detection_of_the_wrong_class():
    """The middle case is the whole point of the rule.

    Fragment 0 shows the class; fragment 1 was looked at and does not show it, so
    it scores a raw zero; fragment 2 offered no evidence either way, so it scores
    NaN -- which standardization turns into the neutral zero, above the raw one.
    """
    signal = object_signal(class_of_item=[1, 0, 0], score_of_item=[0.8, 0.5, 0.4],
                           fragment_of=[0, 0, 1])
    values = signal.raw_values([query()], given(object=[phrase("mug")]))
    assert values[0, 0] == pytest.approx(0.8)     # best confidence of that class
    assert values[0, 1] == 0.0                    # looked, did not find it
    assert np.isnan(values[0, 2])                 # nothing was detected at all

    from src.retrieval.fusion import standardize
    standardized = standardize(values)
    assert standardized[0, 2] > standardized[0, 1]


def test_a_fragment_is_worth_the_mean_over_the_matched_classes():
    """Mean over the positions, maximum over the items -- eq. `dopasowanie`."""
    encoder = StubEncoder({"mug": unit(1, 0), "chair": unit(0, 1)})
    signal = ObjectSignal(encoder, VOCABULARY, NAMES, "objects_yolo11", STRICT, 2,
                          class_of_item=np.array([1, 1, 0], np.int32),
                          score_of_item=np.array([0.4, 0.9, 0.6], np.float32),
                          fragment_of=np.array([0, 0, 0], np.int32))
    values = signal.raw_values([query()],
                               given(object=[phrase("mug"), phrase("chair")]))
    # class 1 reaches 0.9 (the better of two detections), class 0 reaches 0.6
    assert values[0, 0] == pytest.approx((0.9 + 0.6) / 2)


def test_a_phrase_below_the_threshold_leaves_the_signal_silent():
    signal = object_signal(class_of_item=[1], score_of_item=[0.8], fragment_of=[0])
    far = {"object": [phrase("stapler")], "expression": [], "action": []}
    signal.encoder.table["stapler"] = unit(1, 1)          # halfway between names
    assert not signal.active(query(), far)
    assert np.all(np.isnan(signal.raw_values([query()], [far])))


def test_a_signal_without_a_threshold_stays_silent_and_says_so():
    messages = []
    signal = ObjectSignal(StubEncoder({"mug": unit(1, 0)}), VOCABULARY, NAMES,
                          "objects_yolo11", None, 2,
                          class_of_item=np.array([1], np.int32),
                          score_of_item=np.array([0.8], np.float32),
                          fragment_of=np.array([0], np.int32),
                          log=messages.append)
    assert not signal.active(query(), given(object=[phrase("mug")])[0])
    assert any("matching block" in message for message in messages)


def test_the_empty_detection_case_splits_into_two_one_dimensional_columns():
    """`_gather` returns (0, 1) when it found nothing; the split must still work."""
    from src.runners.components import _gather

    payload, fragments = _gather([], [], {}, {}, lambda episode: None)
    detections = payload.reshape(len(fragments), 2)
    assert detections[:, 0].shape == (0,) and detections[:, 1].shape == (0,)

    signal = object_signal(class_of_item=detections[:, 0], score_of_item=detections[:, 1],
                           fragment_of=fragments, size=2)
    assert np.all(np.isnan(signal.raw_values([query()], given(object=[phrase("mug")]))))


def test_every_match_reaches_the_log_including_the_rejected_one():
    signal = object_signal(class_of_item=[1], score_of_item=[0.8], fragment_of=[0])
    signal.encoder.table["stapler"] = unit(1, 1)
    signal.raw_values([query()], given(object=[phrase("mug"), phrase("stapler")]))

    logged = {record["phrase"]: record for record in signal.matching_log.values()}
    assert logged["mug"]["accepted"] is True
    assert logged["mug"]["class_name"] == "mug"
    assert logged["stapler"]["accepted"] is False


def test_the_log_is_keyed_so_a_warm_up_pass_writes_no_duplicates():
    signal = object_signal(class_of_item=[1], score_of_item=[0.8], fragment_of=[0])
    for _ in range(3):
        signal.raw_values([query()], given(object=[phrase("mug")]))
    assert len(signal.matching_log) == 1


# -------------------------------------------------------------------- motion

def motion_signal(probability, matching=STRICT):
    encoder = StubEncoder({"run": unit(1, 0), "running": unit(1, 0)})
    return MotionSignal(encoder, VOCABULARY, NAMES, "actions_kinetics400", matching,
                        len(probability), probability=np.asarray(probability, np.float32))


def test_motion_reads_the_probability_of_the_matched_class():
    signal = motion_signal([[0.1, 0.9], [0.7, 0.3]])
    values = signal.raw_values([query()], given(action=[phrase("run", "running")]))
    assert values[0, 0] == pytest.approx(0.9)
    assert values[0, 1] == pytest.approx(0.3)


def test_an_unclassified_fragment_is_nan_and_not_a_sentinel_class():
    """The old buffer filled it with class 0 at probability 1.0.

    Under the new scoring that fragment would score a perfect 1.0 for every query
    whose phrase happened to match class 0 -- the whole collection at the top.
    """
    signal = motion_signal([[0.1, 0.9], [np.nan, np.nan]])
    values = signal.raw_values([query()], given(action=[phrase("run", "running")]))
    assert values[0, 0] == pytest.approx(0.9)
    assert np.isnan(values[0, 1])


def test_the_distribution_has_to_cover_the_whole_vocabulary():
    with pytest.raises(ValueError, match="distribution"):
        MotionSignal(StubEncoder({}), VOCABULARY, NAMES, "actions_kinetics400",
                     STRICT, 2, probability=np.zeros((2, 1), np.float32))


# ------------------------------------------------------ the two face signals

def expression_signal(distribution, fragment_of, size, matching=STRICT):
    encoder = StubEncoder({"smile": unit(1, 0), "nervous": unit(1, 1) / np.sqrt(2)})
    return ExpressionSignal(encoder, VOCABULARY, NAMES, "expressions_hsemotion",
                            matching, size,
                            distribution=np.asarray(distribution, np.float32),
                            fragment_of=np.asarray(fragment_of, np.int32))


def test_expression_takes_the_best_face_of_the_fragment():
    signal = expression_signal([[0.2, 0.8], [0.1, 0.9]], [0, 0], size=2)
    values = signal.raw_values([query()], given(expression=[phrase("smile")]))
    assert values[0, 0] == pytest.approx(0.9)      # the second face, not their mean
    assert np.isnan(values[0, 1])                  # no face in that fragment


def region_signal(vectors, fragment_of, size, query_mode="phrases"):
    encoder = StubEncoder({"smile": unit(1, 0), "nervous": unit(0, 1),
                           "a nervous man": unit(0, 1), "very nervous": unit(1, 0)})
    return FaceRegionSignal(encoder, np.asarray(vectors, np.float32),
                            np.asarray(fragment_of, np.int32), size, query=query_mode)


def test_a_region_phrase_takes_the_best_of_its_forms_and_of_the_faces():
    crops = np.stack([unit(1, 0), unit(0, 1)])          # two faces of fragment 0
    signal = region_signal(crops, [0, 0], size=2)
    # "very nervous" matches the first crop, "nervous" the second: the best of
    # both forms against the best of both faces
    values = signal.raw_values([query()],
                               given(expression=[phrase("very nervous", "nervous")]))
    assert values[0, 0] == pytest.approx(1.0)
    assert np.isnan(values[0, 1])


def test_the_region_signal_is_silent_when_the_query_has_no_expression_phrase():
    signal = region_signal(np.stack([unit(1, 0)]), [0], size=2)
    assert not signal.active(query(), given()[0])
    assert np.all(np.isnan(signal.raw_values([query()], given())))


def test_the_sentence_variant_answers_always():
    """E5-Bp: the earlier behaviour, kept so the thesis has its control row."""
    signal = region_signal(np.stack([unit(1, 0)]), [0], size=2,
                           query_mode="sentence")
    signal.encoder.table["a nervous man"] = unit(1, 0)
    assert signal.active(query("a nervous man"), given()[0])
    assert not signal.needs_phrases
    values = signal.raw_values([query("a nervous man")], None)
    assert values[0, 0] == pytest.approx(1.0)


def test_an_unknown_query_mode_is_refused():
    with pytest.raises(ValueError, match="face_regions.query"):
        FaceRegionSignal(StubEncoder({}), np.zeros((0, 2), np.float32),
                         np.zeros((0,), np.int32), 1, query="whole_sentence")


def test_the_two_face_mechanisms_share_the_phrases_but_not_the_gate():
    """E5 measures the vocabulary AND the coverage together, by design.

    `nervous` reaches no name among the eight emotions, so HSEmotion stays quiet;
    it reaches the face crops perfectly well, so the regions answer.
    """
    phrases = given(expression=[phrase("nervous")])[0]
    closed = expression_signal([[0.5, 0.5]], [0], size=1)
    open_ended = region_signal(np.stack([unit(0, 1)]), [0], size=1)

    assert not closed.active(query(), phrases)      # below the threshold
    assert open_ended.active(query(), phrases)      # no threshold to be below
    assert ExpressionSignal.kind == "expression"


def test_the_pipeline_extracts_the_phrases_once_for_every_signal(monkeypatch):
    """One parse per query, handed to each signal -- that is what makes the two
    face mechanisms see the same list rather than two equal ones."""
    from src.retrieval import pipeline as pipeline_module

    seen = []

    def fake_phrases_for(record, **_):
        seen.append(record.desc)
        return {"object": [], "expression": [], "action": []}

    class Collecting:
        name = "objects"
        needs_phrases = True

        def __init__(self):
            self.got = []

        def raw_values(self, queries, phrases=None):
            self.got.append(phrases)
            return np.zeros((len(queries), 2), dtype=np.float32)

        def active(self, query, phrases=None):
            return True

    monkeypatch.setattr("src.retrieval.phrases.phrases_for", fake_phrases_for)

    class Tiny:
        size = 2
        vids = ["a", "b"]

    signal = Collecting()
    pipeline = pipeline_module.Pipeline(Tiny(), [signal])
    pipeline.signal_matrices([query("a man smiles")])

    assert seen == ["a man smiles"]                  # parsed once
    assert signal.got[0] is not None                 # and handed over


def test_a_pipeline_of_signals_that_read_no_phrases_parses_nothing(monkeypatch):
    from src.retrieval import pipeline as pipeline_module

    def explode(record, **_):
        raise AssertionError("no signal here reads phrases")

    monkeypatch.setattr("src.retrieval.phrases.phrases_for", explode)

    class Tiny:
        size = 2
        vids = ["a", "b"]

    signal = CaptionSignal(StubEncoder({"a dog": unit(1, 0)}),
                           np.stack([unit(1, 0)]), np.array([0]), size=2)
    pipeline = pipeline_module.Pipeline(Tiny(), [signal])
    pipeline.signal_matrices([query("a dog")])


# ------------------------------------------------------------------ identity

def test_a_name_matches_a_profile_by_its_first_word():
    profiles = {"Michael Scott": None, "Pam Beesly": None}
    assert resolve_characters(["Michael"], profiles) == ["Michael Scott"]
    assert resolve_characters(["pam"], profiles) == ["Pam Beesly"]
    assert resolve_characters(["Toby"], profiles) == []


def identity_signal():
    #                     fragment 0: two faces      fragment 1: one face
    vectors = np.stack([unit(1, 0), unit(0, 1), unit(1, 0)])
    profiles = {"Michael Scott": np.stack([unit(1, 0)]),
                "Pam Beesly": np.stack([unit(0, 1)])}
    return IdentitySignal(profiles, vectors, np.array([0, 0, 1]), size=3)


def test_the_identity_signal_is_inactive_for_a_query_naming_nobody_profiled():
    signal = identity_signal()
    assert not signal.active(query("a man walks in"))
    assert not signal.active(query("Toby smiles", ["Toby"]))
    assert signal.active(query("Michael smiles", ["Michael"]))


def test_one_name_scores_the_best_matching_face():
    values = identity_signal().raw_values([query("Michael smiles", ["Michael"])])
    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(1.0)
    assert np.isnan(values[0, 2])


def test_two_names_are_combined_conjunctively():
    values = identity_signal().raw_values(
        [query("Michael and Pam talk", ["Michael", "Pam"])])
    # fragment 0 has both faces, fragment 1 only one -- the weaker decides
    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(0.0)


def test_an_inactive_query_leaves_the_row_empty():
    values = identity_signal().raw_values([query("a man walks in")])
    assert np.all(np.isnan(values[0]))


def test_signal_names_are_the_names_of_the_configuration_components():
    """Manual weights are written against components, so the two must agree."""
    from src.retrieval import signals as sig
    from src.utils.config import Components

    classes = [sig.SceneSignal, sig.XClipSceneSignal, sig.CaptionSignal,
               sig.ObjectSignal, sig.FaceRegionSignal, sig.ExpressionSignal,
               sig.IdentitySignal, sig.MotionSignal]
    assert {c.name for c in classes} <= set(Components.model_fields)


# ------------------------------------------------ the encoder input contract

def test_both_encoders_accept_arrays_and_images_alike():
    """Grid frames arrive as images, face crops as arrays -- both must work.

    The check is on the contract, not on the weights: a stub with the same two
    methods stands in for the models, which are too heavy for a unit test.
    """
    import inspect

    from src.features.openclip import ClipEncoder, _as_image
    from src.features.xclip import XClipEncoder

    for cls in (ClipEncoder, XClipEncoder):
        assert {"encode_texts", "encode_images"} <= set(dir(cls))
        assert "images" in inspect.signature(cls.encode_images).parameters

    from PIL import Image
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    assert isinstance(_as_image(array), Image.Image)
    assert _as_image(Image.fromarray(array)) is not None
