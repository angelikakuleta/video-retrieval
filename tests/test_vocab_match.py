"""Matching a phrase to a class name: the threshold and what clears it."""

import numpy as np
import pytest

from src.retrieval import vocab_match
from src.retrieval.phrases import Phrase
from src.utils.config import Matching


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FakeEncoder:
    """Deterministic vectors, so a test decides what lies close to what."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.encoded = []

    def encode_texts(self, texts):
        self.encoded.extend(texts)
        return np.stack([self.vectors[text] for text in texts])


#: three class names, mutually orthogonal
VOCABULARY = np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])


def phrase(*forms):
    return Phrase(tuple(forms), forms[-1], " ".join(forms))


# ------------------------------------------------------------- the threshold
def test_a_confidence_equal_to_the_threshold_is_accepted():
    """The inequality is not strict: the phrase REACHES the threshold, not exceeds."""
    encoder = FakeEncoder({"mug": unit(3, 1, 0)})
    item = phrase("mug")

    exactly = vocab_match.score([item], VOCABULARY, encoder,
                                measure="top1")[0].confidence
    accepted = vocab_match.match([item], VOCABULARY, encoder,
                                 threshold=exactly, measure="top1")
    assert [found.class_id for found in accepted] == [0]
    assert accepted[0].confidence == exactly


def test_a_confidence_below_the_threshold_is_dropped():
    encoder = FakeEncoder({"mug": unit(3, 1, 0)})
    item = phrase("mug")
    exactly = vocab_match.score([item], VOCABULARY, encoder,
                                measure="top1")[0].confidence

    assert vocab_match.match([item], VOCABULARY, encoder,
                             threshold=exactly + 1e-6, measure="top1") == []


def test_nothing_matches_when_the_vocabulary_holds_no_such_word():
    """The point of the threshold: some name is always nearest, few are right."""
    encoder = FakeEncoder({"stapler": unit(1, 1, 1)})
    found = vocab_match.match([phrase("stapler")], VOCABULARY, encoder,
                              threshold=0.9, measure="top1")
    assert found == []


# ---------------------------------------------------------- the two measures
def test_top1_is_the_absolute_similarity():
    encoder = FakeEncoder({"mug": unit(3, 1, 0)})
    found = vocab_match.score([phrase("mug")], VOCABULARY, encoder, measure="top1")[0]
    assert found.class_id == 0
    assert found.confidence == pytest.approx(3 / np.sqrt(10), abs=1e-6)


def test_z_standardizes_over_the_names_of_the_vocabulary():
    encoder = FakeEncoder({"mug": unit(3, 1, 0)})
    similarity = unit(3, 1, 0) @ VOCABULARY.T
    expected = (similarity.max() - similarity.mean()) / similarity.std()

    found = vocab_match.score([phrase("mug")], VOCABULARY, encoder, measure="z")[0]
    assert found.class_id == 0
    assert found.confidence == pytest.approx(expected, abs=1e-6)


def test_z_claims_nothing_when_every_name_is_equally_close():
    encoder = FakeEncoder({"thing": unit(1, 1, 1)})
    found = vocab_match.score([phrase("thing")], VOCABULARY, encoder, measure="z")[0]
    assert found.confidence == 0.0


def test_an_unknown_measure_is_rejected():
    encoder = FakeEncoder({"mug": unit(1, 0, 0)})
    with pytest.raises(ValueError, match="unknown confidence measure"):
        vocab_match.score([phrase("mug")], VOCABULARY, encoder, measure="cosine")


# --------------------------------------------------------------------- forms
def test_the_best_form_of_a_phrase_decides_it():
    """"coffee mug" and "mug" are one question, and the better spelling answers it."""
    encoder = FakeEncoder({"coffee mug": unit(1, 1, 0), "mug": unit(1, 0, 0)})
    found = vocab_match.score([phrase("coffee mug", "mug")], VOCABULARY, encoder,
                              measure="top1")[0]
    assert found.class_id == 0
    assert found.confidence == pytest.approx(1.0, abs=1e-6)


def test_a_form_is_encoded_once_per_encoder():
    encoder = FakeEncoder({"walk": unit(1, 0, 0), "walking": unit(0, 1, 0)})
    items = [phrase("walk", "walking"), phrase("walk", "walking")]

    vocab_match.score(items, VOCABULARY, encoder, measure="top1")
    vocab_match.score(items, VOCABULARY, encoder, measure="top1")
    assert sorted(encoder.encoded) == ["walk", "walking"]


def test_every_phrase_gets_a_match_before_the_threshold_is_applied():
    """`score` feeds the log of what was rejected; `match` feeds the signals."""
    encoder = FakeEncoder({"mug": unit(1, 0, 0), "stapler": unit(1, 1, 1)})
    items = [phrase("mug"), phrase("stapler")]

    assert len(vocab_match.score(items, VOCABULARY, encoder, measure="top1")) == 2
    assert len(vocab_match.match(items, VOCABULARY, encoder,
                                 threshold=0.9, measure="top1")) == 1


def test_no_phrases_is_an_empty_result():
    encoder = FakeEncoder({})
    assert vocab_match.match([], VOCABULARY, encoder,
                             threshold=0.5, measure="top1") == []


def test_vectors_from_another_encoder_are_refused():
    encoder = FakeEncoder({"mug": unit(1, 0, 0, 0)})
    with pytest.raises(ValueError, match="different encoders"):
        vocab_match.score([phrase("mug")], VOCABULARY, encoder, measure="top1")


def test_an_empty_vocabulary_is_refused():
    encoder = FakeEncoder({"mug": unit(1, 0, 0)})
    with pytest.raises(ValueError, match="vocabulary"):
        vocab_match.score([phrase("mug")], np.zeros((0, 3), dtype=np.float32),
                          encoder, measure="top1")


# ---------------------------------------------- the per-vocabulary threshold
def test_threshold_for_falls_back_to_the_measured_one():
    matching = Matching(measure="top1", threshold=0.25)
    assert vocab_match.threshold_for("actions_kinetics400", matching) == 0.25


def test_threshold_for_uses_the_raised_one_where_the_transfer_check_set_it():
    matching = Matching(measure="top1", threshold=0.25,
                        threshold_by_vocab={"objects_yoloe_promptfree": 0.4})
    assert vocab_match.threshold_for("objects_yoloe_promptfree", matching) == 0.4
    assert vocab_match.threshold_for("objects_yolo11", matching) == 0.25
