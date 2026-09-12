"""Matching a query phrase to a name in a closed vocabulary (chapter 4).

The comparison used to run between the whole query sentence and the list of class
names, taking the nearest name. Some name always won, including when the
vocabulary held no word from the query at all: on the development set the gap
between the first and the second emotion name was 0.02 while the fixed offsets
between the names themselves were 0.05-0.08, so what decided was which name sits
closer to sentences in general. For Kinetics, *walk* landed on *writing* 72 times
out of 72.

So a phrase is matched instead of a sentence, and the match has to CLEAR A
THRESHOLD to count. Below it the signal simply has nothing to say about the
query, which is what the activation gate in :mod:`src.retrieval.signals` reads.

No anchors, no name templates, no lexicons, no synonyms: the class names are
compared exactly as they lie in ``data/cache/vocab/<encoder>/``.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.retrieval.phrases import Phrase
from src.utils.config import Matching

#: the two candidate confidence measures; which one is used is decided by the
#: threshold measurement, not here
MEASURES = ("top1", "z")


@dataclass(frozen=True)
class Match:
    """The best class a phrase reached, and how strongly."""

    phrase: Phrase
    class_id: int
    confidence: float


#: text -> vector, per encoder instance. Forms repeat heavily across queries
#: ("man", "door", "walk"), and the encoder is the expensive part; the entry
#: dies with the encoder that produced it, so vectors from two different spaces
#: can never be mixed.
_EMBEDDINGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def embed(texts: Sequence[str], encoder) -> np.ndarray:
    """Embeddings of ``texts``, each encoded at most once per encoder.

    The same encoder as the query, which is what makes the similarity comparable
    with the one the scene signal reports.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    cache = _EMBEDDINGS.setdefault(encoder, {})
    missing = [text for text in dict.fromkeys(texts) if text not in cache]
    if missing:
        vectors = np.asarray(encoder.encode_texts(missing), dtype=np.float32)
        for text, vector in zip(missing, vectors):
            cache[text] = vector
    return np.stack([cache[text] for text in texts])


def threshold_for(vocab_name: str, matching: Matching) -> float:
    """The threshold this vocabulary is judged by.

    One threshold is measured, on Kinetics-400; the transfer check may RAISE it
    for a vocabulary in which an unrelated phrase lands too easily -- a catalogue
    of some 4500 names has a near neighbour for almost anything, eight emotion
    names have one for very little.
    """
    return matching.threshold_by_vocab.get(vocab_name, matching.threshold)


def score(phrases: Sequence[Phrase], vocabulary: np.ndarray, encoder, *,
          measure: str) -> list[Match]:
    """The best class of every phrase, threshold not applied.

    Each phrase is embedded in ALL its forms and takes the best of them, so
    "coffee mug" and "mug", or "pour drink" and "pouring drink", are one
    question and not two.
    """
    if measure not in MEASURES:
        raise ValueError(f"unknown confidence measure: {measure!r}; "
                         f"expected one of {MEASURES}")
    names = np.asarray(vocabulary, dtype=np.float32)
    if names.ndim != 2 or not names.size:
        raise ValueError(f"expected a (n_classes, dim) vocabulary, got {names.shape}")
    if not phrases:
        return []

    forms = [form for phrase in phrases for form in phrase.forms]
    vectors = embed(forms, encoder)
    if vectors.shape[1] != names.shape[1]:
        raise ValueError(
            f"the phrases are encoded in {vectors.shape[1]} dimensions and the "
            f"vocabulary in {names.shape[1]} - they come from different encoders")

    matches, start = [], 0
    for phrase in phrases:
        stop = start + len(phrase.forms)
        similarity = vectors[start:stop] @ names.T          # (n_forms, n_classes)
        start = stop

        best = int(np.argmax(similarity))
        form, class_id = divmod(best, similarity.shape[1])
        matches.append(Match(phrase, class_id,
                             _confidence(similarity[form], class_id, measure)))
    return matches


def _confidence(similarity: np.ndarray, class_id: int, measure: str) -> float:
    """How strongly the winning form picked its class, on the chosen scale.

    ``top1`` is the absolute similarity, comparable between vocabularies of any
    size. ``z`` says how far the winner stands out from the other names, which
    means something different for eight names than for four hundred; the choice
    between them is a measurement, not a preference.
    """
    top1 = float(similarity[class_id])
    if measure == "top1":
        return top1
    deviation = float(similarity.std())
    if deviation == 0.0:
        # every name equally close: nothing stands out, so nothing is claimed
        return 0.0
    return (top1 - float(similarity.mean())) / deviation


def match(phrases: Sequence[Phrase], vocabulary: np.ndarray, encoder, *,
          threshold: float, measure: str) -> list[Match]:
    """The phrases whose best class REACHES the threshold.

    The inequality is not strict: a confidence equal to the threshold counts as a
    match. Everything else is dropped -- the vocabulary has no name for it, and
    answering anyway is what the threshold exists to stop.
    """
    return [found for found in score(phrases, vocabulary, encoder, measure=measure)
            if found.confidence >= threshold]
