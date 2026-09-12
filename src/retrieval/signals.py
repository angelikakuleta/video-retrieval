"""Component signals: each scores a query against every fragment.

``raw_values`` returns a (n_queries, n_fragments) matrix of raw values, with NaN
where the component had no data for a fragment. ``active`` says whether the
component has anything to say about a query at all; if not, it is left out of
that query's weighting. Both conventions are handled in
:mod:`src.retrieval.fusion`.

Three shapes appear, and the difference is all of what chapter 4 prescribes:
comparison with one vector per fragment (scene), the best match among the
fragment's items (captions, face regions), and -- for the components with a
CLOSED vocabulary -- the mean over the class names the query actually matched of
the best evidence any item of the fragment offers for that name.

The closed-vocabulary signals take their phrases ready-made, extracted once per
query in :meth:`src.retrieval.pipeline.Pipeline.signal_matrices`. That is what
makes it structural, rather than a matter of calling the same function twice,
that the two mechanisms of the face component see the same list of expression
phrases -- which is exactly what the E5 contrast has to compare.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.indexing.faiss_index import Collection
from src.retrieval import vocab_match
from src.retrieval.phrases import Phrase
from src.utils.config import Matching
from src.utils.queries import Query
from src.utils.vocabulary import resolve_characters

#: the three phrase lists of one query, as :func:`phrases.phrases_for` returns them
QueryPhrases = dict[str, list[Phrase]]


class Signal(Protocol):
    """Shared interface of a component signal."""

    name: str
    #: whether the pipeline has to extract query phrases for this signal
    needs_phrases: bool

    def raw_values(self, queries: list[Query],
                   phrases: list[QueryPhrases] | None = None) -> np.ndarray:
        """Raw values: a matrix (n_queries, n_fragments); ``NaN`` = no data."""
        ...

    def active(self, query: Query, phrases: QueryPhrases | None = None) -> bool:
        """Whether the signal takes part in the ranking of this query."""
        ...


def aggregate_max(values: np.ndarray, fragment_of: np.ndarray, size: int) -> np.ndarray:
    """Best per-fragment value out of its items; NaN where a fragment has none.

    Captions, face crops and object labels belong to a fragment as a set, and
    chapter 4 takes the highest similarity any of them reaches, so something
    visible in one frame is not averaged away. The scoring scope is the whole
    collection, so this is an exact maximum over a full product, not a top-k
    approximation.
    """
    best = np.full((len(values), size), -np.inf, dtype=np.float32)
    if values.shape[1]:
        np.maximum.at(best.T, np.asarray(fragment_of, dtype=np.int32), values.T)
    return np.where(np.isfinite(best), best, np.nan)


def _texts(queries: list[Query]) -> list[str]:
    return [q.desc for q in queries]


def _phrases_of(phrases: list[QueryPhrases] | None, index: int) -> QueryPhrases:
    return (phrases[index] if phrases is not None and index < len(phrases) else {})


# --------------------------------------------------- the base representation
class SceneSignal:
    """Similarity of the query to the fragment's global representation.

    Vectors come from the FAISS collection, queries from the same encoder; both are
    normalized, so the dot product is the cosine. Every fragment has one, so this
    signal never reports missing data.

    Its name is the one the configuration gives the component, which is true of
    every signal here and is what lets manual weights be written against components.
    """

    name = "scene_embedding"
    needs_phrases = False

    def __init__(self, collection: Collection, encoder) -> None:
        self.encoder = encoder
        self._fragments = collection.matrix()      # (n_fragments, dim)

    def raw_values(self, queries, phrases=None) -> np.ndarray:
        embeddings = self.encoder.encode_texts(_texts(queries))
        return embeddings @ self._fragments.T      # (n_queries, n_fragments)

    def active(self, query, phrases=None) -> bool:
        return True


class XClipSceneSignal:
    """Scene signal of the video-language representation, with prompting.

    The query embedding is shifted by a term the prompt module derives from the
    fragment being scored, and only then compared, which is why this variant's query
    phase grows with the collection.
    """

    name = "scene_embedding"
    needs_phrases = False

    def __init__(self, collection: Collection, encoder, prompt_features: np.ndarray) -> None:
        self.encoder = encoder
        self._fragments = collection.matrix()               # (n_fragments, dim)
        self._prompts = np.asarray(prompt_features, dtype=np.float32)
        if len(self._prompts) != len(self._fragments):
            raise ValueError("prompt features do not cover the collection: "
                             f"{len(self._prompts)} vs {len(self._fragments)} fragments")

    def raw_values(self, queries, phrases=None) -> np.ndarray:
        embeddings = self.encoder.encode_texts_projected(_texts(queries))
        return self.encoder.prompted_similarity(embeddings, self._fragments, self._prompts)

    def active(self, query, phrases=None) -> bool:
        return True


# ---------------------------------- best match among the items of a fragment
class _BestItemSignal:
    """Highest similarity between the query and any item of the fragment."""

    name = "item"
    needs_phrases = False

    def __init__(self, encoder, vectors: np.ndarray, fragment_of: np.ndarray,
                 size: int) -> None:
        self.encoder = encoder
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.fragment_of = np.asarray(fragment_of, dtype=np.int32)
        self.size = size
        if len(self.vectors) != len(self.fragment_of):
            raise ValueError(f"{self.name}: {len(self.vectors)} vectors for "
                             f"{len(self.fragment_of)} items")

    def raw_values(self, queries, phrases=None) -> np.ndarray:
        if not len(self.vectors):
            return np.full((len(queries), self.size), np.nan, dtype=np.float32)
        embeddings = self.encoder.encode_texts(_texts(queries))
        return aggregate_max(embeddings @ self.vectors.T, self.fragment_of, self.size)

    def active(self, query, phrases=None) -> bool:
        return True


class CaptionSignal(_BestItemSignal):
    """Best match among the descriptions of the fragment's frames."""

    name = "caption"


class FaceRegionSignal(_BestItemSignal):
    """The open-vocabulary face mechanism: the query against the face crops.

    Asked with PHRASES by default, one entry per expression phrase, so that the
    E5 contrast against HSEmotion measures the width of the vocabulary and not
    the difference between a phrase and a sentence. The value of a phrase for a
    fragment is the highest similarity any of its forms reaches against any face
    of that fragment; the fragment's value is the mean over the phrases.

    There is no threshold here, because there are no class names to reach one:
    the signal is active whenever the query has an expression phrase at all. That
    is a WIDER gate than HSEmotion's, deliberately -- *uncomfortable* and
    *nervous* reach the crops and reach no name among the eight emotions -- so
    the E5 contrast covers vocabulary and coverage together.

    ``query="sentence"`` restores the earlier behaviour, comparing the whole
    query with the crops and answering always. It exists for the E5-Bp control
    row and for nothing else.
    """

    name = "face_regions"

    def __init__(self, encoder, vectors: np.ndarray, fragment_of: np.ndarray,
                 size: int, query: str = "phrases") -> None:
        super().__init__(encoder, vectors, fragment_of, size)
        if query not in ("phrases", "sentence"):
            raise ValueError(f"unknown face_regions.query: {query!r}")
        self.query = query
        self.needs_phrases = query == "phrases"

    def _phrase_values(self, phrase: Phrase) -> np.ndarray:
        """One row over the fragments: max over the forms and over the faces."""
        forms = vocab_match.embed(phrase.forms, self.encoder)
        per_face = (forms @ self.vectors.T).max(axis=0)[None, :]
        return aggregate_max(per_face, self.fragment_of, self.size)[0]

    def raw_values(self, queries, phrases=None) -> np.ndarray:
        if self.query == "sentence":
            return super().raw_values(queries)
        out = np.full((len(queries), self.size), np.nan, dtype=np.float32)
        if not len(self.vectors):
            return out
        for i in range(len(queries)):
            found = _phrases_of(phrases, i).get("expression", [])
            if not found:
                continue                       # the signal is inactive for this query
            out[i] = np.stack([self._phrase_values(p) for p in found]).mean(axis=0)
        return out

    def active(self, query, phrases=None) -> bool:
        if self.query == "sentence":
            return True
        return bool((phrases or {}).get("expression"))


# --------------------------------------------------- the closed vocabularies
class _ClosedVocabularySignal:
    """What a fragment offers for the class names the query matched.

    The pattern of chapter 4, shared by objects, expressions and motion: the
    query's phrases are matched against the vocabulary, everything below the
    threshold is dropped, and the fragment's value is the MEAN over the surviving
    positions of the BEST evidence any item of the fragment gives for that class.

    A fragment holding no item at all reports NaN -- the component had nothing to
    say about it -- which standardization turns into the neutral zero. That is not
    the same as a fragment whose items name other classes; see
    :class:`ObjectSignal`.
    """

    name = "closed_vocabulary"
    #: which of the three phrase lists this signal reads
    kind = "object"
    needs_phrases = True

    def __init__(self, encoder, vocabulary: np.ndarray, names: list[str],
                 vocab_name: str, matching: Matching | None, size: int,
                 log=None) -> None:
        self.encoder = encoder
        self.vocabulary = np.asarray(vocabulary, dtype=np.float32)
        self.names = list(names)
        self.vocab_name = vocab_name
        self.matching = matching
        self.size = size
        #: {(desc_id, signal, forms): record} for matching.jsonl. Keyed rather
        #: than appended, so the pipeline's warm-up pass writes no duplicates.
        self.matching_log: dict[tuple, dict] = {}
        if matching is None and log is not None:
            log(f"{self.name}: no matching block - the signal stays inactive "
                "everywhere; freeze the threshold first")

    def _accepted(self, query: Query, phrases: QueryPhrases) -> list:
        """The matches that reach the threshold; every match reaches the log."""
        if self.matching is None:
            return []
        found = vocab_match.score(phrases.get(self.kind, []), self.vocabulary,
                                  self.encoder, measure=self.matching.measure)
        threshold = vocab_match.threshold_for(self.vocab_name, self.matching)
        accepted = []
        for match in found:
            passed = match.confidence >= threshold
            self.matching_log[(query.desc_id, self.name, match.phrase.forms)] = {
                "desc_id": query.desc_id,
                "signal": self.name,
                "phrase": match.phrase.head,
                "forms": list(match.phrase.forms),
                "class_id": int(match.class_id),
                "class_name": self.names[match.class_id],
                "confidence": round(float(match.confidence), 4),
                "accepted": bool(passed),
            }
            if passed:
                accepted.append(match)
        return accepted

    def _class_values(self, class_id: int) -> np.ndarray:
        """Best evidence for one class, per fragment; NaN where there is no item."""
        raise NotImplementedError

    def raw_values(self, queries, phrases=None) -> np.ndarray:
        out = np.full((len(queries), self.size), np.nan, dtype=np.float32)
        for i, query in enumerate(queries):
            accepted = self._accepted(query, _phrases_of(phrases, i))
            if not accepted:
                continue                       # the signal is inactive for this query
            per_class = np.stack([self._class_values(m.class_id) for m in accepted])
            # NaN marks a fragment without items, and is NaN for every class of
            # that fragment at once, so the plain mean carries it through
            out[i] = per_class.mean(axis=0)
        return out

    def active(self, query, phrases=None) -> bool:
        return bool(self._accepted(query, phrases or {}))


class ObjectSignal(_ClosedVocabularySignal):
    """How strongly the detector saw the objects the query names.

    Three cases, and the middle one is the point: the class was detected (its
    best confidence), the class was NOT detected although the fragment has other
    detections (a raw zero -- the detector looked and did not find it), or the
    fragment has no detection at all (NaN -- the detector offered no evidence
    either way). After standardization the first outranks the third and the third
    outranks the second, which is what chapter 4 asks for.
    """

    name = "objects"
    kind = "object"

    def __init__(self, encoder, vocabulary, names, vocab_name, matching, size,
                 class_of_item: np.ndarray, score_of_item: np.ndarray,
                 fragment_of: np.ndarray, log=None) -> None:
        super().__init__(encoder, vocabulary, names, vocab_name, matching, size,
                         log=log)
        self.class_of_item = np.asarray(class_of_item, dtype=np.int32)
        self.score_of_item = np.asarray(score_of_item, dtype=np.float32)
        self.fragment_of = np.asarray(fragment_of, dtype=np.int32)
        if not (len(self.class_of_item) == len(self.score_of_item)
                == len(self.fragment_of)):
            raise ValueError(
                f"{self.name}: {len(self.class_of_item)} classes, "
                f"{len(self.score_of_item)} scores, {len(self.fragment_of)} fragments")
        self._has_items = np.zeros(size, dtype=bool)
        self._has_items[self.fragment_of] = True

    def _class_values(self, class_id: int) -> np.ndarray:
        best = np.full(self.size, -np.inf, dtype=np.float32)
        chosen = self.class_of_item == class_id
        if chosen.any():
            np.maximum.at(best, self.fragment_of[chosen], self.score_of_item[chosen])
        detected = np.where(np.isfinite(best), best, 0.0)
        return np.where(self._has_items, detected, np.nan)


class ExpressionSignal(_ClosedVocabularySignal):
    """The closed-vocabulary face mechanism: HSEmotion's eight class names.

    Shares its name with :class:`FaceRegionSignal` on purpose: they are the two
    mechanisms of ONE component (``face_regions`` with its ``mode``) and E5 puts
    them against each other, so exactly one is ever active.

    Every face carries a probability for every one of the eight classes, so there
    is no "looked and did not find it" case here: either the fragment has a face
    or it reports nothing.
    """

    name = "face_regions"
    kind = "expression"

    def __init__(self, encoder, vocabulary, names, vocab_name, matching, size,
                 distribution: np.ndarray, fragment_of: np.ndarray, log=None) -> None:
        super().__init__(encoder, vocabulary, names, vocab_name, matching, size,
                         log=log)
        self.distribution = np.asarray(distribution, dtype=np.float32)
        self.fragment_of = np.asarray(fragment_of, dtype=np.int32)

    def _class_values(self, class_id: int) -> np.ndarray:
        if not len(self.distribution):
            return np.full(self.size, np.nan, dtype=np.float32)
        per_face = self.distribution[:, class_id][None, :]
        return aggregate_max(per_face, self.fragment_of, self.size)[0]


class MotionSignal(_ClosedVocabularySignal):
    """How probable the actions the query names are, per fragment.

    One distribution per fragment over the whole Kinetics-400 vocabulary, so the
    matched class always has a value. A fragment the model never classified is a
    NaN row, not a row of zeros with a sentinel class.
    """

    name = "motion"
    kind = "action"

    def __init__(self, encoder, vocabulary, names, vocab_name, matching, size,
                 probability: np.ndarray, log=None) -> None:
        super().__init__(encoder, vocabulary, names, vocab_name, matching, size,
                         log=log)
        self.probability = np.asarray(probability, dtype=np.float32)
        if self.probability.shape != (size, len(self.names)):
            raise ValueError(
                f"{self.name}: distribution {self.probability.shape}, expected "
                f"{(size, len(self.names))}")

    def _class_values(self, class_id: int) -> np.ndarray:
        return self.probability[:, class_id]


# ------------------------------------------------------------------ identity
class IdentitySignal:
    """How well the fragment's faces match the profiles the query names.

    Active only for queries naming a profiled character. With several names the
    combination is conjunctive: the fragment scores high only when every named
    person is there, which is what rewards co-occurrence.
    """

    name = "identity"
    needs_phrases = False

    def __init__(self, profiles: dict[str, np.ndarray], vectors: np.ndarray,
                 fragment_of: np.ndarray, size: int) -> None:
        self.profiles = {name: np.asarray(matrix, dtype=np.float32)
                         for name, matrix in profiles.items()}
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.fragment_of = np.asarray(fragment_of, dtype=np.int32)
        self.size = size

    def _per_character(self, character: str) -> np.ndarray:
        """Best similarity of any face of a fragment to any example: (n_fragments,)."""
        examples = self.profiles[character]
        per_face = (self.vectors @ examples.T).max(axis=1)[None, :]
        return aggregate_max(per_face, self.fragment_of, self.size)[0]

    def raw_values(self, queries, phrases=None) -> np.ndarray:
        out = np.full((len(queries), self.size), np.nan, dtype=np.float32)
        if not len(self.vectors):
            return out
        cache: dict[str, np.ndarray] = {}
        for i, query in enumerate(queries):
            characters = resolve_characters(query.identities or [], self.profiles)
            if not characters:
                continue                       # the signal is inactive for this query
            for character in characters:
                if character not in cache:
                    cache[character] = self._per_character(character)
            # conjunction: the weakest of the named people decides the fragment
            out[i] = np.min(np.stack([cache[c] for c in characters]), axis=0)
        return out

    def active(self, query, phrases=None) -> bool:
        return bool(resolve_characters(query.identities or [], self.profiles))
