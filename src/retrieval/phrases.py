"""Query phrases: the text side of the closed-vocabulary signals (chapter 4).

Before a signal can ask "does this fragment show a coffee mug", the query has to
say which words are worth asking about. Comparing the whole sentence with a list
of class names always returns a nearest name, even when the vocabulary holds no
word from the query, so the question is asked PHRASE by phrase instead.

Three kinds, extracted by independent rules, and the independence is deliberate:
*laugh* is both an action (Kinetics has ``laughing``) and a facial expression, and
both signals are meant to see it.

    object       what the detectors could find in a frame
    expression   what a face could be showing
    action       what a person could be doing

The rules are frozen: they were written before the matching threshold was
measured and must not be widened after seeing a result. Determinism comes from
pinned resources, checked the first time they are loaded.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from src.utils.queries import Query

#: the three kinds of phrase, in the order :func:`phrases_for` returns them
KINDS: tuple[str, ...] = ("object", "expression", "action")

Kind = Literal["object", "expression", "action"]

#: spaCy pipeline the rules were written against
SPACY_MODEL = "en_core_web_sm"

#: the threshold was measured with exactly these; a different version can move
#: the phrases, and with them the threshold, without anything looking broken
REQUIRED_VERSIONS = {"en_core_web_sm": "3.8.0", "nltk": "3.10.3", "wordnet": "3.0"}

#: nouns of relation and of position in the frame: WordNet files them as
#: artifacts or locations, but no detector reports them as an object.
#: FROZEN before the threshold measurement -- do not extend after seeing results.
STOP_WORDS = frozenset({
    "way", "side", "front", "back", "top", "bottom", "background", "foreground",
    "sort", "part", "end", "middle", "lot", "bit",
})

#: a noun is concrete enough to look for when ANY of its senses is one of these
CONCRETE_LEXNAMES = frozenset({
    "noun.artifact", "noun.object", "noun.animal", "noun.plant", "noun.food",
    "noun.substance", "noun.body",
})

#: the person filter; the closure of a synset does not contain the synset
#: itself, so equality has to be tested as well -- without it "a person" enters
#: as an object phrase, and COCO does have a `person` class
PERSON_ROOT = "person.n.01"

#: an expression is a feeling, a state or a face; nouns, by the dominant sense
FEELING_ROOTS = ("feeling.n.01", "emotional_state.n.01", "facial_expression.n.01")

#: ...or a verb of showing one; any sense, so `giggles` tagged as a noun counts.
#: `scream`, `yell` and `shout` sit under neither: WordNet calls them
#: vocalization, HSEmotion has no class for a scream and Kinetics-400 no
#: `screaming`, so they stay actions.
EXPRESSION_VERB_ROOTS = ("express_emotion.v.01", "grimace.v.01")

#: verbs that carry no action of their own
AUXILIARY_LEMMAS = frozenset({"be", "have", "do"})

#: object phrases per query; descriptions are short, this only bounds the odd one
MAX_OBJECT_PHRASES = 10


@dataclass(frozen=True)
class Phrase:
    """One thing asked about, in every spelling worth encoding.

    ``forms`` are what the matcher embeds -- the cleaned phrase and its head for
    an object, the base and the ``-ing`` spelling for an action -- and the best
    of them decides the phrase. ``head`` is the lemma the rules judged, ``span``
    the words the phrase was read from, kept for the logs.
    """

    forms: tuple[str, ...]
    head: str
    span: str


# ----------------------------------------------------------------- resources
@functools.lru_cache(maxsize=1)
def _resources():
    """The spaCy pipeline and WordNet, loaded once and version-checked.

    Loaded on first use rather than at import: the WordNet corpus alone costs
    about three seconds, and every module that ends up importing this one --
    signals, the runner, the cost script -- would pay it whether or not a phrase
    is ever extracted.
    """
    import spacy

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as error:                                  # pragma: no cover
        raise RuntimeError(
            f"the spaCy model {SPACY_MODEL!r} is not installed; it comes with "
            "requirements.txt") from error

    import nltk
    from nltk.corpus import wordnet

    try:
        wordnet_version = wordnet.get_version()
    except LookupError as error:                              # pragma: no cover
        raise RuntimeError(
            "the WordNet corpus is missing - run nltk.download(\"wordnet\")"
        ) from error

    _check_versions({SPACY_MODEL: nlp.meta["version"],
                     "nltk": nltk.__version__,
                     "wordnet": wordnet_version})
    return nlp, wordnet


def _check_versions(found: dict[str, str]) -> None:
    """Refuses to run on resources other than the ones the rules were written for.

    The rules and the matching threshold are one measurement: a phrase that a
    newer parser splits differently changes what the threshold was calibrated on.
    Failing loudly here is the point -- a quiet difference would show up as a
    threshold that no longer means what the thesis says it means.
    """
    wrong = {name: (value, REQUIRED_VERSIONS[name])
             for name, value in found.items() if value != REQUIRED_VERSIONS[name]}
    if wrong:
        detail = ", ".join(f"{name} {have} (expected {want})"
                           for name, (have, want) in sorted(wrong.items()))
        raise RuntimeError(
            f"phrase rules pinned to other resource versions: {detail}. "
            "The matching threshold was measured with the expected ones; "
            "measure it again before changing this check.")


# -------------------------------------------------------- WordNet predicates
def _under(synset, roots) -> bool:
    """Whether a synset IS one of the roots or has one among its hypernyms."""
    if synset in roots:
        return True
    hypernyms = set(synset.closure(lambda s: s.hypernyms()))
    return any(root in hypernyms for root in roots)


@functools.lru_cache(maxsize=4096)
def _noun_senses(lemma: str) -> tuple:
    _, wordnet = _resources()
    return tuple(wordnet.synsets(lemma, pos="n"))


@functools.lru_cache(maxsize=4096)
def _verb_senses(lemma: str) -> tuple:
    _, wordnet = _resources()
    return tuple(wordnet.synsets(lemma, pos="v"))


@functools.lru_cache(maxsize=1)
def _roots() -> tuple:
    """``(person, feeling nouns, expression verbs)`` as synsets."""
    _, wordnet = _resources()
    return ([wordnet.synset(PERSON_ROOT)],
            [wordnet.synset(name) for name in FEELING_ROOTS],
            [wordnet.synset(name) for name in EXPRESSION_VERB_ROOTS])


def _is_person(lemma: str) -> bool:
    """The person filter, by the DOMINANT sense.

    Looking at every sense instead would keep *man* and *guy*, which both have
    non-person senses, and would drop *mug*, which has one (*chump*).
    """
    senses = _noun_senses(lemma)
    person, _, _ = _roots()
    return bool(senses) and _under(senses[0], person)


def _is_concrete(lemma: str) -> bool:
    """The concreteness filter, by ANY sense.

    By the dominant sense instead this would drop *mug*, whose first sense is a
    measure of volume.
    """
    return any(sense.lexname() in CONCRETE_LEXNAMES for sense in _noun_senses(lemma))


def _is_expression_word(lemma: str, pos: str) -> bool:
    """Feeling noun, verb of showing one, or an adjective/adverb one step away."""
    _, feelings, expression_verbs = _roots()

    # An adjective or adverb is judged ONLY by the branch written for it.
    # `wordnet.synsets` lemmatises silently, so asking it about "broken" answers
    # about the verb "break" -- which has a sense under `express_emotion` -- and
    # every participial adjective would enter as an emotion.
    if pos in ("ADJ", "ADV"):
        return _derives_to_expression(lemma, pos)

    senses = _noun_senses(lemma)
    if senses and _under(senses[0], feelings):
        return True
    return any(_under(sense, expression_verbs) for sense in _verb_senses(lemma))


@functools.lru_cache(maxsize=4096)
def _derives_to_expression(lemma: str, pos: str) -> bool:
    """Whether an adjective or adverb reaches a feeling, by either route.

    Two are needed because WordNet 3.0 links these words in two different ways
    and neither covers the other. *sad* has a derivational link straight to
    ``sadness.n.01``; *surprised*, *scared* and the other participial adjectives
    have no derivational link at all, and are reached only through the verb they
    were formed from.
    """
    return (_derives_directly(lemma, pos)
            or any(_base_is_expression(_base_form(word))
                   for word in _morphological_sources(lemma, pos)))


def _derives_directly(lemma: str, pos: str) -> bool:
    """One derivational step from the word's OWN adjective or adverb senses.

    Only from the entry that spells the word itself, never from the other
    members of the synset. Following those would be synonym expansion in
    disguise, and it reaches absurd places: *broken* shares ``broken.s.01`` with
    *low*, which derives ``downheartedness.n.01``, and *blue* shares a synset
    with *gloomy*, so "a man in a blue shirt" reported an emotion.
    """
    _, wordnet = _resources()
    _, feelings, expression_verbs = _roots()
    wanted = lemma.lower().replace(" ", "_")
    for tag in ("a", "s", "r"):
        for synset in wordnet.synsets(lemma, pos=tag):
            for entry in synset.lemmas():
                if entry.name().lower() != wanted:
                    continue
                for related in (*entry.derivationally_related_forms(),
                                *entry.pertainyms()):
                    target = related.synset()
                    if target.pos() == "n" and _under(target, feelings):
                        return True
                    if target.pos() == "v" and _under(target, expression_verbs):
                        return True
    return False


def _morphological_sources(lemma: str, pos: str) -> tuple[str, ...]:
    """The words whose base to look at: for an adverb, the adjective it pertains to.

    *excitedly* says nothing on its own; it pertains to *excited*, and that is
    the word the base of which leads anywhere.
    """
    if pos == "ADV":
        _, wordnet = _resources()
        adjectives = tuple(dict.fromkeys(
            pertainym.name()
            for synset in wordnet.synsets(lemma, pos="r")
            for entry in synset.lemmas()
            for pertainym in entry.pertainyms()))
        if adjectives:
            return adjectives
    return (lemma,)


def _base_form(word: str) -> str:
    """The morphological base WordNet knows, the verb reading first.

    *surprised* -> *surprise*, *amused* -> *amuse*: the verb comes first because
    the participial adjective is exactly the case this exists for. A word that is
    no inflection of anything comes back unchanged.
    """
    _, wordnet = _resources()
    for tag in ("v", "n", "a"):
        found = wordnet.morphy(word, tag)
        if found:
            return found
    return wordnet.morphy(word) or word


@functools.lru_cache(maxsize=4096)
def _base_is_expression(base: str) -> bool:
    """Whether the base names a feeling, or makes one in a single step.

    First the base's own dominant noun sense (*surprise*, *shock*, *disgust*).
    Failing that, ONE derivation from its verb senses to a noun, judged by the
    same three roots -- *amuse* has no noun of its own but derives
    ``amusement.n.01``. One step only: two would walk from any verb to almost
    anything.
    """
    senses = _noun_senses(base)
    _, feelings, _ = _roots()
    if senses and _under(senses[0], feelings):
        return True
    for synset in _verb_senses(base):
        for entry in synset.lemmas():
            for related in entry.derivationally_related_forms():
                target = related.synset()
                if target.pos() == "n" and _under(target, feelings):
                    return True
    return False


# ---------------------------------------------------------- spelling helpers
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def gerund(verb: str) -> str:
    """The ``-ing`` spelling of a verb lemma; Kinetics-400 names its classes so.

    The ordinary English rules: *lie* -> *lying*, *dance* -> *dancing*,
    *run* -> *running*, everything else takes the suffix as it stands. Doubling
    is limited to one-syllable stems, which is where the stress certainly falls;
    *visit* and *travel* keep their single consonant.
    """
    word = verb.strip().lower()
    if not word:
        return word
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith(("ee", "oe", "ye")):
        return word[:-1] + "ing"
    if (len(_VOWEL_GROUPS.findall(word)) == 1
            and len(word) >= 3
            and word[-1] not in "wxy"
            and word[-1] not in "aeiou"
            and word[-2] in "aeiou"
            and word[-3] not in "aeiou"):
        return word + word[-1] + "ing"
    return word + "ing"


def _phrase_gerund(phrase: str) -> str:
    """The same phrase with its first word in the ``-ing`` spelling."""
    head, _, rest = phrase.partition(" ")
    return f"{gerund(head)} {rest}".strip() if rest else gerund(head)


def _unique(values) -> tuple[str, ...]:
    """Order-preserving deduplication of non-empty strings."""
    seen: set[str] = set()
    out = []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _character_names(characters: Sequence[str]) -> frozenset[str]:
    """What a query word has to equal to count as naming a character.

    The full name and its first word, as :func:`utils.vocabulary.resolve_characters`
    reads them: a profile is spelled "Michael Scott", a description says "Michael".
    """
    names: set[str] = set()
    for character in characters or ():
        label = str(character).strip().casefold()
        if label:
            names.add(label)
            names.add(label.split()[0])
    return frozenset(names)


def _names_a_character(token, names: frozenset[str]) -> bool:
    return token.text.casefold() in names or token.lemma_.casefold() in names


# ----------------------------------------------------------- the three rules
def _objects(doc, names: frozenset[str], log) -> list[Phrase]:
    """Noun chunks that name something a detector could report."""
    phrases = []
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.pos_ == "PRON":
            continue
        lemma = head.lemma_.lower()

        if head.ent_type_ == "PERSON" or _names_a_character(head, names):
            continue
        if _is_person(lemma):
            continue
        if lemma in STOP_WORDS:
            continue

        if not _noun_senses(lemma):
            log(f"phrase head unknown to WordNet, kept: {lemma!r}")
        elif not _is_concrete(lemma):
            continue

        # determiners and possessives say nothing about what is in the frame
        kept = [token.text for token in chunk
                if token.pos_ != "DET" and token.dep_ != "poss"]
        cleaned = " ".join(kept).lower()
        phrases.append(Phrase(_unique([cleaned, lemma]), lemma, chunk.text))
    return _dedup(phrases)[:MAX_OBJECT_PHRASES]


def _expressions(doc, names: frozenset[str], log) -> list[Phrase]:
    """Words for what a face is showing."""
    phrases = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        # "Joy" is a name before it is an emotion
        if token.pos_ == "PROPN" or _names_a_character(token, names):
            continue
        lemma = token.lemma_.lower()
        if not _is_expression_word(lemma, token.pos_):
            continue

        modifiers = [child for child in token.children if child.pos_ in ("ADJ", "ADV")]
        ordered = sorted([*modifiers, token], key=lambda t: t.i)
        # the token itself enters as its lemma, so a phrase without a modifier
        # is exactly the lemma and collapses to a single form
        qualified = " ".join(lemma if item is token else item.text.lower()
                             for item in ordered)
        span = doc[ordered[0].i:ordered[-1].i + 1].text
        phrases.append(Phrase(_unique([qualified, lemma]), lemma, span))
    return _dedup(phrases)


def _actions(doc, log) -> list[Phrase]:
    """Verbs, and the nouns that stand where a verb would."""
    phrases = []
    for token in doc:
        lemma = token.lemma_.lower()
        if token.pos_ == "AUX" or lemma in AUXILIARY_LEMMAS:
            continue

        if token.pos_ != "VERB":
            # spaCy reads "sprints" in "A man sprints from the lobby" as a noun,
            # so without this gate the rule loses the verb of a typical
            # description; without the predicate guard it would gain "lobby",
            # "board" and "man" instead
            if not (token.pos_ == "NOUN" and _verb_senses(lemma)
                    and _in_predicate(token)):
                continue
            log(f"noun in predicate position taken as an action: {lemma!r}")

        objects = [child for child in token.children if child.dep_ == "dobj"]
        base = f"{lemma} {objects[0].lemma_.lower()}" if objects else lemma
        phrases.append(Phrase(_unique([base, _phrase_gerund(base)]), lemma, token.text))
    return _dedup(phrases)


def _in_predicate(token) -> bool:
    """Whether the token stands where a sentence puts its verb."""
    return token.dep_ == "ROOT" or any(child.dep_ == "nsubj" for child in token.children)


def _dedup(phrases: list[Phrase]) -> list[Phrase]:
    """One entry per set of forms, in the order the description gives them."""
    seen: set[tuple[str, ...]] = set()
    out = []
    for phrase in phrases:
        if phrase.forms and phrase.forms not in seen:
            seen.add(phrase.forms)
            out.append(phrase)
    return out


# ------------------------------------------------------ the public interface
def _silent(_message: str) -> None:
    """Default log sink: extraction runs once per query, so it says nothing."""


def _from_doc(doc, kind: str, characters: Sequence[str], log) -> list[Phrase]:
    names = _character_names(characters)
    if kind == "object":
        return _objects(doc, names, log)
    if kind == "expression":
        return _expressions(doc, names, log)
    if kind == "action":
        return _actions(doc, log)
    raise ValueError(f"unknown phrase kind: {kind!r}; expected one of {KINDS}")


def extract(text: str, kind: Kind, *, characters: Sequence[str] = (),
            log: Callable[[str], None] = _silent) -> list[Phrase]:
    """The phrases of one kind in a query text; an empty list is a valid answer.

    ``characters`` are the query's ``identities`` -- names that have a profile,
    already filtered by :mod:`src.annotation.tags`, and empty for VATEX. A name
    without a profile can still get through as a word WordNet does not know,
    which is accepted behaviour and goes to the log.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown phrase kind: {kind!r}; expected one of {KINDS}")
    nlp, _ = _resources()
    return _from_doc(nlp(text), kind, characters, log)


def phrases_for(query: Query, *,
                log: Callable[[str], None] = _silent) -> dict[str, list[Phrase]]:
    """All three kinds of a query at once, from a SINGLE parse.

    The signals are handed these ready lists instead of extracting their own.
    That saves two parses per query, gives one object to log, and makes it
    structural -- not a matter of calling the same function twice -- that the
    face-region signal and the expression signal see the same list of expression
    phrases, which is what the E5 contrast has to compare.
    """
    nlp, _ = _resources()
    doc = nlp(query.desc)
    return {kind: _from_doc(doc, kind, query.identities, log) for kind in KINDS}
