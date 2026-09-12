"""Query phrase rules: the input-output pairs the rules were frozen on.

Every case below is a row of the rule table in the rebuild plan. They are not
illustrations: the matching threshold is calibrated on exactly these rules, so a
row that stops passing after a spaCy or WordNet upgrade is a reason to measure
the threshold again, not a reason to edit the expectation.
"""

import pytest

from src.retrieval import phrases
from src.utils.queries import Query


def rendered(items):
    """Phrases as the rule table writes them: forms joined by '/', then ', '."""
    return ", ".join("/".join(item.forms) for item in items)


def bases(items):
    """Only the base form of each phrase, which is how the table lists actions."""
    return ", ".join(item.forms[0] for item in items)


# ------------------------------------------------------------ the rule table
OBJECT_CASES = [
    ("A man in a blue shirt puts a coffee mug on the desk for a moment", (),
     "blue shirt/shirt, coffee mug/mug, desk"),
    ("Sheldon laughs at Leonard", ("Sheldon", "Leonard"), ""),
    ("A robot vacuum Roomba cleans the way", (), "robot vacuum roomba/roomba"),
    ("A person walks into the room", (), "room"),
    ("Two people are dancing on a stage", (), "stage"),
]

EXPRESSION_CASES = [
    ("Penny giggles and pours herself an alcoholic drink", (), "giggle"),
    ("A woman screams at the television", (), ""),
]

ACTION_CASES = [
    ("Sheldon sprints from the lobby", "sprint"),
    ("Penny giggles and pours herself an alcoholic drink", "giggle, pour drink"),
    ("A man reads a board", "read board"),
    ("A woman walks into the kitchen", "walk"),
]


@pytest.mark.parametrize("text, characters, expected", OBJECT_CASES)
def test_object_rule(text, characters, expected):
    assert rendered(phrases.extract(text, "object", characters=characters)) == expected


@pytest.mark.parametrize("text, characters, expected", EXPRESSION_CASES)
def test_expression_rule(text, characters, expected):
    assert rendered(phrases.extract(text, "expression",
                                    characters=characters)) == expected


@pytest.mark.parametrize("text, expected", ACTION_CASES)
def test_action_rule(text, expected):
    assert bases(phrases.extract(text, "action")) == expected


def test_an_action_also_carries_the_ing_spelling():
    """Kinetics-400 names its classes as gerunds, so the phrase offers both."""
    found = {phrase.forms for phrase in
             phrases.extract("Penny giggles and pours herself an alcoholic drink",
                             "action")}
    assert found == {("giggle", "giggling"), ("pour drink", "pouring drink")}


# --------------------------------------------- the decisions behind the rows
def test_a_person_is_rejected_by_equality_not_only_by_hypernyms():
    """`person.n.01` is not in its own hypernym closure.

    Without the equality test "a person" enters as an object phrase, and COCO
    does have a `person` class, so the signal would answer on it.
    """
    assert phrases._is_person("person")
    assert phrases._is_person("man")            # by the closure
    assert not phrases._is_person("mug")        # a non-dominant person sense only


def test_the_concreteness_filter_looks_at_every_sense():
    """By the dominant sense alone `mug` would go, its first sense being a volume."""
    assert phrases._is_concrete("mug")
    assert not phrases._is_concrete("moment")


def test_the_stop_list_is_the_frozen_fourteen():
    assert len(phrases.STOP_WORDS) == 14
    # dropped from the first version: the concreteness filter already rejects them
    for dead in ("moment", "time", "kind"):
        assert dead not in phrases.STOP_WORDS
        assert not phrases._is_concrete(dead)


def test_a_stop_word_is_dropped_although_wordnet_calls_it_concrete():
    assert phrases._is_concrete("way")          # noun.artifact among its senses
    assert rendered(phrases.extract("A dog blocks the way", "object")) == "dog"


def test_scream_is_an_action_and_not_an_expression():
    """WordNet files it as vocalization; HSEmotion has no class for a scream."""
    text = "A woman screams at the television"
    assert phrases.extract(text, "expression") == []
    assert bases(phrases.extract(text, "action")) == "scream"


def test_the_kinds_do_not_exclude_each_other():
    """`laugh` is a Kinetics class and a facial expression; both signals see it."""
    text = "Sheldon laughs at the joke"
    assert bases(phrases.extract(text, "expression")) == "laugh"
    assert bases(phrases.extract(text, "action")) == "laugh"


def test_the_predicate_guard_keeps_the_nouns_out():
    """Without it "A man reads a board" also yields `man` and `board`."""
    assert bases(phrases.extract("A man reads a board", "action")) == "read board"
    assert bases(phrases.extract("Sheldon sprints from the lobby", "action")) == "sprint"


def test_an_auxiliary_is_not_an_action():
    assert bases(phrases.extract("Two people are dancing on a stage", "action")) == "dance"


def test_an_expression_keeps_its_modifier_as_a_second_form():
    assert rendered(phrases.extract("He looks very uncomfortable", "expression")) == \
        "very uncomfortable/uncomfortable"


def test_a_name_is_not_a_thing_to_detect_nor_an_emotion():
    """Otherwise `Sheldon` enters as an object and `Joy` as a feeling."""
    assert phrases.extract("Sheldon laughs at Leonard", "object",
                           characters=("Sheldon", "Leonard")) == []
    # the profile is spelled in full, the description uses the first name
    assert phrases.extract("Michael laughs at Dwight", "object",
                           characters=("Michael Scott", "Dwight Schrute")) == []

    # `joy` is a feeling noun, but here the word is a name; `smiles` stays
    found = phrases.extract("Joy smiles at the camera", "expression",
                            characters=("Joy",))
    assert [phrase.head for phrase in found] == ["smile"]


def test_a_word_unknown_to_wordnet_is_kept_and_logged():
    messages = []
    found = phrases.extract("A robot vacuum Roomba cleans the way", "object",
                            log=messages.append)
    assert rendered(found) == "robot vacuum roomba/roomba"
    assert any("roomba" in message for message in messages)


# ------------------------------------------------------------- the interface
def test_phrases_for_returns_all_three_kinds_of_a_query():
    query = Query(desc_id=1, desc="Penny giggles and pours herself an alcoholic drink",
                  vid_name="tbbt_s01e15", ts=(0.0, 5.0), source="tbbt",
                  identities=["Penny"])
    found = phrases.phrases_for(query)

    assert list(found) == list(phrases.KINDS)
    assert rendered(found["expression"]) == "giggle"
    assert bases(found["action"]) == "giggle, pour drink"
    assert rendered(found["object"]) == "alcoholic drink/drink"


def test_phrases_for_parses_the_query_once(monkeypatch):
    """Three kinds off ONE parse: that is why the signals are handed ready lists.

    It saves two parses per query, and it makes it structural rather than a
    matter of calling the same function twice that the face-region signal and
    the expression signal see the same list -- which is what E5 compares.
    """
    nlp, wordnet = phrases._resources()
    parsed = []

    class Counting:
        meta = nlp.meta

        def __call__(self, text):
            parsed.append(text)
            return nlp(text)

    monkeypatch.setattr(phrases, "_resources", lambda: (Counting(), wordnet))
    query = Query(desc_id=1, desc="A nervous man smiles at the camera",
                  vid_name="tbbt_s01e15", ts=(0.0, 5.0), source="tbbt")
    found = phrases.phrases_for(query)

    assert parsed == ["A nervous man smiles at the camera"]
    assert rendered(found["expression"]) == "nervous, smile"


def test_no_phrases_is_an_empty_list_not_an_error():
    for kind in phrases.KINDS:
        assert phrases.extract("", kind) == []


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown phrase kind"):
        phrases.extract("A man reads a board", "objects")


# ---------------------------------------------------- spelling and resources
@pytest.mark.parametrize("verb, expected", [
    ("walk", "walking"), ("read", "reading"), ("pour", "pouring"),
    ("giggle", "giggling"), ("dance", "dancing"), ("run", "running"),
    ("sit", "sitting"), ("hug", "hugging"), ("swim", "swimming"),
    ("cry", "crying"), ("lie", "lying"), ("see", "seeing"), ("dye", "dyeing"),
    ("visit", "visiting"), ("open", "opening"), ("travel", "traveling"),
])
def test_gerund_spelling(verb, expected):
    assert phrases.gerund(verb) == expected


def test_the_pinned_versions_are_the_ones_installed():
    """The rules and the threshold are one measurement; a swap has to be loud."""
    phrases._resources()                      # passes the check on this environment
    with pytest.raises(RuntimeError, match="measure it again"):
        phrases._check_versions({"wordnet": "3.1"})


# ------------------------------------------------ the participial adjectives
#: emotions spelled as a past participle. WordNet 3.0 gives them no derivational
#: link of their own, so they are reached through the verb they were formed from.
PARTICIPLES = ["surprised", "scared", "shocked", "disgusted", "excited",
               "annoyed", "amused", "confused", "embarrassed", "upset"]

#: participles and plain adjectives that describe no feeling at all
NOT_FEELINGS = ["seated", "broken", "wooden", "opened", "red"]


@pytest.mark.parametrize("word", PARTICIPLES)
def test_a_participial_adjective_names_the_feeling_it_was_formed_from(word):
    assert phrases._is_expression_word(word, "ADJ")


@pytest.mark.parametrize("word", NOT_FEELINGS)
def test_an_adjective_that_describes_no_feeling_stays_out(word):
    assert not phrases._is_expression_word(word, "ADJ")


def test_the_base_is_taken_before_the_derivation():
    """`surprise` is a feeling itself; `amuse` only makes one."""
    assert phrases._base_form("surprised") == "surprise"
    assert phrases._base_is_expression("surprise")      # its own dominant noun
    assert phrases._base_form("amused") == "amuse"
    assert phrases._base_is_expression("amuse")         # one step to amusement
    assert not phrases._base_is_expression("break")     # neither route


def test_an_adverb_goes_through_the_adjective_it_pertains_to():
    """`excitedly` says nothing on its own; `excited` leads to `excite`."""
    assert phrases._morphological_sources("excitedly", "ADV") == ("excited",)
    assert phrases._is_expression_word("excitedly", "ADV")
    assert rendered(phrases.extract("She waves excitedly at the camera",
                                    "expression")) == "excitedly"


def test_a_derivation_is_followed_only_from_the_word_itself():
    """Following the other members of a synset would be synonym expansion.

    `broken` shares `broken.s.01` with `low`, which derives `downheartedness`;
    `blue` shares a synset with `gloomy`. Both used to come back as emotions.
    """
    assert not phrases._derives_directly("broken", "ADJ")
    assert phrases._derives_directly("sad", "ADJ")          # its own lemma derives
    assert phrases.extract("A man in a blue shirt puts a coffee mug on the desk",
                           "expression") == []


def test_an_adjective_is_not_judged_as_the_verb_it_inflects():
    """`wordnet.synsets` lemmatises, and `break` has a sense under express_emotion.

    Without the gate every participial adjective would enter through the verb
    branch, `broken` included.
    """
    assert phrases._is_expression_word("break", "VERB")     # the verb still counts
    assert not phrases._is_expression_word("broken", "ADJ")


@pytest.mark.parametrize("sentence, expected", [
    ("A surprised man opens the door", "surprised"),
    ("He looks embarrassed and upset", "embarrassed upset/embarrassed, upset"),
    ("A broken window lies on the seated man", ""),
    ("The wooden door was opened", ""),
])
def test_the_extended_rule_on_whole_sentences(sentence, expected):
    assert rendered(phrases.extract(sentence, "expression")) == expected

