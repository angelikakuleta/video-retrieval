"""Closed vocabularies of the annotation schema.

The requirement tags and the complexity label (chapter 4) travel through the
annotator, the register, the tag file and the query records. Spelling them once
here is what keeps those four from drifting apart.

The names are Polish because that is how the thesis and the annotator define
them, and how they appear on the command line
(``--contrast requirements:wymaga_scenerii``).
"""

from __future__ import annotations

from enum import StrEnum


class RequirementTag(StrEnum):
    """What kind of visual information a query needs to be resolved.

    Independent of one another: a query may carry several or none. The
    experiment each one is the confirmatory contrast of is given in
    :data:`TAG_EXPERIMENT`.
    """

    PERSON = "wymaga_osoby"
    OBJECT = "wymaga_obiektu"
    SCENERY = "wymaga_scenerii"
    MOTION = "wymaga_ruchu"
    EXPRESSION = "wymaga_mimiki"


class Complexity(StrEnum):
    """One person or object with one action, or more than that."""

    SIMPLE = "P"
    COMPLEX = "Z"


#: tags in the order of table "znaczniki" in chapter 4
REQUIREMENT_TAGS: list[str] = [tag.value for tag in RequirementTag]

#: complexity labels, as written in table "format-adnotacji"
COMPLEXITY_VALUES: list[str] = [level.value for level in Complexity]

#: the experiment whose confirmatory contrast each tag carries (chapter 4,
#: table "znaczniki"); mimicry is the hypothesis of two experiments at once
TAG_EXPERIMENT: dict[str, tuple[str, ...]] = {
    RequirementTag.PERSON: ("E6",),
    RequirementTag.OBJECT: ("E4",),
    RequirementTag.SCENERY: ("E1",),
    RequirementTag.MOTION: ("E2",),
    RequirementTag.EXPRESSION: ("E2", "E5"),
}

#: the tag an annotator puts on a MASK to mark a transition absorbed by the
#: content axis; not a requirement tag, and listed here so that reading the
#: annotator's tags can tell the two apart
MASK_TAG = "przejscie"


def resolve_characters(names, profiled) -> list[str]:
    """Which of the profiled characters a list of names refers to.

    A query says "Michael", a profile may be "Michael Scott": within a series the
    first name identifies the character, so a name matches on equality with the
    profile or its first word, ignoring case. The result is spelled as ``profiled``
    spells it. Names without a profile drop out -- the identity signal cannot act
    on them.
    """
    resolved = []
    for name in names or []:
        wanted = str(name).strip().casefold()
        if not wanted:
            continue
        for profile in profiled or []:
            label = str(profile).casefold()
            if wanted == label or wanted == label.split()[0]:
                resolved.append(profile)
                break
    seen, out = set(), []
    for name in resolved:            # keep the order of the query, drop repeats
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def unknown_tags(names) -> list[str]:
    """The given tags that belong to no vocabulary, sorted and deduplicated.

    The annotator accepts free-form labels, so a stray one is not an error -- but a
    misspelled requirement tag looks exactly like a note to self and would empty a
    subset without a word.
    """
    known = set(REQUIREMENT_TAGS) | {MASK_TAG}
    return sorted({name for name in names if name and name not in known})
