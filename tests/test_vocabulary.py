"""The closed vocabularies of the annotation schema."""

import json

import pytest

from src.annotation import tags
from src.evaluation.compare import subset_ids
from src.utils.vocabulary import (
    COMPLEXITY_VALUES,
    MASK_TAG,
    REQUIREMENT_TAGS,
    Complexity,
    RequirementTag,
    unknown_tags,
)


def test_the_tags_match_the_thesis_and_keep_its_order():
    assert REQUIREMENT_TAGS == ["wymaga_osoby", "wymaga_obiektu",
                                "wymaga_scenerii", "wymaga_ruchu",
                                "wymaga_mimiki"]
    assert COMPLEXITY_VALUES == ["P", "Z"]


def test_a_member_is_its_own_string():
    assert RequirementTag.SCENERY == "wymaga_scenerii"
    assert f"{Complexity.COMPLEX}" == "Z"
    assert "wymaga_ruchu" in REQUIREMENT_TAGS


def test_the_mask_tag_is_not_a_requirement_tag():
    assert MASK_TAG not in REQUIREMENT_TAGS
    assert RequirementTag.MOTION in REQUIREMENT_TAGS


def test_unknown_tags_pass_the_vocabularies_and_report_the_rest():
    assert unknown_tags(["wymaga_ruchu", MASK_TAG]) == []
    assert unknown_tags(["wymaga_ruch", "todo", "todo", ""]) == ["todo", "wymaga_ruch"]


def test_the_tag_file_uses_the_shared_vocabulary():
    assert tags.REQUIREMENT_TAGS is REQUIREMENT_TAGS
    assert tags.COMPLEXITY_VALUES is COMPLEXITY_VALUES
    assert set(REQUIREMENT_TAGS) <= set(tags.COLUMNS)


def test_a_misspelled_contrast_is_refused_rather_than_matching_nothing(tmp_path,
                                                                       monkeypatch):
    """On its own query file, not on the repository's.

    The annotation notebooks rewrite the real ones after every batch, so a test
    that reads them fails for a reason that has nothing to do with what it
    checks -- and what it checks is that a misspelling is refused before any
    file is opened.
    """
    from src.data import datasets

    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    folder = tmp_path / "data" / "annotations" / "office"
    folder.mkdir(parents=True)
    (folder / "office_queries_dev.jsonl").write_text(json.dumps({
        "desc_id": 1, "desc": "a man walks", "vid_name": "office_s01e01",
        "ts": [0.0, 5.0], "source": "office", "requirements": ["wymaga_scenerii"],
        "complexity": "P", "identities": [], "event_id": ""}),
        encoding="utf-8")

    with pytest.raises(ValueError, match="unknown requirements value"):
        subset_ids("office", "dev", "requirements:wymaga_scenrii")
    with pytest.raises(ValueError, match="unknown complexity value"):
        subset_ids("office", "dev", "complexity:zlozone")
    # the function now also selects subsets that are not contrasts
    # (identities:>=1 is a table column), so the message names a subset
    with pytest.raises(ValueError, match="unknown subset"):
        subset_ids("office", "dev", "whatever:x")
    # and a correct spelling still finds the query
    assert subset_ids("office", "dev", "requirements:wymaga_scenerii")[0] == {1}


# -------------------------------------------- which tag targets which signal
def test_a_signal_belongs_to_the_experiment_of_the_labels_that_add_it():
    from src.utils import experiments as exp

    assert exp.signal_experiment("caption") == "E3"      # E3-B and E3-C both add it
    assert exp.signal_experiment("objects") == "E4"
    assert exp.signal_experiment("identity") == "E6"
    # the base representation is not "added" by any label, so it has no experiment
    assert exp.signal_experiment("scene_embedding") is None


def test_the_target_tags_are_the_composition_of_the_two_tables():
    """Tag -> experiment (chapter 4) composed with experiment -> signal.

    Not a third list: a tag whose experiment moves has to move with it, and a
    signal whose experiment moves likewise. `wymaga_mimiki` is the hypothesis of
    E2 and E5 at once, so it targets the motion signal as well as the face one.
    """
    from src.utils import experiments as exp

    assert exp.target_tags("objects") == ["wymaga_obiektu"]
    assert exp.target_tags("identity") == ["wymaga_osoby"]
    assert exp.target_tags("face_regions") == ["wymaga_mimiki"]
    assert set(exp.target_tags("motion")) == {"wymaga_ruchu", "wymaga_mimiki"}
    # plain strings, because they are handed straight to compare.subset_ids
    assert all(type(tag) is str for tag in exp.target_tags("motion"))


def test_a_signal_no_tag_points_at_has_no_target_column():
    """E3 has no confirmatory tag of its own, and an empty list says so."""
    from src.utils import experiments as exp

    assert exp.target_tags("caption") == []
    assert exp.target_tags("scene_embedding") == []
