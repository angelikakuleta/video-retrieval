"""VATEX query records: which identifier a query gets and how stable it is."""

import pandas as pd

from src.utils import vatex

CLIPS = ["aaaaaaaaaaa_000000_000010", "bbbbbbbbbbb_000012_000022",
         "ccccccccccc_000030_000040"]


def descriptions(clips=CLIPS, per_clip=10):
    """Ten English descriptions per clip, in the shape of vatex_descriptions_test.csv."""
    return pd.DataFrame([
        {"videoID": vid, "desc_no": n, "desc_en": f"{vid} description {n}"}
        for vid in clips for n in range(1, per_clip + 1)
    ])


# ------------------------------------------------- the identifier of a query

def test_the_identifier_is_derived_from_the_clip_name():
    assert vatex.desc_id_for(CLIPS[0]) == vatex.desc_id_for(CLIPS[0])
    assert vatex.desc_id_for(CLIPS[0]) != vatex.desc_id_for(CLIPS[1])
    assert 0 <= vatex.desc_id_for(CLIPS[0]) < 10 ** vatex.DESC_ID_DIGITS


def test_dropping_a_clip_does_not_move_the_other_identifiers():
    """The tag file is keyed by desc_id: a shift would move the hand work."""
    frame = descriptions()
    before = {q.vid_name: q.desc_id for q in vatex.build_queries(CLIPS, frame)}
    after = {q.vid_name: q.desc_id for q in vatex.build_queries(CLIPS[1:], frame)}
    assert all(before[vid] == after[vid] for vid in after)


def test_the_experiment_queries_take_the_first_description_of_every_clip():
    records = vatex.build_queries(CLIPS, descriptions())
    assert [q.vid_name for q in records] == sorted(CLIPS)
    assert all(q.desc.endswith("description 1") for q in records)
    assert all(q.event_id == q.vid_name for q in records)
    assert records[0].ts == (0.0, 10.0)


def test_a_clip_outside_the_given_set_gets_no_query():
    records = vatex.build_queries(CLIPS[:1], descriptions())
    assert [q.vid_name for q in records] == CLIPS[:1]


# ------------------------------------------------------ the check query file

def test_the_check_queries_are_all_ten_descriptions_numbered_by_position():
    records = vatex.build_check_queries(CLIPS, descriptions())
    assert len(records) == 30
    assert [q.desc_id for q in records] == list(range(30))
    assert [q.vid_name for q in records[:10]] == [sorted(CLIPS)[0]] * 10


def test_the_two_sets_are_numbered_independently():
    """The check file keeps its positional numbering; the experiment file does not."""
    frame = descriptions()
    check = vatex.build_check_queries(CLIPS, frame)
    experiment = vatex.build_queries(CLIPS, frame)
    assert {q.desc_id for q in check} == set(range(30))
    assert all(q.desc_id > 30 for q in experiment)


# -------------------------------------------------- the rows of the tag file

def test_the_tag_rows_pin_every_identifier_to_its_clip():
    records = vatex.build_queries(CLIPS, descriptions())
    rows = vatex.tag_source_rows(records)
    assert [row["episode"] for row in rows] == [q.vid_name for q in records]
    assert [row["desc_id"] for row in rows] == [str(q.desc_id) for q in records]
    assert {row["split"] for row in rows} == {"test"}
