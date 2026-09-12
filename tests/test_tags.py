"""The requirement-tag file: vocabulary, round trip and merging."""

import pytest

from src.annotation import tags
from src.utils.queries import Query


def register_row(desc_id, episode="s02e03", split="dev", desc="a query", tag_list=""):
    return {"desc_id": str(desc_id), "episode": episode, "split": split,
            "desc": desc, "tags": tag_list}


def filled(desc_id, **overrides):
    row = {key: "" for key in tags.COLUMNS}
    row.update({"desc_id": str(desc_id), "episode": "s02e03", "split": "dev",
                "desc": "a query"})
    row.update({tag: "0" for tag in tags.REQUIREMENT_TAGS})
    row["complexity"] = "P"
    row.update(overrides)
    return row


# ---------------------------------------------------------------- validation

def test_only_binary_values_and_known_complexity_pass():
    rows = {1: filled(1, wymaga_ruchu="2"), 2: filled(2, complexity="proste")}
    problems = tags.validate(rows)
    assert len(problems) == 2
    assert "wymaga_ruchu" in problems[0]
    assert "complexity" in problems[1]


def test_empty_cells_are_not_an_error():
    row = filled(1)
    row["wymaga_ruchu"] = ""
    assert tags.validate({1: row}) == []
    assert not tags.is_filled(row)


def test_empty_identities_do_not_block_completeness():
    assert tags.is_filled(filled(1, identities=""))


# ------------------------------------------------------------------- reading

def test_requirements_keep_the_order_of_the_vocabulary():
    row = filled(1, wymaga_mimiki="1", wymaga_obiektu="1")
    assert tags.requirements_of(row) == ["wymaga_obiektu", "wymaga_mimiki"]


def test_identities_are_split_and_trimmed():
    assert tags.identities_of(filled(1, identities="Dwight, Michael")) == \
        ["Dwight", "Michael"]
    assert tags.identities_of(filled(1)) == []


# -------------------------------------------------- skeleton keeps hand work

def test_skeleton_never_overwrites_a_filled_cell():
    existing = {1: filled(1, wymaga_ruchu="1", identities="Jim")}
    fresh, counts = tags.skeleton([register_row(1), register_row(2)], existing)
    assert fresh[1]["wymaga_ruchu"] == "1"
    assert fresh[1]["identities"] == "Jim"
    assert counts == {"added": 1, "kept": 1, "dropped": 0, "retexted": [],
                      "from_annotator": 0}
    assert tags.is_filled(fresh[1]) and not tags.is_filled(fresh[2])


def test_skeleton_keeps_the_wording_typed_here_and_reports_the_divergence():
    """Typos are fixed in this file, so the register must not overwrite them."""
    existing = {1: filled(1, desc="the wording fixed here")}
    fresh, counts = tags.skeleton([register_row(1, desc="the register wording")], existing)
    assert fresh[1]["desc"] == "the wording fixed here"
    assert counts["retexted"] == [1]      # flagged for a look, not resolved


def test_skeleton_fills_the_wording_in_when_the_cell_is_empty():
    fresh, counts = tags.skeleton([register_row(1, desc="from the register")], {})
    assert fresh[1]["desc"] == "from the register"
    assert counts["retexted"] == []


def test_skeleton_refreshes_the_context_columns():
    existing = {1: filled(1)}
    existing[1]["episode"] = "s99e99"
    fresh, _ = tags.skeleton([register_row(1)], existing)
    assert fresh[1]["episode"] == register_row(1)["episode"]


def test_skeleton_drops_rows_the_register_no_longer_has():
    fresh, counts = tags.skeleton([register_row(1)], {1: filled(1), 9: filled(9)})
    assert set(fresh) == {1}
    assert counts["dropped"] == 1


# ----------------------------------------- the tag coming from the annotator

def test_annotator_tag_marks_one_and_zeroes_the_rest_of_a_reviewed_episode():
    rows = [register_row(1, tag_list="wymaga_osoby"), register_row(2)]
    from_annotator = tags.from_register(rows)
    assert from_annotator[1]["wymaga_osoby"] == "1"
    assert from_annotator[2]["wymaga_osoby"] == "0"


def test_an_episode_with_no_marked_row_stays_undecided():
    from_annotator = tags.from_register([register_row(1), register_row(2)])
    assert from_annotator[1]["wymaga_osoby"] == ""
    assert from_annotator[2]["wymaga_osoby"] == ""


def test_an_unreviewed_episode_is_not_zeroed_by_a_reviewed_one():
    rows = [register_row(1, episode="s02e03", tag_list="wymaga_osoby"),
            register_row(2, episode="s03e15")]
    from_annotator = tags.from_register(rows)
    assert from_annotator[1]["wymaga_osoby"] == "1"
    assert from_annotator[2]["wymaga_osoby"] == ""


def test_every_requirement_tag_is_picked_up_from_the_annotator():
    rows = [register_row(1, tag_list="wymaga_scenerii wymaga_ruchu"),
            register_row(2, tag_list="wymaga_mimiki")]
    from_annotator = tags.from_register(rows)
    assert from_annotator[1]["wymaga_scenerii"] == "1"
    assert from_annotator[1]["wymaga_ruchu"] == "1"
    assert from_annotator[1]["wymaga_mimiki"] == "0"   # episode reviewed for it
    assert from_annotator[2]["wymaga_mimiki"] == "1"
    assert from_annotator[2]["wymaga_obiektu"] == ""   # nobody used that tag


def test_each_tag_decides_its_own_episodes():
    rows = [register_row(1, episode="s02e03", tag_list="wymaga_ruchu"),
            register_row(2, episode="s03e15", tag_list="wymaga_scenerii")]
    from_annotator = tags.from_register(rows)
    assert from_annotator[1]["wymaga_scenerii"] == ""   # s02e03 not reviewed for it
    assert from_annotator[2]["wymaga_ruchu"] == ""      # s03e15 not reviewed for it


def test_labels_outside_the_vocabulary_are_reported_not_applied():
    rows = [register_row(1, tag_list="wymaga_scenrii"),      # a typo
            register_row(2, tag_list="wymaga_ruchu")]
    summary = tags.annotator_report(rows)
    assert summary["unknown"] == ["wymaga_scenrii"]
    assert summary["marked"]["wymaga_ruchu"] == 1
    assert tags.from_register(rows)[1]["wymaga_scenerii"] == ""


def test_the_mask_tag_is_not_reported_as_unknown():
    assert tags.annotator_report([register_row(1, tag_list="przejscie")])["unknown"] == []


def test_skeleton_takes_the_annotator_tag_only_into_an_empty_cell():
    existing = {1: filled(1, wymaga_osoby="1")}
    fresh, counts = tags.skeleton([register_row(1), register_row(2)], existing,
                                  {1: {"wymaga_osoby": "0"}, 2: {"wymaga_osoby": "1"}})
    assert fresh[1]["wymaga_osoby"] == "1"    # hand work survives
    assert fresh[2]["wymaga_osoby"] == "1"
    assert counts["from_annotator"] == 1


# --------------------------------------------------------------------- merge

def query(desc_id):
    return Query(desc_id=desc_id, desc="a query", vid_name="office_s02e03",
                 ts=(0.0, 5.0), source="office")


def test_merge_writes_tags_into_the_records():
    records = [query(1)]
    counts = tags.merge(records, {1: filled(1, wymaga_scenerii="1", complexity="Z",
                                            identities="Pam")})
    assert records[0].requirements == ["wymaga_scenerii"]
    assert records[0].complexity == "Z"
    assert records[0].identities == ["Pam"]
    assert counts == {"tagged": 1, "untagged": 0, "identities_dropped": 0,
                      "retexted": 0}


def test_merge_leaves_a_record_with_a_half_filled_row_untouched():
    half = filled(1)
    half["wymaga_mimiki"] = ""
    records = [query(1), query(2)]
    counts = tags.merge(records, {1: half})
    assert records[0].requirements == [] and records[0].complexity is None
    assert counts == {"tagged": 0, "untagged": 2, "identities_dropped": 0,
                      "retexted": 0}


# --------------------------------------------------------------------- files

def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "office_query_tags.csv"
    tags.save({2: filled(2, identities="Jim"), 1: filled(1)}, path)
    back = tags.load(path)
    assert list(back) == [1, 2]           # ordered by episode, then identifier
    assert back[2]["identities"] == "Jim"


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert tags.load(tmp_path / "nothing.csv") == {}


def test_coverage_counts_per_episode():
    rows = {1: filled(1), 2: filled(2, wymaga_ruchu=""),
            3: filled(3, episode="s03e15")}
    stats = tags.coverage(rows, "dev")
    assert stats["total"] == 3 and stats["filled"] == 2
    assert stats["per_episode"]["s02e03"] == {"total": 2, "filled": 1}


def test_the_vocabulary_matches_the_thesis():
    assert tags.REQUIREMENT_TAGS == ["wymaga_osoby", "wymaga_obiektu",
                                     "wymaga_scenerii", "wymaga_ruchu",
                                     "wymaga_mimiki"]
    assert tags.COMPLEXITY_VALUES == ["P", "Z"]
    with pytest.raises(tags.TagError):
        tags._binary("tak", "wymaga_ruchu", 1)


# ----------------------------- the identity filter and its effect on the tag

PROFILED = ["Michael", "Dwight", "Jim", "Pam", "Kevin", "Angela"]


def test_only_profiled_characters_survive_the_merge():
    records = [query(1)]
    counts = tags.merge(records, {1: filled(1, identities="Michael, Erin, Dwight")},
                        PROFILED)
    assert records[0].identities == ["Michael", "Dwight"]   # Erin has no profile
    assert counts["identities_dropped"] == 1


def test_the_person_tag_follows_what_is_left_after_the_filter():
    # the file says the query needs a person, but the only name has no profile
    row = filled(1, wymaga_osoby="1", identities="Erin")
    records = [query(1)]
    tags.merge(records, {1: row}, PROFILED)
    assert records[0].identities == []
    assert "wymaga_osoby" not in records[0].requirements


def test_the_person_tag_is_set_when_a_profiled_name_remains():
    # and the other way round: the file says no, the name says yes
    row = filled(1, wymaga_osoby="0", identities="Pam")
    records = [query(1)]
    tags.merge(records, {1: row}, PROFILED)
    assert records[0].requirements == ["wymaga_osoby"]


def test_the_other_tags_are_untouched_by_the_filter():
    row = filled(1, wymaga_osoby="1", wymaga_ruchu="1", wymaga_scenerii="1",
                 identities="Erin")
    records = [query(1)]
    tags.merge(records, {1: row}, PROFILED)
    assert records[0].requirements == ["wymaga_scenerii", "wymaga_ruchu"]


def test_requirements_keep_the_vocabulary_order_after_the_filter():
    row = filled(1, wymaga_mimiki="1", wymaga_obiektu="1", identities="Jim")
    records = [query(1)]
    tags.merge(records, {1: row}, PROFILED)
    assert records[0].requirements == ["wymaga_osoby", "wymaga_obiektu",
                                       "wymaga_mimiki"]


def test_without_a_profile_list_the_file_is_taken_as_it_is():
    row = filled(1, wymaga_osoby="1", identities="Erin")
    records = [query(1)]
    tags.merge(records, {1: row})
    assert records[0].identities == ["Erin"]
    assert records[0].requirements == ["wymaga_osoby"]


def test_a_repeated_name_counts_once():
    records = [query(1)]
    tags.merge(records, {1: filled(1, identities="Jim, Jim")}, PROFILED)
    assert records[0].identities == ["Jim"]


def test_a_full_profile_name_matches_the_first_name_in_the_query():
    records = [query(1)]
    tags.merge(records, {1: filled(1, identities="Michael")}, ["Michael Scott"])
    assert records[0].identities == ["Michael Scott"]   # spelled as the profile


# ---------------------------------------------------------------- statistics

def test_statistics_count_tags_complexity_and_how_many_characters():
    records = [query(i) for i in range(4)]
    rows = {0: filled(0, wymaga_ruchu="1", complexity="Z", identities="Jim"),
            1: filled(1, wymaga_ruchu="1", complexity="P", identities="Jim, Pam"),
            2: filled(2, complexity="P", identities="Jim, Pam, Kevin"),
            3: filled(3, complexity="Z")}
    tags.merge(records, rows, PROFILED)
    stats = tags.statistics(records)
    assert stats["count"] == 4
    assert stats["requirements"]["wymaga_ruchu"] == 2
    assert stats["requirements"]["wymaga_osoby"] == 3
    assert stats["complexity"] == {"P": 2, "Z": 2}
    # the breakdown chapter 6 asks for: one, two, three characters, not "two or more"
    assert stats["identities"] == {0: 1, 1: 1, 2: 1, 3: 1}


def test_statistics_report_records_without_a_complexity_label():
    stats = tags.statistics([query(1)])
    assert stats["complexity_missing"] == 1
    assert stats["identities"] == {0: 1}

def test_merge_takes_the_corrected_wording_into_the_record():
    """A typo fix reaches the query files even when the tags are still empty."""
    records = [query(1)]
    counts = tags.merge(records, {1: filled(1, desc="a query, spelled right")})
    assert records[0].desc == "a query, spelled right"
    assert counts["retexted"] == 1


# --------------------------------------------- the narrower VATEX column set

def vatex_row(desc_id, episode="G9zN5TTuGO4_000179_000189", **overrides):
    row = {key: "" for key in tags.VATEX_COLUMNS}
    row.update({"desc_id": str(desc_id), "episode": episode, "split": "test",
                "desc": "a clip description", "wymaga_obiektu": "1",
                "wymaga_scenerii": "0", "wymaga_ruchu": "1", "complexity": "P"})
    row.update(overrides)
    return row


def test_vatex_carries_three_tags_and_no_characters():
    assert tags.columns_for("vatex") == tags.VATEX_COLUMNS
    assert tags.tags_in(tags.VATEX_COLUMNS) == ["wymaga_obiektu",
                                                "wymaga_scenerii", "wymaga_ruchu"]
    assert "identities" not in tags.VATEX_COLUMNS


def test_an_unknown_source_gets_the_series_columns():
    assert tags.columns_for("office") == tags.COLUMNS
    assert tags.tags_in(tags.COLUMNS) == tags.REQUIREMENT_TAGS


def test_a_row_is_filled_without_the_tags_vatex_does_not_assign():
    """The two absent tags must not make every VATEX row look half-filled."""
    row = vatex_row(1)
    assert tags.is_filled(row, tags.VATEX_COLUMNS)
    assert not tags.is_filled(row)                     # judged as a series row
    assert tags.validate({1: row}, tags.VATEX_COLUMNS) == []


def test_vatex_merge_writes_the_three_tags_and_leaves_identities_empty():
    records = [query(1)]
    counts = tags.merge(records, {1: vatex_row(1)}, columns=tags.VATEX_COLUMNS)
    assert records[0].requirements == ["wymaga_obiektu", "wymaga_ruchu"]
    assert records[0].complexity == "P"
    assert records[0].identities == []
    assert counts["tagged"] == 1


def test_vatex_statistics_report_no_row_for_a_tag_that_is_never_assigned():
    records = [query(1)]
    tags.merge(records, {1: vatex_row(1)}, columns=tags.VATEX_COLUMNS)
    stats = tags.statistics(records, tags.VATEX_COLUMNS)
    assert set(stats["requirements"]) == {"wymaga_obiektu", "wymaga_scenerii",
                                          "wymaga_ruchu"}
    assert stats["requirements"]["wymaga_obiektu"] == 1


def test_the_vatex_file_holds_only_its_own_columns(tmp_path):
    path = tmp_path / "vatex_query_tags.csv"
    tags.save({1: vatex_row(1)}, path, tags.VATEX_COLUMNS)
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == "desc_id;episode;split;desc;" \
                     "wymaga_obiektu;wymaga_scenerii;wymaga_ruchu;complexity"
    assert set(tags.load(path, tags.VATEX_COLUMNS)[1]) == set(tags.VATEX_COLUMNS)


def test_the_vatex_skeleton_keeps_hand_work_and_writes_no_absent_column():
    source = [{"desc_id": "1", "episode": "G9zN5TTuGO4_000179_000189",
               "split": "test", "desc": "a clip description"}]
    fresh, counts = tags.skeleton(source, {1: vatex_row(1, wymaga_ruchu="0")},
                                 columns=tags.VATEX_COLUMNS)
    assert fresh[1]["wymaga_ruchu"] == "0"
    assert "wymaga_osoby" not in fresh[1] and "identities" not in fresh[1]
    assert counts["kept"] == 1 and counts["added"] == 0


def test_the_series_path_is_untouched_by_the_new_argument():
    """Every default is the series column set, so nothing above changes."""
    existing = {1: filled(1, wymaga_ruchu="1", identities="Jim")}
    rows = [register_row(1), register_row(2)]
    assert (tags.skeleton(rows, existing)
            == tags.skeleton(rows, existing, columns=tags.COLUMNS))
    assert tags.coverage(existing) == tags.coverage(existing, columns=tags.COLUMNS)


# -------------------------------------- the identifier and what it points at

def test_a_row_naming_another_episode_than_its_identifier_is_reported():
    rows = {1: filled(1, episode="s02e03"), 2: filled(2, episode="s02e03")}
    drift = tags.misaligned(rows, {1: "s02e03", 2: "s09e14"})
    assert drift == [(2, "s02e03", "s09e14")]


def test_a_row_the_current_set_no_longer_has_is_not_a_mismatch():
    """Those are dropped by `skeleton`; only a MOVED identifier is a problem."""
    rows = {1: filled(1, episode="s02e03"), 9: filled(9, episode="s02e03")}
    assert tags.misaligned(rows, {1: "s02e03"}) == []
