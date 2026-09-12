"""Validation of an interval file: what stops the pipeline and what only warns."""

from src.annotation import intervals as iv


def mask(seq, start, end):
    return iv.Interval(f"tbbt_s03e02_m{seq:03d}", iv.MASK, start, end)


def event(seq, start, end):
    return iv.Interval(f"tbbt_s03e02_{seq:03d}", iv.EVENT, start, end, "x" * 10)


def test_clean_file_has_no_errors():
    items = [mask(1, 0.0, 3.4), event(1, 10.0, 14.0), mask(2, 138.0, 163.0)]
    assert iv.validate(items, "tbbt", "s03e02") == []
    assert iv.warnings(items) == []


def test_overlapping_masks_are_an_error():
    items = [mask(1, 100.0, 110.0), mask(2, 105.0, 120.0)]
    problems = iv.validate(items, "tbbt", "s03e02")
    assert len(problems) == 1
    assert "overlaps mask" in problems[0]


def test_masks_touching_at_a_boundary_are_fine():
    # half-open intervals share no frame
    items = [mask(1, 100.0, 110.0), mask(2, 110.0, 120.0)]
    assert iv.validate(items, "tbbt", "s03e02") == []


def test_annotation_on_a_mask_warns_but_does_not_stop_the_file():
    # a TVR marker that starts inside a transition: it must reach the annotator
    items = [mask(1, 100.0, 101.35), event(1, 100.5, 103.0)]
    assert iv.validate(items, "tbbt", "s03e02") == []
    assert iv.warnings(items) == ["tbbt_s03e02_001: overlaps mask tbbt_s03e02_m001"]


def test_broken_identifiers_and_lengths_are_still_errors():
    items = [iv.Interval("nonsense", iv.EVENT, 1.0, 2.0),
             event(1, 5.0, 5.0)]
    problems = iv.validate(items, "tbbt", "s03e02")
    assert len(problems) == 2


# ---------------------------------------------------------------------- tags

def test_tags_survive_a_round_trip(tmp_path):
    # the class of a mask and the requirement markers of a query both live in
    # this column, so losing it silently would corrupt hand-made work
    items = [mask(1, 0.0, 3.4),
             iv.Interval("tbbt_s03e02_m002", iv.MASK, 733.85, 735.19,
                         tags=[iv.TAG_TRANSITION]),
             iv.Interval("tbbt_s03e02_001", iv.EVENT, 10.0, 14.0,
                         "x" * 10, ["wymaga_osoby", "wymaga_ruchu"])]
    path = iv.write(tmp_path / "tbbt_s03e02_intervals.csv", items)
    back = iv.read(path)
    assert [i.tags for i in back] == [[], ["przejscie"], ["wymaga_osoby", "wymaga_ruchu"]]
    assert [i.is_transition for i in back] == [False, True, False]


def test_a_file_without_a_tags_column_reads_untagged(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("id;type;start;end;desc\r\ntbbt_s03e02_m001;mask;0.00;3.40;\r\n",
                    encoding="utf-8-sig")
    assert iv.read(path)[0].tags == []
