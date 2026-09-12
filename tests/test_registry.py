"""The register: what the verification round trip does to a row."""

from src.annotation import intervals as iv
from src.annotation import registry as reg
from src.utils.settings import MAX_DURATION, MIN_DURATION


def revalidate(episode: str, start: float, end: float) -> str:
    """Length verdict for the times in the file, as the notebooks build it."""
    length = end - start
    if length < MIN_DURATION:
        return reg.REASON_TOO_SHORT
    if length > MAX_DURATION:
        return reg.REASON_TOO_LONG
    return ""


REJECTED, ACCEPTED = "tbbt_s01e01_d001", "tbbt_s01e01_d002"


def register():
    """One row rejected on its length, one accepted."""
    return [
        reg.row(REJECTED, "s01e01", "dev", "f.mp4", 10.0, 10.5, "short",
                status=reg.STATUS_REJECTED, reason=reg.REASON_TOO_SHORT),
        reg.row(ACCEPTED, "s01e01", "dev", "f.mp4", 20.0, 25.0, "fine"),
    ]


def event(start, end, annotation_id=REJECTED, desc=None):
    if desc is None:
        desc = "short" if annotation_id == REJECTED else "fine"
    return iv.Interval(annotation_id, iv.EVENT, start, end, desc)


def apply(items, rows=None, **kwargs):
    rows = rows if rows is not None else register()
    summary = reg.update(rows, {"s01e01": items}, {"s01e01": "dev"},
                         {"s01e01": "f.mp4"}, **kwargs)
    return {r["annotation_id"]: r for r in rows}, summary["s01e01"]


def both(rejected=(10.0, 10.5), accepted=(20.0, 25.0),
         rejected_desc=None, accepted_desc=None):
    """The whole file: both rows, optionally moved or reworded."""
    return [event(*rejected, REJECTED, rejected_desc),
            event(*accepted, ACCEPTED, accepted_desc)]


# ------------------------------------------------------------------- removal

def test_deleting_a_rejected_row_changes_nothing():
    """Only accepted rows count as removed, so the automatic reason stays."""
    by_id, counts = apply([event(20.0, 25.0, ACCEPTED)])
    assert by_id[REJECTED]["status"] == reg.STATUS_REJECTED
    assert by_id[REJECTED]["reason"] == reg.REASON_TOO_SHORT
    assert counts["removed"] == 0


def test_deleting_an_accepted_row_marks_it_manual():
    by_id, counts = apply([event(10.0, 10.5)])
    assert by_id[ACCEPTED]["status"] == reg.STATUS_REJECTED
    assert by_id[ACCEPTED]["reason"] == reg.REASON_MANUAL
    assert counts["removed"] == 1


def test_a_manual_rejection_returns_when_the_row_is_back():
    rows = register()
    rows[0]["reason"] = reg.REASON_MANUAL
    by_id, counts = apply(both(), rows=rows, revalidate=revalidate)
    assert by_id[REJECTED]["status"] == reg.STATUS_ACCEPTED
    assert counts["restored"] == 1


def test_a_new_row_gets_manual_origin():
    extra = iv.Interval("tbbt_s01e01_001", iv.EVENT, 40.0, 45.0, "added by hand")
    by_id, counts = apply(both() + [extra])
    assert by_id["tbbt_s01e01_001"]["origin"] == reg.ORIGIN_MANUAL
    assert by_id["tbbt_s01e01_001"]["status"] == reg.STATUS_ACCEPTED
    assert counts["added"] == 1


# --------------------------------------------------------------------- times

def test_a_pure_shift_only_updates_the_times():
    """Sliding an interval cannot change the verdict, so nothing is flagged."""
    by_id, counts = apply(both(accepted=(21.0, 26.0)), revalidate=revalidate)
    row = by_id[ACCEPTED]
    assert float(row["start"]) == 21.0 and float(row["end"]) == 26.0
    assert row["edited_duration"] == reg.NO and row["edited_desc"] == reg.NO
    assert row["status"] == reg.STATUS_ACCEPTED
    assert counts["moved"] == 1 and counts["edited_duration"] == 0


def test_a_length_change_is_flagged_and_re_measured():
    by_id, counts = apply(both(rejected=(10.0, 14.0)), revalidate=revalidate)
    row = by_id[REJECTED]
    assert row["edited_duration"] == reg.YES
    assert row["status"] == reg.STATUS_ACCEPTED and row["reason"] == ""
    assert counts["edited_duration"] == 1 and counts["recovered"] == 1


def test_a_repaired_row_that_still_fails_keeps_a_current_reason():
    by_id, counts = apply(both(rejected=(10.0, 40.0)), revalidate=revalidate)
    row = by_id[REJECTED]
    assert row["status"] == reg.STATUS_REJECTED
    assert row["reason"] == reg.REASON_TOO_LONG      # was too_short
    assert float(row["end"]) == 40.0
    assert counts["recovered"] == 0 and counts["demoted"] == 0


def test_stretching_an_accepted_row_past_the_limit_rejects_it():
    """The verdict follows the length in the file, in both directions."""
    by_id, counts = apply(both(accepted=(20.0, 50.0)), revalidate=revalidate)
    row = by_id[ACCEPTED]
    assert row["status"] == reg.STATUS_REJECTED
    assert row["reason"] == reg.REASON_TOO_LONG
    assert counts["demoted"] == 1 and counts["edited_duration"] == 1


def test_without_revalidate_times_are_taken_but_no_verdict_is_passed():
    by_id, counts = apply(both(rejected=(10.0, 14.0)))
    assert float(by_id[REJECTED]["end"]) == 14.0
    assert by_id[REJECTED]["edited_duration"] == reg.YES
    assert by_id[REJECTED]["reason"] == reg.REASON_TOO_SHORT
    assert counts["recovered"] == 0


# ------------------------------------------------------------------- wording

def test_a_typo_fix_sets_only_edited_desc():
    """The text is what the query files are built from, so it stays traceable."""
    by_id, counts = apply(both(accepted_desc="fine, spelled right"),
                          revalidate=revalidate)
    row = by_id[ACCEPTED]
    assert row["desc"] == "fine, spelled right"
    assert row["edited_desc"] == reg.YES and row["edited_duration"] == reg.NO
    assert counts["edited_desc"] == 1 and counts["moved"] == 0


def test_length_and_wording_are_counted_apart():
    by_id, counts = apply(both(accepted=(20.0, 28.0), accepted_desc="reworded"),
                          revalidate=revalidate)
    row = by_id[ACCEPTED]
    assert row["edited_duration"] == reg.YES and row["edited_desc"] == reg.YES
    assert counts["edited_duration"] == 1 and counts["edited_desc"] == 1


# -------------------------------------------------------------------- schema

def test_the_verdict_vocabulary_is_closed():
    """in_mask is gone; the annotator watches over that case instead."""
    assert reg.REASONS == ["too_short", "too_long", "manual"]
    assert not hasattr(reg, "REASON_IN_MASK")
    assert "note" not in reg.COLUMNS
    assert {"edited_duration", "edited_desc"} <= set(reg.COLUMNS)
    assert "edited" not in reg.COLUMNS


def test_a_fresh_row_carries_its_original_times_in_source():
    """source_* keeps what the first run wrote, so a later shift stays visible."""
    entry = reg.row(REJECTED, "s01e01", "dev", "f.mp4", 10.0, 12.0, "x")
    assert entry["source_start"] == 10.0 and entry["source_end"] == 12.0
    reg.update([entry], {"s01e01": [event(14.0, 16.0)]},
               {"s01e01": "dev"}, {"s01e01": "f.mp4"})
    assert entry["source_start"] == 10.0 and float(entry["start"]) == 14.0
