"""The relevance rule from chapter 4."""

from src.evaluation.relevance import is_relevant, relevance_sets
from src.utils.queries import Query


def test_fragment_containing_the_midpoint_is_relevant():
    assert is_relevant((10, 20), (14, 16))          # event inside, midpoint 15
    # midpoint of (18, 30) is 24 -> outside; overlap = 2 < 5 -> NOT relevant
    assert not is_relevant((10, 20), (18, 30))


def test_half_overlap_rule():
    # fragment 10 s, event covers exactly half of it
    assert is_relevant((10, 20), (15, 40))
    # covers a bit less than half and midpoint outside
    assert not is_relevant((10, 20), (15.1, 40))


def test_event_spanning_many_fragments():
    fragments = [(0, 10), (10, 20), (20, 30)]
    event = (5, 25)   # midpoint 15
    flags = [is_relevant(f, event) for f in fragments]
    assert flags == [True, True, True]  # 5s overlap = half of each 10 s fragment


# --------------------------------------- occurrences of one event are merged

def _segments(episode="s01e01", count=6, length=10.0):
    return [{"episode": episode, "segment_id": i + 1,
             "start": i * length, "end": (i + 1) * length} for i in range(count)]


def _query(desc_id, ts, event_id):
    return Query(desc_id=desc_id, desc="a query", vid_name="tbbt_s01e01",
                 ts=ts, source="tbbt", event_id=event_id)


def test_records_of_one_event_share_the_union_of_their_fragments():
    queries = [_query(1, (5.0, 8.0), "e1"), _query(2, (45.0, 48.0), "e1")]
    sets = relevance_sets(queries, _segments())
    assert sets[1] == sets[2] == {"s01e01_001", "s01e01_005"}


def test_records_of_different_events_stay_separate():
    queries = [_query(1, (5.0, 8.0), "e1"), _query(2, (45.0, 48.0), "e2")]
    sets = relevance_sets(queries, _segments())
    assert sets[1] == {"s01e01_001"}
    assert sets[2] == {"s01e01_005"}


def test_a_record_without_an_event_id_stands_for_an_event_of_its_own():
    queries = [_query(1, (5.0, 8.0), ""), _query(2, (45.0, 48.0), "")]
    sets = relevance_sets(queries, _segments())
    assert sets[1] == {"s01e01_001"}
    assert sets[2] == {"s01e01_005"}


def test_an_event_matching_no_fragment_gets_an_empty_set():
    queries = [_query(1, (500.0, 505.0), "e1")]
    assert relevance_sets(queries, _segments())[1] == set()
