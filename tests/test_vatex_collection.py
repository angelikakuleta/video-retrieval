"""The VATEX collection in the runner: a clip is an episode with one fragment.

Everything the runner does to a series it has to do to a clip, without a ranges
file, without a segmentation section and without decoding a few thousand files
to write one row each.
"""

import json

import pytest

from src.data import datasets
from src.data.queries import episode_of
from src.data.ranges import load_ranges, vatex_ranges, vatex_range_files
from src.segmentation.build import WHOLE_CLIP, ensure_segments
from src.utils.config import dump_resolved, load_experiment
from src.utils.queries import Query

CLIPS = ["--07WQ2iBlw_000001_000011", "zzzTESTCLIP_000020_000030"]


@pytest.fixture
def vatex(tmp_path, monkeypatch):
    """A VATEX dataset whose query files and caches live under tmp_path."""
    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    folder = tmp_path / "data" / "annotations" / "vatex"
    folder.mkdir(parents=True)
    for split, clips in (("dev", CLIPS[:1]), ("test", CLIPS[1:])):
        with (folder / f"vatex_queries_{split}.jsonl").open("w", encoding="utf-8") as f:
            for clip in clips:
                f.write(json.dumps({
                    "desc_id": abs(hash(clip)) % 10_000, "desc": "a man walks",
                    "vid_name": clip, "ts": [0.0, 10.0], "source": "vatex",
                    "requirements": [], "complexity": None, "identities": [],
                    "event_id": clip}) + "\n")
    return tmp_path


# ---------------------------------------------- ranges without a ranges file
def test_the_ranges_are_synthesized_from_the_query_files(vatex):
    """One row per clip: the clip IS its own range, so no file says it twice."""
    ranges = load_ranges("vatex")

    assert set(ranges) == set(CLIPS)
    assert ranges[CLIPS[0]] == {"split": "dev", "spans": [(0.0, 10.0)],
                                "video_file": f"dev/{CLIPS[0]}.mp4"}
    # each part under a subdirectory named after it: the test clips were moved
    # into test/ after section 03 argued against moving them, and the code
    # follows the disk (src/data/ranges.py::VATEX_SUBDIR)
    assert ranges[CLIPS[1]]["video_file"] == f"test/{CLIPS[1]}.mp4"


def test_a_part_whose_query_file_is_absent_is_simply_not_there(vatex):
    """The development clips arrive while the collection is already usable."""
    (vatex / "data" / "annotations" / "vatex" / "vatex_queries_dev.jsonl").unlink()
    ranges = vatex_ranges()

    assert set(ranges) == {CLIPS[1]}
    assert [p.name for p in vatex_range_files()] == ["vatex_queries_test.jsonl"]


def test_the_query_files_stand_in_for_the_ranges_file(vatex):
    assert [p.name for p in vatex_range_files()] == ["vatex_queries_dev.jsonl",
                                                     "vatex_queries_test.jsonl"]


def test_the_clips_of_each_part_sit_under_their_own_root(vatex):
    """Downloaded excerpts stay in data/raw where the acquisition put them."""
    dev = datasets.video_path("vatex", f"dev/{CLIPS[0]}.mp4")
    test = datasets.video_path("vatex", f"{CLIPS[1]}.mp4")

    assert dev.relative_to(vatex).parts[:3] == ("data", "raw", "vatex")
    assert test.parent == dev.parent.parent
    assert datasets.video_path("office", "office_s01e01.mp4"
                               ).relative_to(vatex).parts[1] == "processed"


# ----------------------------------------- segmentation that decodes nothing
def test_whole_clip_segments_without_opening_a_single_file(vatex, monkeypatch):
    """2489 clips would mean 2489 decoder passes to write one row each."""
    from src.segmentation import decode

    def explode(*args, **kwargs):                      # pragma: no cover
        raise AssertionError("whole_clip must not decode anything")

    monkeypatch.setattr(decode, "FrameStream", explode)
    rows = ensure_segments("vatex", {WHOLE_CLIP: None}, log=lambda *_: None)[WHOLE_CLIP]

    assert len(rows) == len(CLIPS)                     # exactly one fragment per clip
    row = next(r for r in rows if r["episode"] == CLIPS[0])
    assert (row["start"], row["end"]) == (0.0, 10.0)   # the whole span, uncorrected
    assert row["split"] == "dev"


def test_the_second_pass_recomputes_nothing(vatex):
    first = ensure_segments("vatex", {WHOLE_CLIP: None}, log=lambda *_: None)[WHOLE_CLIP]
    again = ensure_segments("vatex", {WHOLE_CLIP: None}, log=lambda *_: None)[WHOLE_CLIP]
    assert first == again


def test_whole_clip_cannot_share_a_run_with_a_cut_strategy(vatex):
    with pytest.raises(ValueError, match="only strategy"):
        ensure_segments("vatex", {WHOLE_CLIP: None, "fixed_window": None},
                        log=lambda *_: None)


def test_a_fragment_identifier_is_built_by_the_same_function_on_both_sides(vatex):
    """What rules out a silent zero in the relevance sets."""
    from src.evaluation.relevance import fragment_id, relevance_sets

    rows = ensure_segments("vatex", {WHOLE_CLIP: None}, log=lambda *_: None)[WHOLE_CLIP]
    query = Query(desc_id=1, desc="a man walks", vid_name=CLIPS[0], ts=(0.0, 10.0),
                  source="vatex", event_id=CLIPS[0])
    found = relevance_sets([query], [r for r in rows if r["split"] == "dev"])

    row = next(r for r in rows if r["episode"] == CLIPS[0])
    assert found[1] == {fragment_id(CLIPS[0], row["segment_id"])}


# ------------------------------------- one convention for naming a recording
def test_episode_of_follows_the_source_of_the_query():
    """Splitting a clip name at the first underscore cuts the video id in half."""
    clip = Query(desc_id=1, desc="x", vid_name=CLIPS[0], ts=(0.0, 10.0), source="vatex")
    series = Query(desc_id=2, desc="x", vid_name="tbbt_s03e02", ts=(0.0, 5.0),
                   source="tbbt")

    assert episode_of(clip) == CLIPS[0]
    assert episode_of(series) == "s03e02"


def test_the_relevance_sets_use_that_one_convention():
    """They used to carry their own copy of it, and it was wrong for VATEX."""
    import inspect

    from src.evaluation import relevance

    assert "episode_of(query)" in inspect.getsource(relevance.relevance_sets)


# --------------------------------------------------------- the configuration
def test_a_configuration_without_a_segmentation_section_still_names_its_collection():
    config = load_experiment("configs/e2a_vatex.yaml", split="dev")
    assert config.segmentation is None
    assert config.collection_strategy == WHOLE_CLIP
    assert config.experiment == "E2-A"                 # the BASE of all three datasets


@pytest.mark.parametrize("name, split", [
    ("e2a_office", "dev"), ("full_tbbt", "dev"),
    ("e2a_vatex", "dev"), ("e2c_vatex", "test"),
])
def test_a_resolved_configuration_round_trips(tmp_path, name, split):
    """What a run writes down must read back as what it ran.

    The reason the strategy is a property and not an injected section: a section
    written here would be dumped, and the validator refuses to read it back.
    """
    config = load_experiment(f"configs/{name}.yaml", split=split)
    written = dump_resolved(config, tmp_path / f"{name}.resolved.yaml")

    assert load_experiment(written).model_dump() == config.model_dump()
    if config.dataset == "vatex":
        assert "segmentation: null" in written.read_text(encoding="utf-8")


def test_only_the_vatex_file_that_scores_a_vocabulary_carries_the_threshold():
    """E2-A is the base alone and needs none; E2-C adds motion and needs one.

    Which of them has a block is not a matter of when the freeze happened but of
    what the file scores against, so this holds before and after it.
    """
    from src.runners import run
    from src.utils.frozen import load_frozen

    base = load_experiment("configs/e2a_vatex.yaml", "dev")
    assert not base.needs_matching and base.matching is None
    run._require_matching(base, "configs/e2a_vatex.yaml")   # never refused

    motion = load_experiment("configs/e2c_vatex.yaml", "dev")
    assert motion.needs_matching
    if load_frozen().matching is None:
        # before the freeze the block is absent and the runner says so
        assert motion.matching is None
        with pytest.raises(ValueError, match="matching block"):
            run._require_matching(motion, "configs/e2c_vatex.yaml")
    else:
        assert motion.matching is not None
        run._require_matching(motion, "configs/e2c_vatex.yaml")


# -------------------------------------------------------------- the K400 cut
def test_k400_membership_reads_the_split_of_its_part(tmp_path, monkeypatch):
    from src.utils import vatex as adapter

    monkeypatch.setattr(adapter, "INTERIM_DIR", tmp_path)
    (tmp_path / "vatex_split_dev.csv").write_text(
        "videoID;class_k600;leak_k400;class_in_k400_vocab\n"
        "clip_a;javelin throw;no;yes\n"
        "clip_b;bee keeping;no;no\n", encoding="utf-8-sig")

    assert adapter.k400_membership("dev") == {"clip_a": True, "clip_b": False}


def test_a_missing_split_file_says_which_command_makes_it(tmp_path, monkeypatch):
    """The development one is written while the clips are still downloading."""
    from src.utils import vatex as adapter

    monkeypatch.setattr(adapter, "INTERIM_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="vatex_leak_filter"):
        adapter.k400_membership("dev")


# ------------------------------------------------------- logging that scales
def test_a_short_loop_keeps_reporting_every_item():
    from src.utils.progress import progress

    seen = []
    report = progress(seen.append, total=6)
    for i in range(6):
        report(f"item {i}")
    assert len(seen) == 6


def test_a_long_loop_reports_every_hundredth_and_the_last():
    from src.utils.progress import progress

    seen = []
    report = progress(seen.append, total=250)
    for i in range(250):
        report(f"item {i}")
    assert len(seen) == 3                      # 100, 200 and the final one
    assert "250" in seen[-1]


def test_a_missing_split_file_stops_a_test_run_but_only_warns_on_dev(tmp_path,
                                                                    monkeypatch):
    """On dev the file is legitimately absent; on test its rows are a table.

    A test run that quietly finished without the K400 breakdown would look
    exactly like one that has it, until somebody opened metrics.json.
    """
    from src.runners import run
    from src.utils import vatex as adapter

    monkeypatch.setattr(adapter, "INTERIM_DIR", tmp_path)

    class Config:
        dataset = "vatex"
        split = "dev"

        class evaluation:
            report_by = ["kinetics_vocab"]

    messages = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: messages.append(" ".join(map(str, a))))
    assert run._k400_membership(Config) is None
    assert any("skipped" in message for message in messages)

    Config.split = "test"
    with pytest.raises(FileNotFoundError, match="vatex_leak_filter"):
        run._k400_membership(Config)

