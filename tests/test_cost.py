"""The cost measurement's crop components, and the one that had no row.

The region signal has no model of its own -- it runs the encoder of the base
representation over the face crops -- and that is why it went unmeasured: the
encoder already had a row, so the component looked covered. It is not. That row
walks the sampling grid of the recording and this one walks every accepted face
crop of the corpus, and an hour of material holds a different number of each.

Nothing here loads a model. The encoder arrives through a stub, so what is under
test is the accounting: which unit the row is paid in, where the encoder's name
comes from, and which rows are skipped when there is nothing to walk.
"""

import time

import numpy as np
import pytest

from src.features.faces import CROP_SIZE
from src.measurement import cost


def crops(count: int) -> np.ndarray:
    return np.zeros((count, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)


class StubEncoder:
    """Records the size of every batch it was handed.

    ``delay`` buys the timing something to measure: a stub that returns instantly
    makes ``seconds_per_unit`` round to zero and the row would look empty for a
    reason that has nothing to do with the component.
    """

    def __init__(self, delay: float = 0.0):
        self.batches = []
        self.delay = delay

    def encode_images(self, images):
        self.batches.append(len(images))
        if self.delay:
            time.sleep(self.delay)
        return np.zeros((len(images), 4), dtype=np.float32)


@pytest.fixture
def stub(monkeypatch):
    """`encoders.build` returns the stub, and remembers what it was asked for."""
    from src.features import encoders

    encoder = StubEncoder(delay=0.005)
    asked = []
    monkeypatch.setattr(encoders, "build",
                        lambda name: (asked.append(name), encoder)[1])
    encoder.asked = asked
    return encoder


# --------------------------------- where the row sits and what it is paid in
def test_the_region_row_exists_and_is_paid_per_crop(stub):
    model, run, count, unit = cost._regions(crops(200), count=100)
    assert (model, count, unit) == (stub, 100, "crop")


def test_the_crops_go_through_in_the_batch_production_uses(stub):
    """The peak memory has to be the one `ensure_regions` reaches, not a nicer one."""
    from src.features.regions import BATCH

    _, run, _, _ = cost._regions(crops(200), count=100)
    run()
    assert stub.batches == [BATCH] * (100 // BATCH) + [100 % BATCH]
    assert sum(stub.batches) == 100


def test_the_component_is_listed_where_chapter_six_lists_the_row():
    """Between ArcFace and HSEmotion, which is the order of `tab:koszt-wyniki`."""
    order = list(cost.COMPONENTS)
    assert order.index("arcface") + 1 == order.index("regions")
    assert order.index("regions") + 1 == order.index("hsemotion")
    assert set(cost.CROP_COMPONENTS) == {"arcface", "regions", "hsemotion"}


# ----------------------------------- the encoder is a verdict, not a literal
def test_the_encoder_is_the_one_configs_frozen_yaml_names(stub, monkeypatch):
    """E2 decides the base representation; this module must not decide it again."""
    monkeypatch.setattr(cost, "base_representation", lambda: "openclip_vit_b32")
    cost._regions(crops(4), count=4)
    assert stub.asked == ["openclip_vit_b32"]


def test_the_verdict_is_read_from_frozen_and_not_from_a_constant():
    from src.utils.frozen import load_frozen

    assert cost.base_representation() == load_frozen().scene_embedding
    # and it is not one of the two E2 candidates written out here: those rows
    # measure the candidates on FRAMES, which is a different question
    source = cost._regions.__code__.co_consts
    assert not any(isinstance(c, str) and c.startswith("openclip_") for c in source)


# ----------------------------------- the whole row, through extraction_costs
@pytest.fixture
def counted(monkeypatch):
    """Units per hour fixed, so `min_per_hour` follows from the timing alone."""
    monkeypatch.setattr(cost, "faces_per_hour", lambda dataset, split: 6000.0)
    monkeypatch.setattr(cost, "REPEATS", 2)
    monkeypatch.setattr(cost, "WARMUP", 1)


def test_extraction_costs_returns_a_crop_row_for_the_regions(stub, counted):
    out = cost.extraction_costs("office", "dev", frames=[], crops=crops(8),
                                only=["regions"], log=lambda *_: None)
    assert len(out) == 1
    row = out[0]
    assert (row["key"], row["unit"], row["component"]) == ("regions", "crop",
                                                           "face regions")
    assert row["batch"] == 8                    # crops the sample covered
    assert row["seconds_per_unit"] > 0
    # 6000 crops an hour of the fixture, so the row is minutes per hour of
    # material and not seconds per crop dressed up
    assert row["min_per_hour"] == pytest.approx(
        cost.per_hour(row["seconds_per_unit"], 6000.0), abs=0.01)
    assert row["min_per_hour"] > 0


def test_an_empty_crop_buffer_skips_every_component_paid_per_crop(stub, counted):
    said = []
    out = cost.extraction_costs("office", "dev", frames=[],
                                crops=crops(0), only=list(cost.CROP_COMPONENTS),
                                log=said.append)
    assert out == []
    assert sum("no face crops cached" in line for line in said) == 3


def test_without_the_e2_verdict_the_region_row_is_skipped_not_invented(
        stub, counted, monkeypatch):
    """A row this module cannot honestly fill is a row it leaves out.

    `encoders.build(None)` would raise somewhere far from here, and the message
    would be about a model name rather than about a verdict nobody has written
    down yet.
    """
    monkeypatch.setattr(cost, "base_representation", lambda: None)
    said = []
    out = cost.extraction_costs("office", "dev", frames=[], crops=crops(8),
                                only=["regions"], log=said.append)
    assert out == []
    assert any("frozen.yaml" in line for line in said)
    assert stub.asked == []
