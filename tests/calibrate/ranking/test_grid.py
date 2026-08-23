from __future__ import annotations

from collections.abc import Callable

import pytest

from optisample.calibrate.ranking import RankingGrid, widened_encodings
from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import UNLOOPED, EncodingParams

SR = 44_100
_LADDER = (8_000, 11_025, 16_000, 22_050)
_SETTLED_RATE = 16_000
_SETTLED_DEPTH = 16


@pytest.fixture
def ladder_sweep(sweep: Callable[..., SweepConfig]) -> SweepConfig:
    """A sweep whose rate ladder is short enough for a test to name every rung it holds."""
    return sweep(rates=_LADDER, depths=(_SETTLED_DEPTH,), compress=True)


@pytest.fixture
def swept() -> tuple[EncodingParams, ...]:
    """What the sweep offers one clip: the played span, then the two loops the stage settled for it."""
    span = EncodingParams(target_rate=_SETTLED_RATE, depth_bits=_SETTLED_DEPTH, trim_s=2.0, loop_index=UNLOOPED)
    return (span, *(EncodingParams(**{**vars(span), "loop_index": index}) for index in range(2)))


def test_the_sweeps_own_encodings_lead_the_set(
    swept: tuple[EncodingParams, ...],
    ladder_sweep: SweepConfig,
) -> None:
    widened = widened_encodings(swept, RankingGrid(depths=(16, 8), rate_steps=1), ladder_sweep, sample_rate=SR)

    assert widened[: len(swept)] == swept


def test_a_depth_the_grid_names_joins_the_set_at_the_settled_rate(
    swept: tuple[EncodingParams, ...],
    ladder_sweep: SweepConfig,
) -> None:
    widened = widened_encodings(swept, RankingGrid(depths=(8,), rate_steps=0), ladder_sweep, sample_rate=SR)

    added = [params for params in widened if params not in swept]
    assert {(params.target_rate, params.depth_bits) for params in added} == {(_SETTLED_RATE, 8)}


def test_a_widened_depth_is_offered_at_both_settings_of_the_compressor(
    swept: tuple[EncodingParams, ...],
    ladder_sweep: SweepConfig,
) -> None:
    widened = widened_encodings(swept, RankingGrid(depths=(16, 8), rate_steps=0), ladder_sweep, sample_rate=SR)

    added = {(params.depth_bits, params.compress) for params in widened if params not in swept}
    assert added == {(8, True), (8, False)}


def test_the_rungs_read_are_the_ones_the_ladder_holds_below_the_settled_rate(
    swept: tuple[EncodingParams, ...],
    ladder_sweep: SweepConfig,
) -> None:
    widened = widened_encodings(swept, RankingGrid(depths=(16,), rate_steps=2), ladder_sweep, sample_rate=SR)

    assert sorted({params.target_rate for params in widened}) == [8_000, 11_025, 16_000]


def test_a_clip_is_read_at_its_own_rate_alone_where_the_grid_steps_nowhere(
    swept: tuple[EncodingParams, ...],
    ladder_sweep: SweepConfig,
) -> None:
    widened = widened_encodings(swept, RankingGrid(depths=(16,), rate_steps=0), ladder_sweep, sample_rate=SR)

    assert widened == swept


def test_every_member_stores_the_played_span_the_sweep_leads_with(
    swept: tuple[EncodingParams, ...],
    ladder_sweep: SweepConfig,
) -> None:
    widened = widened_encodings(swept, RankingGrid(depths=(16, 8), rate_steps=1), ladder_sweep, sample_rate=SR)

    added = [params for params in widened if params not in swept]
    assert all(params.loop_index is UNLOOPED and params.trim_s == swept[0].trim_s for params in added)


def test_a_set_names_each_encoding_once(
    swept: tuple[EncodingParams, ...],
    ladder_sweep: SweepConfig,
) -> None:
    widened = widened_encodings(swept, RankingGrid(depths=(16, 16), rate_steps=1), ladder_sweep, sample_rate=SR)

    assert len(set(widened)) == len(widened)
