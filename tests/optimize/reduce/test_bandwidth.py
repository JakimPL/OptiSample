"""Narrowing the stored rate/depth grid before the sweep runs (``optimize/reduce/bandwidth.py``)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig, ReduceConfig
from optisample.dsp.spectral import bandlimit
from optisample.dsp.surrogate import EncodingParams
from optisample.metrics import CompositeFidelity
from optisample.optimize.operating_points import sweep_param_grid
from optisample.optimize.reduce.bandwidth import ClipDemand, candidate_params, useful_rate_hz
from trackmod.module.storage import Storage

SR = 22_050
_TRIM_S = 0.5
_FRAMES = int(_TRIM_S * SR)
_OCTAVE = 12
_ONE_KEY = 1
_UNTRANSPOSED = ClipDemand(trim_s=_TRIM_S, delta_semitones=0, key_count=_ONE_KEY)
_AN_OCTAVE_UP = ClipDemand(trim_s=_TRIM_S, delta_semitones=_OCTAVE, key_count=_ONE_KEY)
_A_WHOLE_ZONE = ClipDemand(trim_s=_TRIM_S, delta_semitones=0, key_count=8)
_RATES = (16_000, 8_000, 4_000)  # an explicit grid, so a test states which rates it expects back
_FULL_GRID = 9  # the explicit rates once at 16 bits and twice at 8, under the one configured loop setting
_RATE_PER_BANDWIDTH = 2.0  # Nyquist, which turns a content-edge tolerance into a rate tolerance
_TIGHT = 0  # a byte target no encoding can undercut, so the cheapest vertex wins
_GENEROUS = 10**6  # a byte target no encoding reaches, so the costliest vertex wins

ReduceFactory = Callable[..., ReduceConfig]
SweepFactory = Callable[..., SweepConfig]


@dataclass(frozen=True)
class _Context:
    """A stand-in for the run's scoring context, carrying only what narrowing a grid reads from it."""

    sample_rate: int
    composite: CompositeFidelity
    encode: EncodeConfig
    storage: Storage
    sweep: SweepConfig
    bandwidth: BandwidthConfig
    byte_target: int


@pytest.fixture
def make_context(
    composite: CompositeFidelity,
    encode_config: EncodeConfig,
    storage: Storage,
    sweep: SweepFactory,
    reduce: ReduceFactory,
) -> Callable[..., _Context]:
    """Factory: a narrowing context over the explicit ``_RATES`` grid, varying the knobs a test needs."""

    def _build(*, byte_target: int, candidates: int, **grid: object) -> _Context:
        return _Context(
            sample_rate=SR,
            composite=composite,
            encode=encode_config,
            storage=storage,
            sweep=sweep(rates=_RATES, **grid),
            bandwidth=reduce(bandwidth={"candidates": candidates}).bandwidth,
            byte_target=byte_target,
        )

    return _build


def tone(freq: float, amplitude: float = 0.5) -> NDArray[np.float64]:
    """A pure tone filling the trim window, whose energy sits in a single spectral bin."""
    times = np.arange(_FRAMES, dtype=np.float64) / SR
    return np.asarray(amplitude * np.sin(2.0 * np.pi * freq * times), dtype=np.float64)


def broadband(amplitude: float = 0.5) -> NDArray[np.float64]:
    """A decaying noise burst: every candidate rate keeps part of it, so nothing scores against silence."""
    generator = np.random.default_rng(0)
    envelope = np.exp(-3.0 * np.arange(_FRAMES, dtype=np.float64) / _FRAMES)
    return np.asarray(amplitude * envelope * generator.standard_normal(_FRAMES), dtype=np.float64)


def stored_rates(params: Sequence[EncodingParams]) -> set[int]:
    return {item.target_rate for item in params}


# --- the bandwidth bound -----------------------------------------------------------------------------


def test_a_clip_needs_twice_the_rate_of_the_band_it_occupies(reduce: ReduceFactory) -> None:
    config = reduce().bandwidth
    useful = useful_rate_hz(tone(2_000.0), _UNTRANSPOSED, SR, config)
    assert useful == pytest.approx(4_000.0, abs=_RATE_PER_BANDWIDTH * config.content_band_hz)


def test_a_clip_carrying_nothing_asks_for_no_rate_at_all(reduce: ReduceFactory) -> None:
    silence = np.zeros(_FRAMES, dtype=np.float64)
    assert useful_rate_hz(silence, _UNTRANSPOSED, SR, reduce().bandwidth) == 0.0


def test_transposing_an_octave_up_halves_the_rate_worth_storing(reduce: ReduceFactory) -> None:
    """Transposition moves the audible ceiling: a stored band played an octave up lands an octave up."""
    config = reduce(bandwidth={"ceiling_hz": 6_000.0}).bandwidth
    bright = tone(9_000.0)  # content above the ceiling, so the ceiling is the binding bound
    assert useful_rate_hz(bright, _UNTRANSPOSED, SR, config) == pytest.approx(12_000.0)
    assert useful_rate_hz(bright, _AN_OCTAVE_UP, SR, config) == pytest.approx(6_000.0)


def test_the_ceiling_never_exceeds_the_band_the_run_measures_over(reduce: ReduceFactory) -> None:
    """A ceiling past the analysis Nyquist counts as the Nyquist, the highest frequency any score sees."""
    config = reduce(bandwidth={"ceiling_hz": 1e6}).bandwidth
    assert useful_rate_hz(broadband(), _AN_OCTAVE_UP, SR, config) == pytest.approx(SR / 2.0)


# --- the shortlist -----------------------------------------------------------------------------------


def test_a_candidate_count_reaching_the_whole_grid_returns_it_untouched(
    make_context: Callable[..., _Context],
) -> None:
    """The setting that reproduces an unnarrowed run: every encoding, in the sweep's own order."""
    context = make_context(byte_target=8_000, candidates=_FULL_GRID)
    grid = tuple(sweep_param_grid(context.sweep, SR, trim_s=_TRIM_S))
    assert len(grid) == _FULL_GRID
    assert candidate_params(broadband(), _UNTRANSPOSED, context) == grid


def test_the_shortlist_is_a_subsequence_of_the_full_grid(make_context: Callable[..., _Context]) -> None:
    context = make_context(byte_target=8_000, candidates=2)
    grid = list(sweep_param_grid(context.sweep, SR, trim_s=_TRIM_S))
    shortlist = list(candidate_params(broadband(), _UNTRANSPOSED, context))
    assert shortlist  # narrowing always leaves a pitch something to encode
    assert [params for params in grid if params in shortlist] == shortlist


def test_the_candidate_count_bounds_how_many_encodings_the_sweep_runs(
    make_context: Callable[..., _Context],
) -> None:
    context = make_context(byte_target=8_000, candidates=2, loops=(True, False))
    assert len(tuple(sweep_param_grid(context.sweep, SR, trim_s=_TRIM_S))) == 2 * _FULL_GRID
    assert len(candidate_params(broadband(), _UNTRANSPOSED, context)) == 2


def test_rates_beyond_the_useful_bound_leave_only_their_cheapest(
    make_context: Callable[..., _Context],
) -> None:
    """A clip stopping at 1.8 kHz wants 3.6 kHz; 4000 is the one rate above that worth keeping."""
    context = make_context(byte_target=_GENEROUS, candidates=_FULL_GRID - 1)
    muffled = bandlimit(broadband(), SR, 0.0, 1_800.0)
    assert stored_rates(candidate_params(muffled, _UNTRANSPOSED, context)) == {4_000}


def test_a_generous_budget_shortlists_a_costlier_encoding_than_a_tight_one(
    make_context: Callable[..., _Context],
) -> None:
    clip = broadband()
    lean = candidate_params(clip, _UNTRANSPOSED, make_context(byte_target=_TIGHT, candidates=1))
    rich = candidate_params(clip, _UNTRANSPOSED, make_context(byte_target=_GENEROUS, candidates=1))
    assert max(stored_rates(rich)) > max(stored_rates(lean))


def test_a_sample_serving_a_whole_zone_may_spend_what_all_its_keys_bring(
    make_context: Callable[..., _Context],
) -> None:
    """A zone's share of the budget is every key's share in it, so one sample there buys more."""
    context = make_context(byte_target=1_000, candidates=1)
    clip = broadband()
    alone = candidate_params(clip, _UNTRANSPOSED, context)
    for_a_zone = candidate_params(clip, _A_WHOLE_ZONE, context)
    assert max(stored_rates(for_a_zone)) > max(stored_rates(alone))


def test_the_same_clip_earns_the_same_shortlist_every_time(make_context: Callable[..., _Context]) -> None:
    """The proxy draws fixed dither, so narrowing one grid says nothing about the next."""
    context = make_context(byte_target=8_000, candidates=2)
    clip = broadband()
    assert candidate_params(clip, _UNTRANSPOSED, context) == candidate_params(clip, _UNTRANSPOSED, context)
