from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.codec import EncodeConfig
from optisample.config.optimize import TRIMMED_ONLY, SweepConfig
from optisample.config.reduce import BandwidthConfig, ReduceConfig
from optisample.dsp.spectral import bandlimit
from optisample.dsp.surrogate import EncodingParams
from optisample.metrics import CompositeFidelity
from optisample.optimize.operating_points import sweep_param_grid
from optisample.optimize.reduce.bandwidth import (
    ClipDemand,
    candidate_params,
    narrowed_params,
    proxy_grid,
    useful_rate_hz,
)
from trackmod.module.storage import Storage

SR = 22_050
_TRIM_S = 0.5
_FRAMES = int(_TRIM_S * SR)
_OCTAVE = 12
_NO_TRANSPOSE = 0
_A_ZONE_OF_KEYS = 8
_RATES = (16_000, 8_000, 4_000)  # an explicit ladder, so a test states which rates it expects back
_GRID_RATES = len(_RATES) + 1  # the ladder, plus the clip's own rate, which every clip is also offered
_ENCODINGS_PER_RATE = 3  # one 16-bit entry and both compressions of the 8-bit one, at one loop choice
_FULL_GRID = _ENCODINGS_PER_RATE * _GRID_RATES
_RATE_PER_BANDWIDTH = 2.0  # Nyquist, which turns a content-edge tolerance into a rate tolerance
_TIGHT = 0  # a byte target no encoding can undercut, so the cheapest vertex wins
_GENEROUS = 10**6  # a byte target no encoding reaches, so the costliest vertex wins
_ANY_BUDGET = 8_000  # a sample's share, where the test asks about the band rather than the price

ReduceFactory = Callable[..., ReduceConfig]
SweepFactory = Callable[..., SweepConfig]


def untransposed(byte_target: int) -> ClipDemand:
    """A key sounding its own recording: no transpose, spending its own share of the budget."""
    return ClipDemand(trim_s=_TRIM_S, delta_semitones=_NO_TRANSPOSE, byte_target=byte_target)


def an_octave_up(byte_target: int) -> ClipDemand:
    """One recording also serving the key twelve semitones above the one it was made at."""
    return ClipDemand(trim_s=_TRIM_S, delta_semitones=_OCTAVE, byte_target=byte_target)


def a_whole_zone(byte_target: int) -> ClipDemand:
    """One recording serving a zone of keys, which pool their shares of the budget behind it."""
    return ClipDemand(
        trim_s=_TRIM_S,
        delta_semitones=_NO_TRANSPOSE,
        byte_target=byte_target * _A_ZONE_OF_KEYS,
    )


@dataclass(frozen=True)
class _Context:
    """A stand-in for the run's scoring context, carrying only what narrowing a grid reads from it."""

    sample_rate: int
    composite: CompositeFidelity
    encode: EncodeConfig
    storage: Storage
    sweep: SweepConfig
    bandwidth: BandwidthConfig


@pytest.fixture
def make_context(
    composite: CompositeFidelity,
    encode_config: EncodeConfig,
    storage: Storage,
    sweep: SweepFactory,
    reduce: ReduceFactory,
) -> Callable[..., _Context]:
    """Factory: a narrowing context over the explicit ``_RATES`` grid, varying the knobs a test needs.

    The loop axis is pinned to the trimmed sample alone unless a test asks for more, so ``_FULL_GRID``
    states the grid a test is reasoning about rather than tracking the bundled sweep.
    """

    def _build(*, candidates: int, **grid: object) -> _Context:
        return _Context(
            sample_rate=SR,
            composite=composite,
            encode=encode_config,
            storage=storage,
            sweep=sweep(rates=_RATES, **{"loop_choices": TRIMMED_ONLY, **grid}),
            bandwidth=reduce(bandwidth={"candidates": candidates}).bandwidth,
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
    useful = useful_rate_hz(tone(2_000.0), untransposed(_ANY_BUDGET), SR, config)
    assert useful == pytest.approx(4_000.0, abs=_RATE_PER_BANDWIDTH * config.content_band_hz)


def test_a_clip_carrying_nothing_asks_for_no_rate_at_all(reduce: ReduceFactory) -> None:
    silence = np.zeros(_FRAMES, dtype=np.float64)
    assert useful_rate_hz(silence, untransposed(_ANY_BUDGET), SR, reduce().bandwidth) == 0.0


def test_transposing_an_octave_up_halves_the_rate_worth_storing(reduce: ReduceFactory) -> None:
    """Transposition moves the audible ceiling: a stored band played an octave up lands an octave up."""
    config = reduce(bandwidth={"ceiling_hz": 6_000.0}).bandwidth
    bright = tone(9_000.0)  # content above the ceiling, so the ceiling is the binding bound
    assert useful_rate_hz(bright, untransposed(_ANY_BUDGET), SR, config) == pytest.approx(12_000.0)
    assert useful_rate_hz(bright, an_octave_up(_ANY_BUDGET), SR, config) == pytest.approx(6_000.0)


def test_the_ceiling_never_exceeds_the_band_the_run_measures_over(reduce: ReduceFactory) -> None:
    """A ceiling past the analysis Nyquist counts as the Nyquist, the highest frequency any score sees."""
    config = reduce(bandwidth={"ceiling_hz": 1e6}).bandwidth
    assert useful_rate_hz(broadband(), an_octave_up(_ANY_BUDGET), SR, config) == pytest.approx(SR / 2.0)


# --- the shortlist -----------------------------------------------------------------------------------


def test_a_candidate_count_reaching_the_whole_grid_returns_it_untouched(
    make_context: Callable[..., _Context],
) -> None:
    """The setting that reproduces an unnarrowed run: every encoding, in the sweep's own order."""
    context = make_context(candidates=_FULL_GRID)
    grid = tuple(sweep_param_grid(context.sweep, SR, trim_s=_TRIM_S))
    assert len(grid) == _FULL_GRID
    assert candidate_params(broadband(), untransposed(_ANY_BUDGET), context) == grid


def test_the_shortlist_is_a_subsequence_of_the_full_grid(make_context: Callable[..., _Context]) -> None:
    context = make_context(candidates=2)
    grid = list(sweep_param_grid(context.sweep, SR, trim_s=_TRIM_S))
    shortlist = list(candidate_params(broadband(), untransposed(_ANY_BUDGET), context))
    assert shortlist  # narrowing always leaves a pitch something to encode
    assert [params for params in grid if params in shortlist] == shortlist


def test_the_candidate_count_bounds_how_many_encodings_the_sweep_runs(
    make_context: Callable[..., _Context],
) -> None:
    context = make_context(candidates=2, loop_choices=1)  # the trimmed sample and one loop candidate
    assert len(tuple(sweep_param_grid(context.sweep, SR, trim_s=_TRIM_S))) == 2 * _FULL_GRID
    assert len(candidate_params(broadband(), untransposed(_ANY_BUDGET), context)) == 2


def test_rates_beyond_the_useful_bound_leave_only_their_cheapest(
    make_context: Callable[..., _Context],
) -> None:
    """A clip stopping at 1.8 kHz wants 3.6 kHz; 4000 is the one rate above that worth keeping."""
    context = make_context(candidates=_FULL_GRID - 1)
    muffled = bandlimit(broadband(), SR, 0.0, 1_800.0)
    assert stored_rates(candidate_params(muffled, untransposed(_GENEROUS), context)) == {4_000}


def test_a_generous_budget_shortlists_a_costlier_encoding_than_a_tight_one(
    make_context: Callable[..., _Context],
) -> None:
    context = make_context(candidates=1)
    clip = broadband()
    lean = candidate_params(clip, untransposed(_TIGHT), context)
    rich = candidate_params(clip, untransposed(_GENEROUS), context)
    assert max(stored_rates(rich)) > max(stored_rates(lean))


def test_a_sample_serving_a_whole_zone_may_spend_what_all_its_keys_bring(
    make_context: Callable[..., _Context],
) -> None:
    """A zone's share of the budget is every key's share in it, so one sample there buys more."""
    context = make_context(candidates=1)
    clip = broadband()
    alone = candidate_params(clip, untransposed(1_000), context)
    for_a_zone = candidate_params(clip, a_whole_zone(1_000), context)
    assert max(stored_rates(for_a_zone)) > max(stored_rates(alone))


def test_the_same_clip_earns_the_same_shortlist_every_time(make_context: Callable[..., _Context]) -> None:
    """The proxy draws fixed dither, so narrowing one grid says nothing about the next."""
    context = make_context(candidates=2)
    clip = broadband()
    demand = untransposed(_ANY_BUDGET)
    assert candidate_params(clip, demand, context) == candidate_params(clip, demand, context)


# --- pricing once, narrowing many times ---------------------------------------------------------------


def test_one_priced_grid_answers_every_demand_holding_the_clip_that_long(
    make_context: Callable[..., _Context],
) -> None:
    """What a demand adds -- its transpose and its budget -- is arithmetic over points already measured."""
    context = make_context(candidates=2)
    clip = broadband()
    priced = proxy_grid(clip, _TRIM_S, context)
    for demand in (untransposed(1_000), an_octave_up(1_000), a_whole_zone(1_000)):
        assert narrowed_params(priced, demand, context) == candidate_params(clip, demand, context)


def test_pricing_admits_every_rate_a_transposed_demand_can_still_ask_for(
    make_context: Callable[..., _Context],
) -> None:
    """A transpose only lowers the audible rate, so pricing untransposed leaves a demand nothing to add."""
    context = make_context(candidates=_FULL_GRID - 1)
    clip = broadband()
    priced = {point.params.target_rate for point in proxy_grid(clip, _TRIM_S, context).points}
    for delta_semitones in (0, 1, _OCTAVE, 2 * _OCTAVE):
        demand = ClipDemand(trim_s=_TRIM_S, delta_semitones=delta_semitones, byte_target=_GENEROUS)
        assert stored_rates(candidate_params(clip, demand, context)) <= priced


def test_a_candidate_count_reaching_the_whole_grid_settles_before_anything_is_priced(
    make_context: Callable[..., _Context],
) -> None:
    context = make_context(candidates=_FULL_GRID)
    priced = proxy_grid(broadband(), _TRIM_S, context)
    assert priced.entries == tuple(sweep_param_grid(context.sweep, SR, trim_s=_TRIM_S))
    assert priced.points == ()
