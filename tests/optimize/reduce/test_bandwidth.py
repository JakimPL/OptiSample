from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig, ReduceConfig
from optisample.dsp.spectral import bandlimit
from optisample.dsp.surrogate import UNLOOPED
from optisample.optimize.reduce.bandwidth import (
    ClipDemand,
    clip_band_hz,
    format_from_band,
    stored_encodings,
    stored_format,
    useful_rate_hz,
)

SR = 22_050
_TRIM_S = 0.5
_FRAMES = int(_TRIM_S * SR)
_OCTAVE = 12
_NO_TRANSPOSE = 0
_RATES = (16_000, 8_000, 4_000)  # an explicit ladder, so a test states which rung it expects back
_CHEAPEST_RUNG = min(_RATES)
_RATE_PER_BANDWIDTH = 2.0  # Nyquist, which turns a content-edge tolerance into a rate tolerance
_DEEP_DEPTH = 16  # bits, where the quantizer already sits below what compression would protect
_SHALLOW_DEPTH = 8  # bits, where compression buys headroom the quantizer can be heard against

ReduceFactory = Callable[..., ReduceConfig]
SweepFactory = Callable[..., SweepConfig]

UNTRANSPOSED = ClipDemand(trim_s=_TRIM_S, delta_semitones=_NO_TRANSPOSE)
AN_OCTAVE_UP = ClipDemand(trim_s=_TRIM_S, delta_semitones=_OCTAVE)


@dataclass(frozen=True)
class _Context:
    """A stand-in for the run's context, carrying only what settling a stored format reads from it."""

    sample_rate: int
    sweep: SweepConfig
    bandwidth: BandwidthConfig


@pytest.fixture
def make_context(sweep: SweepFactory, reduce: ReduceFactory) -> Callable[..., _Context]:
    """Factory: a context over the explicit ``_RATES`` ladder, varying the knobs a test needs."""

    def _build(*, bandwidth: dict[str, object] | None = None, **grid: object) -> _Context:
        return _Context(
            sample_rate=SR,
            sweep=sweep(rates=_RATES, **grid),
            bandwidth=reduce(bandwidth=bandwidth or {}).bandwidth,
        )

    return _build


def tone(freq: float, amplitude: float = 0.5) -> NDArray[np.float64]:
    """A pure tone filling the trim window, whose energy sits in a single spectral bin."""
    times = np.arange(_FRAMES, dtype=np.float64) / SR
    return np.asarray(amplitude * np.sin(2.0 * np.pi * freq * times), dtype=np.float64)


def broadband(amplitude: float = 0.5) -> NDArray[np.float64]:
    """A decaying noise burst, whose content reaches as far up as the recording's own rate holds."""
    generator = np.random.default_rng(0)
    envelope = np.exp(-3.0 * np.arange(_FRAMES, dtype=np.float64) / _FRAMES)
    return np.asarray(amplitude * envelope * generator.standard_normal(_FRAMES), dtype=np.float64)


# --- the bandwidth bound -----------------------------------------------------------------------------


def test_a_clip_needs_twice_the_rate_of_the_band_it_occupies(reduce: ReduceFactory) -> None:
    config = reduce().bandwidth
    useful = useful_rate_hz(tone(2_000.0), UNTRANSPOSED, SR, config)
    assert useful == pytest.approx(4_000.0, abs=_RATE_PER_BANDWIDTH * config.content_band_hz)


def test_a_clip_carrying_nothing_asks_for_no_rate_at_all(reduce: ReduceFactory) -> None:
    silence = np.zeros(_FRAMES, dtype=np.float64)
    assert useful_rate_hz(silence, UNTRANSPOSED, SR, reduce().bandwidth) == 0.0


def test_transposing_an_octave_up_halves_the_rate_worth_storing(reduce: ReduceFactory) -> None:
    """Transposition moves the audible ceiling: a stored band played an octave up lands an octave up."""
    config = reduce(bandwidth={"ceiling_hz": 6_000.0}).bandwidth
    bright = tone(9_000.0)  # content above the ceiling, so the ceiling is the binding bound
    assert useful_rate_hz(bright, UNTRANSPOSED, SR, config) == pytest.approx(12_000.0)
    assert useful_rate_hz(bright, AN_OCTAVE_UP, SR, config) == pytest.approx(6_000.0)


def test_the_ceiling_never_exceeds_the_band_the_run_measures_over(reduce: ReduceFactory) -> None:
    """A ceiling past the analysis Nyquist counts as the Nyquist, the highest frequency any score sees."""
    config = reduce(bandwidth={"ceiling_hz": 1e6}).bandwidth
    assert useful_rate_hz(broadband(), AN_OCTAVE_UP, SR, config) == pytest.approx(SR / 2.0)


# --- the format the reduction settles -----------------------------------------------------------------


def test_the_stored_rate_is_the_lowest_rung_carrying_the_clips_own_band(
    make_context: Callable[..., _Context],
) -> None:
    """A clip stopping at 1.8 kHz asks for 3.6 kHz, and 4000 is the cheapest rung that carries it."""
    context = make_context()
    muffled = bandlimit(broadband(), SR, 0.0, 1_800.0)
    assert stored_format(muffled, UNTRANSPOSED, context).target_rate == 4_000


def test_a_brighter_clip_is_stored_at_a_higher_rung(make_context: Callable[..., _Context]) -> None:
    context = make_context()
    dull = bandlimit(broadband(), SR, 0.0, 1_800.0)
    bright = bandlimit(broadband(), SR, 0.0, 5_000.0)
    assert (
        stored_format(bright, UNTRANSPOSED, context).target_rate
        > stored_format(dull, UNTRANSPOSED, context).target_rate
    )


def test_a_clip_carrying_nothing_is_stored_at_the_cheapest_rung(make_context: Callable[..., _Context]) -> None:
    """Silence asks for no band at all, so the ladder's lowest rung is what carries it."""
    silence = np.zeros(_FRAMES, dtype=np.float64)
    assert stored_format(silence, UNTRANSPOSED, make_context()).target_rate == _CHEAPEST_RUNG


def test_a_band_reaching_past_every_rung_is_stored_as_recorded(make_context: Callable[..., _Context]) -> None:
    """The recording's own rate joins the ladder as its top rung, so a wide band is stored untouched."""
    context = make_context(bandwidth={"ceiling_hz": 1e6})
    assert stored_format(broadband(), UNTRANSPOSED, context).target_rate == SR


def test_a_sample_transposed_up_is_stored_at_a_lower_rung(make_context: Callable[..., _Context]) -> None:
    """Playing a sample an octave up lifts its stored band, so less of it stays under the ceiling."""
    context = make_context(bandwidth={"ceiling_hz": 6_000.0})
    bright = tone(9_000.0)
    assert (
        stored_format(bright, AN_OCTAVE_UP, context).target_rate
        < stored_format(bright, UNTRANSPOSED, context).target_rate
    )


def test_the_depth_is_the_one_the_run_stores_every_sample_at(make_context: Callable[..., _Context]) -> None:
    context = make_context(depth=_SHALLOW_DEPTH)
    assert stored_format(broadband(), UNTRANSPOSED, context).depth_bits == _SHALLOW_DEPTH


@pytest.mark.parametrize(
    ("depth", "compressed"),
    [
        pytest.param(_SHALLOW_DEPTH, True, id="shallow depth hears the headroom compression buys"),
        pytest.param(_DEEP_DEPTH, False, id="deep depth keeps the waveform as recorded"),
    ],
)
def test_compression_reaches_the_depths_that_stand_to_win_by_it(
    make_context: Callable[..., _Context], depth: int, compressed: bool
) -> None:
    context = make_context(depth=depth, compress=True)
    assert stored_format(broadband(), UNTRANSPOSED, context).compress is compressed


def test_a_run_asking_for_no_compression_stores_every_depth_as_recorded(
    make_context: Callable[..., _Context],
) -> None:
    context = make_context(depth=_SHALLOW_DEPTH, compress=False)
    assert stored_format(broadband(), UNTRANSPOSED, context).compress is False


# --- measuring a band once, settling many demands from it ---------------------------------------------


def test_one_measured_band_settles_every_demand_on_the_clip(make_context: Callable[..., _Context]) -> None:
    """What a demand adds -- the transpose it plays at -- is arithmetic over a band already measured."""
    context = make_context()
    clip = broadband()
    band = clip_band_hz(clip, _TRIM_S, SR, context.bandwidth)
    for demand in (UNTRANSPOSED, AN_OCTAVE_UP):
        assert format_from_band(band, demand, context) == stored_format(clip, demand, context)


def test_the_same_clip_earns_the_same_format_every_time(make_context: Callable[..., _Context]) -> None:
    """The band is read off the recording alone, so one clip's format says nothing about the next."""
    context = make_context()
    clip = broadband()
    assert stored_format(clip, UNTRANSPOSED, context) == stored_format(clip, UNTRANSPOSED, context)


# --- what the sweep is offered ------------------------------------------------------------------------


def test_every_stored_span_is_offered_at_the_one_settled_format(make_context: Callable[..., _Context]) -> None:
    """The format is settled before the sweep, so what the sweep prices is how far the sample carries on."""
    context = make_context()
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, trim_s=_TRIM_S, loops=3)

    assert [params.loop_index for params in offered] == [UNLOOPED, 0, 1, 2]
    assert {(params.target_rate, params.depth_bits, params.compress) for params in offered} == {
        (stored.target_rate, stored.depth_bits, stored.compress)
    }


def test_the_stored_length_reaches_every_encoding_offered(make_context: Callable[..., _Context]) -> None:
    context = make_context()
    stored = stored_format(broadband(), UNTRANSPOSED, context)
    offered = stored_encodings(stored, context.sweep, trim_s=_TRIM_S, loops=2)
    assert {params.trim_s for params in offered} == {_TRIM_S}


def test_a_clip_with_no_settled_loop_offers_the_trimmed_span_alone(make_context: Callable[..., _Context]) -> None:
    """Nothing prices a loop for a recording the loop stage found none in, so the played span stands alone."""
    context = make_context()
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, trim_s=_TRIM_S, loops=0)

    assert [params.loop_index for params in offered] == [UNLOOPED]
