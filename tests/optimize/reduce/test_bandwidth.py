from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig, ReduceConfig
from optisample.dsp.spectral import bandlimit
from optisample.dsp.surrogate import UNLOOPED, EncodingParams
from optisample.optimize.reduce.bandwidth import (
    ClipDemand,
    DiscardPricer,
    carried_storages,
    clip_band_hz,
    clip_reach_hz,
    discard_surcharge,
    discarded_octaves,
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
_AS_PLAYED = (False,)  # one storage, so a test counting encodings measures the axis it names
_TEMPO = 125  # the clock a written curve turns its corners on, which no test here varies
_SLOW_ATTACK_S = 0.1  # an attack running several ticks, which a written curve states with room to spare
_ONE_TICK = 1.0  # the gate as it ships: a curve needs a tick of run to state an attack across
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
    tempo: int = _TEMPO


@pytest.fixture
def make_context(sweep: SweepFactory, reduce: ReduceFactory) -> Callable[..., _Context]:
    """Factory: a context over the explicit ``_RATES`` ladder, varying the knobs a test needs."""

    def _build(*, bandwidth: dict[str, object] | None = None, **grid: object) -> _Context:
        return _Context(
            sample_rate=SR,
            sweep=sweep(rates=_RATES, **{"carriers": _AS_PLAYED, **grid}),
            bandwidth=reduce(bandwidth=bandwidth or {}).bandwidth,
        )

    return _build


def tone(freq: float, amplitude: float = 0.5) -> NDArray[np.float64]:
    """A pure tone filling the trim window, whose energy sits in a single spectral bin."""
    times = np.arange(_FRAMES, dtype=np.float64) / SR
    return np.asarray(amplitude * np.sin(2.0 * np.pi * freq * times), dtype=np.float64)


def stored_as(target_rate: int) -> EncodingParams:
    """One encoding of the trim window at ``target_rate``, which is all the pricer reads of it."""
    return EncodingParams(target_rate=target_rate, depth_bits=_DEEP_DEPTH, trim_s=_TRIM_S)


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


def test_the_depths_a_run_prices_reach_the_format_it_settles(make_context: Callable[..., _Context]) -> None:
    """The reduction settles the rate alone; which grid a sample lands on is the sweep's to price."""
    context = make_context(depths=(_SHALLOW_DEPTH, _DEEP_DEPTH))
    assert stored_format(broadband(), UNTRANSPOSED, context).depths == (_SHALLOW_DEPTH, _DEEP_DEPTH)


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
    context = make_context(depths=(depth,), compress=True)
    stored = stored_format(broadband(), UNTRANSPOSED, context)
    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=0)
    assert any(params.compress for params in offered) is compressed


def test_a_run_asking_for_no_compression_stores_every_depth_as_recorded(
    make_context: Callable[..., _Context],
) -> None:
    context = make_context(depths=(_SHALLOW_DEPTH,), compress=False)
    stored = stored_format(broadband(), UNTRANSPOSED, context)
    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=0)
    assert not any(params.compress for params in offered)


# --- measuring a band once, settling many demands from it ---------------------------------------------


def test_one_measured_band_settles_every_demand_on_the_clip(make_context: Callable[..., _Context]) -> None:
    """What a demand adds -- the transpose it plays at -- is arithmetic over a band already measured."""
    context = make_context()
    clip = broadband()
    band = clip_band_hz(clip, _TRIM_S, SR, context.bandwidth)
    for demand in (UNTRANSPOSED, AN_OCTAVE_UP):
        storages = carried_storages(clip, context)
        assert format_from_band(band, demand, context, storages) == stored_format(clip, demand, context)


def test_the_same_clip_earns_the_same_format_every_time(make_context: Callable[..., _Context]) -> None:
    """The band is read off the recording alone, so one clip's format says nothing about the next."""
    context = make_context()
    clip = broadband()
    assert stored_format(clip, UNTRANSPOSED, context) == stored_format(clip, UNTRANSPOSED, context)


# --- what the sweep is offered ------------------------------------------------------------------------


def test_every_stored_span_is_offered_at_the_one_settled_format(make_context: Callable[..., _Context]) -> None:
    """The rate is settled before the sweep, so what a span varies is how far the sample carries on."""
    context = make_context(depths=(_DEEP_DEPTH,), compress=False)
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=3)

    assert [params.loop_index for params in offered] == [UNLOOPED, 0, 1, 2]
    assert {params.target_rate for params in offered} == {stored.target_rate}


def test_both_grids_are_offered_where_a_run_prices_both(make_context: Callable[..., _Context]) -> None:
    """A shallow copy costs half the frames of a deep one, so which grid to store on is a trade to price."""
    context = make_context(depths=(_DEEP_DEPTH, _SHALLOW_DEPTH), compress=False)
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=1)

    assert [params.depth_bits for params in offered] == [_DEEP_DEPTH, _SHALLOW_DEPTH] * 2


def _risen(clip: NDArray[np.float64], attack_s: float = _SLOW_ATTACK_S) -> NDArray[np.float64]:
    """``clip`` ramped in over ``attack_s``, which is a level a written curve has room to state."""
    rise = min(clip.size, round(attack_s * SR))
    ramped = np.array(clip, dtype=np.float64)
    ramped[:rise] *= np.linspace(0.0, 1.0, rise)
    return ramped


def test_a_recording_a_curve_cannot_follow_is_offered_the_one_storage(
    make_context: Callable[..., _Context],
) -> None:
    """A burst at full level from its first frame gives a curve nothing it can state, so it is stored as played.

    Storing it as a carrier would come out as the very same waveform, so offering the pair would price one
    encoding twice over.
    """
    context = make_context(carriers=(True, False), min_carried_attack_ticks=_ONE_TICK)

    assert carried_storages(broadband(), context) == (False,)


def test_a_recording_that_rises_slowly_is_offered_every_storage_the_run_prices(
    make_context: Callable[..., _Context],
) -> None:
    """An attack running several ticks is a level a curve follows, so both storages are worth pricing."""
    context = make_context(carriers=(True, False), min_carried_attack_ticks=_ONE_TICK)

    assert carried_storages(_risen(broadband()), context) == (True, False)


def test_a_clip_no_curve_follows_prices_half_the_encodings_a_slower_one_does(
    make_context: Callable[..., _Context],
) -> None:
    """What narrowing the storages is for: the sweep runs one encoding per span rather than the same one twice."""
    context = make_context(
        carriers=(True, False),
        depths=(_DEEP_DEPTH,),
        compress=False,
        min_carried_attack_ticks=_ONE_TICK,
    )
    struck = stored_format(broadband(), UNTRANSPOSED, context)
    risen = stored_format(_risen(broadband()), UNTRANSPOSED, context)

    offered = stored_encodings(struck, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=1)
    both = stored_encodings(risen, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=1)

    assert len(offered) * 2 == len(both)


def test_both_storages_are_offered_where_a_run_prices_both(make_context: Callable[..., _Context]) -> None:
    """Storing timbre alone costs the bytes storing the level with it costs, so fidelity alone separates them."""
    context = make_context(depths=(_DEEP_DEPTH,), compress=False, carriers=(True, False))
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=1)

    assert [params.carrier for params in offered] == [True, False] * 2


def test_the_stored_length_reaches_every_encoding_offered(make_context: Callable[..., _Context]) -> None:
    context = make_context()
    stored = stored_format(broadband(), UNTRANSPOSED, context)
    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=2)
    assert {params.trim_s for params in offered} == {_TRIM_S}


def test_a_shallow_depth_offers_every_span_both_plain_and_compressed(
    make_context: Callable[..., _Context],
) -> None:
    """Compression is an axis the run prices, not a step it takes: both ways are offered and one is picked."""
    context = make_context(depths=(_SHALLOW_DEPTH,), compress=True)
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=2)

    assert [(params.loop_index, params.compress) for params in offered] == [
        (UNLOOPED, False),
        (UNLOOPED, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]


def test_a_run_asking_for_no_dynamics_offers_every_span_once(make_context: Callable[..., _Context]) -> None:
    """A run that turns the stage off prices no compressed encoding, however shallow its grid runs."""
    context = make_context(depths=(_SHALLOW_DEPTH,), compress=False)
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=2)

    assert [params.compress for params in offered] == [False, False, False]


def test_a_depth_deep_enough_to_carry_the_material_offers_no_compressed_encoding(
    make_context: Callable[..., _Context],
) -> None:
    """A sixteen-bit grid already sits under anything compression would protect, so it is offered plain."""
    context = make_context(depths=(_DEEP_DEPTH,), compress=True)
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=2)

    assert [params.compress for params in offered] == [False, False, False]


def test_the_trimmed_span_leads_whatever_the_depth_offers(make_context: Callable[..., _Context]) -> None:
    """Spans lead, so the trimmed one opens the list for every clip and two sweeps score in one order."""
    context = make_context(depths=(_SHALLOW_DEPTH,), compress=True)
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=3)

    assert all(params.loop_index is UNLOOPED for params in offered[:2])
    assert [params.loop_index for params in offered[2:]] == [0, 0, 1, 1, 2, 2]


def test_a_clip_with_no_settled_loop_offers_the_trimmed_span_alone(make_context: Callable[..., _Context]) -> None:
    """Nothing prices a loop for a recording the loop stage found none in, so the played span stands alone."""
    context = make_context(depths=(_DEEP_DEPTH,), compress=False)
    stored = stored_format(broadband(), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=0)

    assert [params.loop_index for params in offered] == [UNLOOPED]


# --- what a narrow rung gives up ---------------------------------------------------------------------


def test_a_rung_reaching_every_frequency_the_recording_holds_gives_up_nothing() -> None:
    assert discarded_octaves(reach_hz=3_000.0, target_rate=8_000) == 0.0


def test_a_rung_meeting_the_reach_exactly_gives_up_nothing() -> None:
    """The edge case the charge is defined at: content ending at the stored Nyquist is content kept."""
    assert discarded_octaves(reach_hz=4_000.0, target_rate=8_000) == 0.0


@pytest.mark.parametrize(
    ("reach_hz", "octaves"),
    [
        (8_000.0, 1.0),
        (16_000.0, 2.0),
        (5_656.85, 0.5),
    ],
)
def test_a_reach_past_the_stored_edge_is_charged_by_the_octaves_between_them(reach_hz: float, octaves: float) -> None:
    assert discarded_octaves(reach_hz=reach_hz, target_rate=8_000) == pytest.approx(octaves, abs=1e-4)


def test_the_surcharge_follows_the_penalty_the_run_states(reduce: ReduceFactory) -> None:
    config = reduce(bandwidth={"discard_penalty": 0.25}).bandwidth

    assert discard_surcharge(8_000.0, 8_000, config) == pytest.approx(0.25)


def test_a_run_stating_no_penalty_charges_nothing_for_the_band_it_gives_up(reduce: ReduceFactory) -> None:
    """Zero is what leaves a rung priced on the metrics alone, which is the reading without this knob."""
    config = reduce(bandwidth={"discard_penalty": 0.0}).bandwidth

    assert discard_surcharge(16_000.0, 8_000, config) == 0.0


def test_the_deeper_floor_reads_further_up_than_the_one_that_settles_the_rung(reduce: ReduceFactory) -> None:
    """The gap between the two readings is the band a settled rung is free to leave out."""
    config = reduce(bandwidth={"content_floor_db": 40.0, "discard_floor_db": 100.0}).bandwidth
    faint_partial = tone(1_000.0) + tone(8_000.0, 1.58e-4)  # ~70 dB under the fundamental, between the floors

    assert clip_band_hz(faint_partial, _TRIM_S, SR, config) == pytest.approx(1_000.0, abs=200.0)
    assert clip_reach_hz(faint_partial, _TRIM_S, SR, config) == pytest.approx(8_000.0, abs=200.0)


def test_a_pricer_reads_one_length_once_however_many_rungs_ask_about_it(reduce: ReduceFactory) -> None:
    """Every rung offered for a span prices off one spectrum, which is what keeps the charge affordable."""
    config = reduce(bandwidth={"discard_penalty": 1.0}).bandwidth
    pricer = DiscardPricer(broadband(), SR, config)

    charged = [pricer.surcharge(stored_as(rate)) for rate in (4_000, 8_000, 16_000)]

    assert list(pricer.reaches) == [_TRIM_S]
    assert charged[0] > charged[1] > charged[2]


def test_a_pricer_stating_no_penalty_reads_no_spectrum_at_all(reduce: ReduceFactory) -> None:
    pricer = DiscardPricer(broadband(), SR, reduce(bandwidth={"discard_penalty": 0.0}).bandwidth)

    assert pricer.surcharge(stored_as(4_000)) == 0.0
    assert not pricer.reaches


# --- the rungs a span is offered at ------------------------------------------------------------------


def test_no_headroom_offers_the_rung_the_recording_asked_for_alone(make_context: Callable[..., _Context]) -> None:
    context = make_context(rate_headroom=0)
    stored = stored_format(tone(1_000.0), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=0)

    assert {params.target_rate for params in offered} == {stored.target_rate}


def test_headroom_offers_each_span_at_the_rungs_above_the_settled_one(make_context: Callable[..., _Context]) -> None:
    """Offering the wider rungs is what puts band among the things the allocation's bytes can buy."""
    context = make_context(rate_headroom=1, depths=(_DEEP_DEPTH,), compress=False)
    stored = stored_format(tone(1_000.0), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=0)

    assert stored.target_rate == _CHEAPEST_RUNG
    assert [params.target_rate for params in offered] == [4_000, 8_000]


def test_headroom_reaching_past_the_ladder_stops_at_the_recording_itself(
    make_context: Callable[..., _Context],
) -> None:
    """The recording's own rate joins the ladder as its top rung, so a generous headroom settles there."""
    context = make_context(rate_headroom=99, depths=(_DEEP_DEPTH,), compress=False)
    stored = stored_format(tone(1_000.0), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=0)

    assert [params.target_rate for params in offered] == [*_RATES[::-1], SR]


def test_the_settled_rung_leads_every_span_it_is_offered_for(make_context: Callable[..., _Context]) -> None:
    """Spans lead and the settled rung opens each, so two sweeps still score in one order."""
    context = make_context(rate_headroom=1, depths=(_DEEP_DEPTH,), compress=False)
    stored = stored_format(tone(1_000.0), UNTRANSPOSED, context)

    offered = stored_encodings(stored, context.sweep, sample_rate=SR, trim_s=_TRIM_S, loops=1)

    assert [(params.loop_index, params.target_rate) for params in offered] == [
        (UNLOOPED, 4_000),
        (UNLOOPED, 8_000),
        (0, 4_000),
        (0, 8_000),
    ]
