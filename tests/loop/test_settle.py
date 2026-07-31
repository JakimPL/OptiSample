from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.loop import LoopConfig
from optisample.loop.settle import Gate, _ladder, settle_loop

SR = 22_050
_FREQ = 220.0
_HELD_S = 3.0  # long enough for the geometry to lay out several candidates and still leave a tail

LoopFactory = Callable[..., LoopConfig]

_OPEN: Final = 1e9  # a gate nothing measures past, which leaves the one under test to decide alone
_SHUT: Final = 1e-9  # a gate nothing clears, which is how a whole ladder is made to report

_WIDE_OPEN: Final = {"max_seam_step": _OPEN, "max_level_drift_db": _OPEN, "max_spectral_distance_db": _OPEN}
_SEAM_SHUT: Final = {**_WIDE_OPEN, "max_seam_step": _SHUT}


def _tone(duration_s: float = _HELD_S, freq: float = _FREQ) -> NDArray[np.float64]:
    """A steady harmonic tone, which the geometry finds a period in and can loop anywhere inside."""
    times = np.arange(int(duration_s * SR), dtype=np.float64) / SR
    return np.asarray(0.7 * np.sin(2.0 * np.pi * freq * times) + 0.2 * np.sin(4.0 * np.pi * freq * times))


def _decaying(duration_s: float = _HELD_S) -> NDArray[np.float64]:
    """A struck note: one pitch throughout, under an amplitude that falls away as it rings."""
    tone = _tone(duration_s)
    return np.asarray(np.exp(-np.arange(tone.size, dtype=np.float64) / (0.8 * SR)) * tone)


# --- what the ladder offers -----------------------------------------------------------------------


def test_every_candidate_clearing_the_gates_is_offered(loop: LoopFactory) -> None:
    """Gates open, so the whole ladder comes back as a frontier and none of it is turned down."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert len(settlement.offered) > 1
    assert settlement.rejected == ()


def test_the_offers_run_from_the_cheapest_stored_span_upward(loop: LoopFactory) -> None:
    """Storing a loop keeps everything up to its end, so ordering by that end orders the frontier by cost."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    ends = [stored.loop.end for stored in settlement.offered]
    assert ends == sorted(ends)
    assert settlement.cheapest is settlement.offered[0]


def test_a_longer_offer_stores_more_of_the_recording_than_a_cheaper_one(loop: LoopFactory) -> None:
    """The frontier is a rate axis, so the offers have to differ in what they cost to be worth pricing."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    ends = {stored.loop.end for stored in settlement.offered}
    assert len(ends) > 1
    assert max(ends) > 2 * min(ends)


def test_a_loop_is_offered_no_earlier_than_the_material_a_wrap_blends_into(loop: LoopFactory) -> None:
    """A tone holds one sound throughout, so what places its window is the blend its wrap reaches back for."""
    config = loop(quality=_WIDE_OPEN)
    settlement = settle_loop(_tone(), SR, config, root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.loops
    earliest = round(config.seam.min_fade_s * SR) - round(SR / _FREQ)
    assert all(stored.loop.start >= earliest for stored in settlement.offered)


def test_every_candidate_the_ladder_turns_down_names_the_gate_it_fell_outside(loop: LoopFactory) -> None:
    """A gate nothing can clear leaves the whole ladder reported, so the reason is readable per candidate."""
    settlement = settle_loop(_tone(), SR, loop(quality=_SEAM_SHUT), root_hz=_FREQ, search_s=_HELD_S)

    assert not settlement.loops
    assert settlement.cheapest is None
    assert settlement.rejected != ()
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.SEAM}


def test_a_loop_holding_a_timbre_the_material_moves_away_from_is_turned_down(loop: LoopFactory) -> None:
    """A recording that brightens halfway leaves its early loops holding a timbre the rest no longer has."""
    brightened = np.concatenate([_tone(1.5), _tone(1.5, freq=5 * _FREQ)])
    config = loop(quality={**_WIDE_OPEN, "max_spectral_distance_db": _SHUT})

    settlement = settle_loop(brightened, SR, config, root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.rejected != ()
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.TIMBRE}
    assert settlement.loops
    # every offer that survives sits in the brightened stretch, which is the timbre the rest of the note has
    assert all(stored.loop.start > round(1.5 * SR) for stored in settlement.offered)


def test_every_rung_the_ladder_lays_out_is_either_offered_or_named(loop: LoopFactory) -> None:
    """Each candidate is measured, so a run accounts for all of them and none is skipped unmeasured."""
    tight = loop(quality={**_WIDE_OPEN, "max_seam_step": 1.0})
    settlement = settle_loop(_decaying(), SR, tight, root_hz=_FREQ, search_s=_HELD_S)

    measured = [stored.loop for stored in settlement.offered] + [turned.loop for turned in settlement.rejected]
    assert sorted(measured, key=lambda region: (region.end, region.start)) == sorted(
        set(measured), key=lambda region: (region.end, region.start)
    )
    assert len(measured) == len(_ladder(_decaying()[: round(_HELD_S * SR)], SR, tight, _FREQ))


# --- what the settlement carries ------------------------------------------------------------------


def test_the_settlement_states_the_span_its_candidates_were_measured_over(loop: LoopFactory) -> None:
    """The bound is what a report reads back, so a run says how far it looked as well as what it found."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=1.5)

    assert settlement.search_s == 1.5


def test_a_loop_is_placed_inside_the_span_the_material_asks_for(loop: LoopFactory) -> None:
    """A loop ending past the played span would store more than keeping that span, so none is offered there."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=1.5)

    assert settlement.loops
    assert all(stored.loop.end <= round(1.5 * SR) for stored in settlement.offered)


def test_a_struck_note_carries_the_decline_the_recording_goes_on_making(loop: LoopFactory) -> None:
    settlement = settle_loop(_decaying(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.loops
    for stored in settlement.offered:
        assert stored.decay is not None
        assert stored.decay.final_gain < 1.0
        # each offer holds one level from its own start, so each fits its own ramp
        assert stored.decay.start_s == stored.loop.start / SR


def test_a_region_falling_faster_than_the_gate_admits_is_climbed_past(loop: LoopFactory) -> None:
    """Flattening a steep region means fighting it, so the gate turns one away the way the other gates do."""
    config = loop(quality={**_WIDE_OPEN, "max_level_drift_db": _SHUT})

    settlement = settle_loop(_decaying(), SR, config, root_hz=_FREQ, search_s=_HELD_S)

    assert not settlement.loops
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.LEVEL}


def test_the_decline_is_read_over_the_whole_recording_rather_than_the_span_searched(loop: LoopFactory) -> None:
    """Every level the recording states reaches the ramp, so a held note falls the way the material did."""
    settlement = settle_loop(_decaying(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=1.5)

    assert settlement.cheapest is not None
    decay = settlement.cheapest.decay
    assert decay is not None
    assert decay.end_s == _HELD_S


def test_material_no_loop_has_purchase_on_settles_none_and_names_nothing(loop: LoopFactory) -> None:
    """Noise carries no period, so the ladder is empty and the recording is stored over the span it plays."""
    noise = np.random.default_rng(0).standard_normal(int(_HELD_S * SR))

    settlement = settle_loop(noise, SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert not settlement.loops
    assert settlement.rejected == ()


def test_a_recording_shorter_than_the_shortest_accepted_loop_settles_none(loop: LoopFactory) -> None:
    config = loop(quality=_WIDE_OPEN)
    brief = config.seam.min_fade_s + config.geometry.min_loop_s / 2

    settlement = settle_loop(_tone(brief), SR, config, root_hz=_FREQ, search_s=brief)

    assert not settlement.loops


def test_the_stored_loop_reads_its_region_and_its_decline_through_one_settled_value(loop: LoopFactory) -> None:
    """What the encoder is handed and what a report states are the same value read two ways."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.cheapest is not None
    assert settlement.cheapest.loop is settlement.cheapest.settled.loop
    assert settlement.cheapest.decay is settlement.cheapest.settled.decay


def test_the_encoder_is_handed_every_offer_in_the_order_it_indexes_them(loop: LoopFactory) -> None:
    """An encoding names a loop by its place on the frontier, so the two orders have to be the one order."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.settled == tuple(stored.settled for stored in settlement.offered)
