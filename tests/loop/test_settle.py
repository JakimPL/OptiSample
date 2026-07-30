from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.loop import LoopConfig
from optisample.loop.settle import Gate, settle_loop

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


def test_the_cheapest_candidate_clearing_the_gates_is_the_one_stored(loop: LoopFactory) -> None:
    """Gates open, so the front of the ladder is kept and nothing is climbed past to reach it."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), search_s=_HELD_S)

    assert settlement.loops
    assert settlement.rejected == ()


def test_a_loop_is_stored_no_earlier_than_the_attack_the_geometry_skips(loop: LoopFactory) -> None:
    config = loop(quality=_WIDE_OPEN)
    settlement = settle_loop(_tone(), SR, config, search_s=_HELD_S)

    assert settlement.stored is not None
    period = round(SR / _FREQ)
    assert settlement.stored.loop.start >= round(config.geometry.attack_skip_s * SR) - period


def test_every_candidate_the_ladder_climbs_past_names_the_gate_it_fell_outside(loop: LoopFactory) -> None:
    """A gate nothing can clear leaves the whole ladder reported, so the reason is readable per candidate."""
    settlement = settle_loop(_tone(), SR, loop(quality=_SEAM_SHUT), search_s=_HELD_S)

    assert settlement.stored is None
    assert settlement.rejected != ()
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.SEAM}


def test_a_loop_holding_a_timbre_the_material_moves_away_from_is_climbed_past(loop: LoopFactory) -> None:
    """A recording that brightens halfway leaves its early loops holding a timbre the rest no longer has."""
    brightened = np.concatenate([_tone(1.5), _tone(1.5, freq=5 * _FREQ)])
    config = loop(quality={**_WIDE_OPEN, "max_spectral_distance_db": _SHUT})

    settlement = settle_loop(brightened, SR, config, search_s=_HELD_S)

    assert settlement.rejected != ()
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.TIMBRE}
    assert settlement.stored is not None
    assert settlement.stored.loop.start > round(1.5 * SR)  # the loop kept sits in the brightened stretch


def test_the_ladder_is_climbed_cheapest_first(loop: LoopFactory) -> None:
    """Candidates are tried in the order they cost, so what is rejected is always cheaper than what is kept."""
    tight = loop(quality={**_WIDE_OPEN, "max_seam_step": 1.0})
    settlement = settle_loop(_decaying(), SR, tight, search_s=_HELD_S)

    ends = [rejected.loop.end for rejected in settlement.rejected]
    assert ends == sorted(ends)
    if settlement.stored is not None:
        assert all(end <= settlement.stored.loop.end for end in ends)


# --- what the settlement carries ------------------------------------------------------------------


def test_the_settlement_states_the_span_its_candidates_were_measured_over(loop: LoopFactory) -> None:
    """The bound is what a report reads back, so a run says how far it looked as well as what it found."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), search_s=1.5)

    assert settlement.search_s == 1.5


def test_a_loop_is_placed_inside_the_span_the_material_asks_for(loop: LoopFactory) -> None:
    """A loop ending past the played span would store more than keeping that span, so none is offered there."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), search_s=1.5)

    assert settlement.stored is not None
    assert settlement.stored.loop.end <= round(1.5 * SR)


def test_a_struck_note_carries_the_decline_the_recording_goes_on_making(loop: LoopFactory) -> None:
    settlement = settle_loop(_decaying(), SR, loop(quality=_WIDE_OPEN), search_s=_HELD_S)

    assert settlement.stored is not None
    decay = settlement.stored.decay
    assert decay is not None
    assert decay.final_gain < 1.0
    assert decay.start_s == settlement.stored.loop.start / SR  # the region is stored at one level from here on


def test_a_region_falling_faster_than_the_gate_admits_is_climbed_past(loop: LoopFactory) -> None:
    """Flattening a steep region means fighting it, so the gate turns one away the way the other gates do."""
    config = loop(quality={**_WIDE_OPEN, "max_level_drift_db": _SHUT})

    settlement = settle_loop(_decaying(), SR, config, search_s=_HELD_S)

    assert not settlement.loops
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.LEVEL}


def test_the_decline_is_read_over_the_whole_recording_rather_than_the_span_searched(loop: LoopFactory) -> None:
    """Every level the recording states reaches the ramp, so a held note falls the way the material did."""
    settlement = settle_loop(_decaying(), SR, loop(quality=_WIDE_OPEN), search_s=1.5)

    assert settlement.stored is not None
    decay = settlement.stored.decay
    assert decay is not None
    assert decay.end_s == _HELD_S


def test_material_no_loop_has_purchase_on_settles_none_and_names_nothing(loop: LoopFactory) -> None:
    """Noise carries no period, so the ladder is empty and the recording is stored over the span it plays."""
    noise = np.random.default_rng(0).standard_normal(int(_HELD_S * SR))

    settlement = settle_loop(noise, SR, loop(quality=_WIDE_OPEN), search_s=_HELD_S)

    assert not settlement.loops
    assert settlement.rejected == ()


def test_a_recording_shorter_than_the_shortest_accepted_loop_settles_none(loop: LoopFactory) -> None:
    config = loop(quality=_WIDE_OPEN)
    brief = config.geometry.attack_skip_s + config.geometry.min_loop_s / 2

    settlement = settle_loop(_tone(brief), SR, config, search_s=brief)

    assert not settlement.loops


def test_the_stored_loop_reads_its_region_and_its_decline_through_one_settled_value(loop: LoopFactory) -> None:
    """What the encoder is handed and what a report states are the same value read two ways."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), search_s=_HELD_S)

    assert settlement.stored is not None
    assert settlement.stored.loop is settlement.stored.settled.loop
    assert settlement.stored.decay is settlement.stored.settled.decay
