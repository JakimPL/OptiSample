from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.loop import LoopConfig
from optisample.dsp.loop import Material, seam_frames
from optisample.loop.settle import Gate, Settlement, StoredLoop, _ordered_search, settle_loop
from tests.conftest import recorded

SR = 22_050
_FREQ = 220.0
_HELD_S = 3.0  # long enough for the frontier to lay out several candidates and still leave a tail
_DECLINE_WINDOW_S: Final = 0.05  # the window a decline's last reading centers on, which bounds where it lands
# the last whole window centers half a window before the material it covers ends, and the reading stops at
# the last whole window, so the furthest moment a decline states sits within one and a half of the end
_DECLINE_REACH_S: Final = 1.5 * _DECLINE_WINDOW_S
_BRIGHTENS_AT = round(_HELD_S / 2 * SR)  # the frame a brightening recording changes its timbre at

LoopFactory = Callable[..., LoopConfig]

_OPEN: Final = 1e9  # a gate nothing measures past, which leaves the one under test to decide alone
_SHUT: Final = 1e-9  # a gate nothing clears, which is how every candidate is made to report

_WIDE_OPEN: Final = {"max_seam_step": _OPEN, "max_spectral_distance_db": _OPEN}
_SEAM_SHUT: Final = {**_WIDE_OPEN, "max_seam_step": _SHUT}


def _tone(duration_s: float = _HELD_S, freq: float = _FREQ) -> NDArray[np.float64]:
    """A steady harmonic tone over a recording's noise floor, which the geometry can loop anywhere inside."""
    times = np.arange(int(duration_s * SR), dtype=np.float64) / SR
    return recorded(0.7 * np.sin(2.0 * np.pi * freq * times) + 0.2 * np.sin(4.0 * np.pi * freq * times))


def _decaying(duration_s: float = _HELD_S) -> NDArray[np.float64]:
    """A struck note: one pitch throughout, under an amplitude that falls away as it rings."""
    tone = _tone(duration_s)
    return np.asarray(np.exp(-np.arange(tone.size, dtype=np.float64) / (0.8 * SR)) * tone)


def _brightening() -> NDArray[np.float64]:
    """A recording that jumps to a far brighter timbre halfway through and holds it to the end."""
    return np.concatenate([_tone(_HELD_S / 2), _tone(_HELD_S / 2, freq=5 * _FREQ)])


def _stored_span(stored: StoredLoop, config: LoopConfig) -> tuple[int, int]:
    """The stretch of the recording a stored loop carries, in frames.

    A wrap blends the frames preceding the loop start into its end, so the material a stored loop carries
    opens one blend before its start and runs to the loop's own end.
    """
    return stored.loop.start - seam_frames(stored.loop, SR, config.seam), stored.loop.end


def _holds_only_the_older_timbre(stored: StoredLoop, config: LoopConfig) -> bool:
    """Whether everything this loop stores was recorded before the brightening."""
    return _stored_span(stored, config)[1] <= _BRIGHTENS_AT


def _holds_only_the_newer_timbre(stored: StoredLoop, config: LoopConfig) -> bool:
    """Whether everything this loop stores was recorded after the brightening."""
    return _stored_span(stored, config)[0] >= _BRIGHTENS_AT


def _split_at_the_brightening(
    settlement: Settlement,
    config: LoopConfig,
) -> tuple[list[StoredLoop], list[StoredLoop]]:
    """The offers holding the recording's first timbre alone, and those holding the one it moved to alone.

    An offer reaching across the brightening holds some of each, so it stands on neither side and is left
    out of both -- which is what makes the two lists comparable as timbres rather than as mixtures.
    """
    early = [stored for stored in settlement.offered if _holds_only_the_older_timbre(stored, config)]
    late = [stored for stored in settlement.offered if _holds_only_the_newer_timbre(stored, config)]
    return early, late


# --- what the frontier offers ---------------------------------------------------------------------


def test_every_candidate_clearing_the_gates_is_offered(loop: LoopFactory) -> None:
    """Gates open, so every candidate the material offers comes back and none of it is turned down."""
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


def test_every_candidate_turned_down_names_the_gate_it_fell_outside(loop: LoopFactory) -> None:
    """A gate nothing can clear leaves every candidate reported, so the reason is readable per candidate."""
    settlement = settle_loop(_tone(), SR, loop(quality=_SEAM_SHUT), root_hz=_FREQ, search_s=_HELD_S)

    assert not settlement.loops
    assert settlement.cheapest is None
    assert settlement.rejected != ()
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.SEAM}


def test_a_loop_holding_a_timbre_the_material_moves_away_from_measures_the_distance(
    loop: LoopFactory,
) -> None:
    """A recording that brightens halfway leaves its early loops holding a timbre the rest no longer has."""
    config = loop(quality=_WIDE_OPEN)
    settlement = settle_loop(_brightening(), SR, config, root_hz=_FREQ, search_s=_HELD_S)

    early, late = _split_at_the_brightening(settlement, config)
    assert early and late
    assert max(stored.quality.spectral_distance for stored in late) < min(
        stored.quality.spectral_distance for stored in early
    )


def test_a_gate_on_that_distance_turns_down_the_loops_holding_the_older_timbre(loop: LoopFactory) -> None:
    """The distance is what the gate reads, so a bound between the two sides admits only the later loops."""
    wide = loop(quality=_WIDE_OPEN)
    measured = settle_loop(_brightening(), SR, wide, root_hz=_FREQ, search_s=_HELD_S)
    early, late = _split_at_the_brightening(measured, wide)
    between = (
        max(stored.quality.spectral_distance for stored in late)
        + min(stored.quality.spectral_distance for stored in early)
    ) / 2

    config = loop(quality={**_WIDE_OPEN, "max_spectral_distance_db": between})
    settlement = settle_loop(_brightening(), SR, config, root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.loops
    assert {rejected.gate for rejected in settlement.rejected} == {Gate.TIMBRE}
    assert not any(_holds_only_the_older_timbre(stored, config) for stored in settlement.offered)


def test_every_candidate_the_frontier_offers_is_either_offered_or_named(loop: LoopFactory) -> None:
    """Each candidate is measured, so a run accounts for all of them and none is skipped unmeasured."""
    tight = loop(quality={**_WIDE_OPEN, "max_seam_step": 1.0})
    settlement = settle_loop(_decaying(), SR, tight, root_hz=_FREQ, search_s=_HELD_S)

    measured = [stored.loop for stored in settlement.offered] + [turned.loop for turned in settlement.rejected]
    assert sorted(measured, key=lambda region: (region.end, region.start)) == sorted(
        set(measured), key=lambda region: (region.end, region.start)
    )
    assert len(measured) == len(_ordered_search(_decaying()[: round(_HELD_S * SR)], SR, tight, _FREQ).candidates)


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
        assert not stored.level.transparent
        assert stored.level.readings.values[-1] < 0.0
        # each offer holds one level from its own start, so each reads its own decline from there
        assert stored.level.db(np.asarray([stored.loop.start / SR])) == pytest.approx(0.0, abs=0.5)


def test_a_region_falling_across_itself_is_offered_with_the_fall_it_states(loop: LoopFactory) -> None:
    """Leveling answers for the decline, so a steep region is offered and reports the fall it flattened."""
    settlement = settle_loop(_decaying(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.loops
    assert all(stored.quality.level_drift_db > 0.0 for stored in settlement.offered)


def test_the_decline_is_read_over_the_whole_recording_rather_than_the_span_searched(loop: LoopFactory) -> None:
    """Every level the recording states reaches the ramp, so a held note falls the way the material did."""
    searched = 1.5
    settlement = settle_loop(_decaying(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=searched)

    assert settlement.cheapest is not None
    reached = settlement.cheapest.level.readings.seconds[-1]
    assert reached > searched
    assert reached == pytest.approx(_HELD_S, abs=_DECLINE_REACH_S)


def test_material_no_loop_has_purchase_on_settles_none_and_names_what_it_lacked(loop: LoopFactory) -> None:
    """Noise carries no period, so the ladder is empty and the recording is stored over the span it plays.

    No candidate reaches a gate, so nothing is rejected -- and the reading that came up short is stated
    instead, which is what separates material a gate turned down from material a gate never saw.
    """
    noise = np.random.default_rng(0).standard_normal(int(_HELD_S * SR))

    settlement = settle_loop(noise, SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert not settlement.loops
    assert settlement.rejected == ()
    assert settlement.lacking is Material.PERIOD


def test_a_recording_shorter_than_the_shortest_accepted_loop_settles_none(loop: LoopFactory) -> None:
    """A note ending before one round fits reports its length rather than an absence a reader must read."""
    config = loop(quality=_WIDE_OPEN)
    brief = config.seam.min_fade_s + config.geometry.min_loop_s / 2

    settlement = settle_loop(_tone(brief), SR, config, root_hz=_FREQ, search_s=brief)

    assert not settlement.loops
    assert settlement.lacking in {Material.STEADY, Material.ROUND}


def test_a_recording_a_gate_turned_down_lacked_nothing_to_measure(loop: LoopFactory) -> None:
    """The two ways a recording ends up unlooped stay apart: one had candidates, the other had none."""
    settlement = settle_loop(_tone(), SR, loop(quality=_SEAM_SHUT), root_hz=_FREQ, search_s=_HELD_S)

    assert not settlement.loops
    assert settlement.rejected != ()
    assert settlement.lacking is None


def test_a_recording_offering_a_loop_lacks_nothing(loop: LoopFactory) -> None:
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.loops
    assert settlement.lacking is None


def test_the_stored_loop_reads_its_region_and_its_decline_through_one_settled_value(loop: LoopFactory) -> None:
    """What the encoder is handed and what a report states are the same value read two ways."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.cheapest is not None
    assert settlement.cheapest.loop is settlement.cheapest.settled.loop
    assert settlement.cheapest.level is settlement.cheapest.settled.level


def test_the_encoder_is_handed_every_offer_in_the_order_it_indexes_them(loop: LoopFactory) -> None:
    """An encoding names a loop by its place on the frontier, so the two orders have to be the one order."""
    settlement = settle_loop(_tone(), SR, loop(quality=_WIDE_OPEN), root_hz=_FREQ, search_s=_HELD_S)

    assert settlement.settled == tuple(stored.settled for stored in settlement.offered)
