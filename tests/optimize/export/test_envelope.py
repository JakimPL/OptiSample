from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from optisample.dsp.decay import LinearDecay
from optisample.dsp.levels import gain_to_db
from optisample.io.tracker.target import ExportTarget
from optisample.optimize.export.envelope import (
    NO_ENVELOPE,
    decay_dispersion,
    shared_decay,
    volume_envelope,
)
from optisample.optimize.layers.slots import plan_slots
from optisample.optimize.plans import InstrumentPlan
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.timing.clock import tick_seconds
from trackmod.limits.bound import Bound
from trackmod.module.protocol import TrackerModule
from trackmod.spec.levels import MAX_VOLUME, MIN_VOLUME

_TEMPO = 125
_TICK_S = tick_seconds(_TEMPO)
_RELEASE_S = 0.25
_TICKS = Bound(minimum=0, maximum=65_535)
_VOLUME = Bound(minimum=MIN_VOLUME, maximum=MAX_VOLUME)
_IT_NODES = 25
_XM_NODES = 12
_LOOPED_BUDGET_KB = 40.0  # tight enough that a plan buys the bytes a loop saves, and wide enough to be feasible

_GENTLE = LinearDecay(start_s=0.5, end_s=3.0, final_gain=0.6)
_MEDIUM = LinearDecay(start_s=0.5, end_s=3.0, final_gain=0.25)
_STEEP = LinearDecay(start_s=0.5, end_s=3.0, final_gain=0.05)


def _envelope(decay: LinearDecay, *, ticks: Bound = _TICKS) -> Envelope:
    return volume_envelope(decay, tempo=_TEMPO, release_s=_RELEASE_S, tick_bound=ticks, value_bound=_VOLUME)


def test_the_curve_holds_full_volume_until_the_stored_material_stops_following_the_recording() -> None:
    """The attack is carried by the waveform, so the envelope leaves it at the level it was stored at."""
    points = _envelope(_MEDIUM).points
    assert points[0].tick == 0
    assert points[0].value == MAX_VOLUME
    assert points[1].value == MAX_VOLUME
    assert points[1].tick == pytest.approx(_MEDIUM.start_s / _TICK_S, abs=1)


def test_the_curve_reaches_the_level_the_recording_fell_to() -> None:
    """The whole point of the envelope: a held note ends up where the recording said it would."""
    held = _envelope(_MEDIUM).points[2]
    assert held.tick == pytest.approx(_MEDIUM.end_s / _TICK_S, abs=1)
    assert held.value == round(MAX_VOLUME * _MEDIUM.final_gain)


def test_a_released_note_is_let_go_rather_than_left_ringing() -> None:
    """An instrument file carries no pattern, so silence has to be stated by the curve itself."""
    envelope = _envelope(_MEDIUM)
    release = envelope.points[-1]
    assert release.value == MIN_VOLUME
    assert release.tick == pytest.approx((_MEDIUM.end_s + _RELEASE_S) / _TICK_S, abs=1)


def test_a_held_note_stays_where_the_decline_left_it() -> None:
    """Sustaining on the level the recording reached is what keeps the release from starting early."""
    envelope = _envelope(_MEDIUM)
    assert envelope.sustain is not None
    assert envelope.sustain.begin == envelope.sustain.end
    assert envelope.points[envelope.sustain.begin].value == round(MAX_VOLUME * _MEDIUM.final_gain)


@pytest.mark.parametrize("nodes", [_IT_NODES, _XM_NODES])
def test_the_curve_fits_the_breakpoints_every_target_format_numbers(nodes: int) -> None:
    """Both formats have to hold it, and the narrower of the two is what bounds the shape."""
    assert _envelope(_STEEP).length <= nodes


def test_the_breakpoints_ascend_even_where_the_ramp_is_shorter_than_a_tick() -> None:
    """A loop settled inside one tick still has to name two distinct ends, which the format requires."""
    instant = LinearDecay(start_s=0.0, end_s=_TICK_S / 100, final_gain=0.5)
    ticks = [point.tick for point in _envelope(instant).points]
    assert ticks == sorted(set(ticks))


def test_a_curve_running_past_the_last_tick_keeps_every_breakpoint() -> None:
    """Losing a node would lose where the decline ends, so the curve is pulled back into the ticks left."""
    envelope = _envelope(_MEDIUM, ticks=Bound(minimum=0, maximum=3))
    ticks = [point.tick for point in envelope.points]
    assert ticks == sorted(set(ticks))
    assert max(ticks) <= 3


@dataclass(frozen=True)
class _SharedCase:
    name: str
    decays: tuple[LinearDecay | None, ...]
    expected: LinearDecay | None


@pytest.mark.parametrize(
    "case",
    [
        _SharedCase("the middle decline of three", (_GENTLE, _MEDIUM, _STEEP), _MEDIUM),
        _SharedCase("the only one stated", (None, _STEEP, None), _STEEP),
        _SharedCase("none where no sample declines", (None, None), NO_ENVELOPE),
        _SharedCase("none where the slot stores nothing", (), NO_ENVELOPE),
    ],
    ids=lambda case: case.name,
)
def test_one_shape_answers_for_every_sample_the_instrument_starts(case: _SharedCase) -> None:
    """A format gives the envelope to the instrument, so the samples inside one share a single curve."""
    assert shared_decay(case.decays) == case.expected


def test_the_shared_shape_is_a_decline_some_recording_actually_makes() -> None:
    """Averaging incompatible ramps would write a curve no key plays, so a real member is taken whole."""
    assert shared_decay((_GENTLE, _MEDIUM, _STEEP)) in (_GENTLE, _MEDIUM, _STEEP)


def test_sharing_one_shape_costs_the_sample_it_suits_worst() -> None:
    """This is the reading that says whether the keys sharing an envelope want splitting apart."""
    decays = (_GENTLE, _MEDIUM, _STEEP)
    shared = shared_decay(decays)
    assert shared is not None
    worst = max(abs(gain_to_db(decay.final_gain) - gain_to_db(shared.final_gain)) for decay in decays)
    assert decay_dispersion(decays, shared) == pytest.approx(worst)


def test_samples_declining_alike_give_up_nothing_for_sharing_a_curve() -> None:
    """Where every key falls at one rate the shared shape is each key's own, which costs none of them."""
    assert decay_dispersion((_MEDIUM, _MEDIUM, _MEDIUM), _MEDIUM) == pytest.approx(0.0)


def test_an_instrument_whose_samples_state_no_decline_gives_up_nothing() -> None:
    """A slot storing only material that carries its own levels has nothing for an envelope to cost."""
    assert decay_dispersion((None, None), _MEDIUM) == pytest.approx(0.0)


def test_the_written_module_plays_a_looped_note_down_rather_than_ringing(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
    target: ExportTarget,
) -> None:
    """The whole gap this closes: a looped sample holds one level, so the module has to state the decline.

    Only a looped span carries a decline (a trimmed one keeps every level in its own material), so an
    instrument storing no loop leaves its voices alone and at least one storing a loop writes the curve.
    The budget is tight enough that the bytes a loop saves are worth what holding a region at one level
    costs, which is the trade the solver declines wherever the material fits whole.
    """
    plan, module = build(budget_kb=_LOOPED_BUDGET_KB)
    song = module.song
    layout = plan_slots(plan, target)
    carried = [instrument.volume_envelope is not None for instrument in song.instruments]
    loops = [any(song.samples[sample].loop is not None for sample in slot.samples) for slot in layout.slots]

    assert any(loops), "the demo plan stores no loop, so this test would prove nothing"
    assert any(carried)
    assert not any(written for written, stores_loop in zip(carried, loops) if not stores_loop)
