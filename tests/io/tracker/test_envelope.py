from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.level import Clock, curve_level, gain_to_db, written_level
from optisample.dsp.piecewise import CurveNode, PiecewiseCurve
from optisample.dsp.series import Series
from optisample.io.tracker.envelope import (
    NO_ENVELOPE,
    EnvelopeGrid,
    carried_signal,
    envelope_level,
    played_gain,
    shape_nodes,
    sounding_gain,
    sounding_level,
    volume_envelope,
)
from trackmod.core.envelopes.curve import Breakpoint, timed_envelope
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.span import EnvelopeSpan
from trackmod.core.timing.clock import tick_seconds
from trackmod.limits.bound import Bound
from trackmod.spec.levels import MAX_VOLUME, MIN_VOLUME

_TEMPO = 125
_TICK_S = tick_seconds(_TEMPO)
_RELEASE_S = 0.25
_TICKS = Bound(minimum=0, maximum=65_535)
_VOLUME = Bound(minimum=MIN_VOLUME, maximum=MAX_VOLUME)
_IT_POINTS = Bound(minimum=1, maximum=25)
_XM_POINTS = Bound(minimum=1, maximum=12)
_QUIETEST_STEP = 1  # the softest level a node states short of silencing the voice
_FLOOR_DB = gain_to_db(_QUIETEST_STEP / MAX_VOLUME)  # how far under its peak any written curve reaches
_MOMENTS = 4096  # moments a written curve is read back at, fine enough to price the runs between its corners
# a run reaching the quietest step turns several decibels over its last tick, which is the widest the
# amplitude walk a tracker makes and the decibel walk a level makes stand apart anywhere
_ONE_TICK_DB = 0.5

# A struck note: a fast fall, a long ring, then the material running out.
_STRUCK = PiecewiseCurve(
    nodes=(
        CurveNode(seconds=0.0, value=-6.0),
        CurveNode(seconds=0.3, value=-9.0),
        CurveNode(seconds=1.0, value=-24.0),
        CurveNode(seconds=2.0, value=-28.0),
    )
)
_DEEP = PiecewiseCurve(  # a note falling further than the grid's own floor reaches
    nodes=(CurveNode(seconds=0.0, value=0.0), CurveNode(seconds=2.0, value=-30.0), CurveNode(seconds=6.0, value=-60.0))
)
_QUIET = PiecewiseCurve(  # a second instrument of the same plan, sitting well under the first
    nodes=(CurveNode(seconds=0.0, value=-18.0), CurveNode(seconds=1.0, value=-24.0))
)


def _grid(ticks: Bound = _TICKS) -> EnvelopeGrid:
    return EnvelopeGrid(tempo=_TEMPO, release_s=_RELEASE_S, tick_bound=ticks, value_bound=_VOLUME)


def _written(curve: PiecewiseCurve, *, ticks: Bound = _TICKS) -> Envelope:
    level = curve_level(curve, Clock.PLAYED)
    return volume_envelope(written_level(level, reference_db=level.peak_db), _grid(ticks))


def _silencing() -> Envelope:
    """A curve running to the step that silences a voice outright, which no division may be taken against."""
    return timed_envelope(
        (Breakpoint(seconds=0.0, value=MAX_VOLUME), Breakpoint(seconds=4 * _TICK_S, value=MIN_VOLUME)),
        tempo=_TEMPO,
        tick_bound=_TICKS,
        value_bound=_VOLUME,
        sustain=EnvelopeSpan(begin=1, end=1),
    )


def _rounded(curve: PiecewiseCurve) -> Envelope:
    """The same curve written by rounding alone, which is what the repair sweep is measured against."""
    levels_db = curve.values - np.max(curve.values)
    seconds = (0.0, *(float(moment) for moment in curve.seconds[1:]))
    shape = tuple(
        Breakpoint(seconds=moment, value=round(MAX_VOLUME * 10.0 ** (level / 20.0)))
        for moment, level in zip(seconds, levels_db)
    )
    return timed_envelope(
        (*shape, Breakpoint(seconds=float(curve.seconds[-1]) + _RELEASE_S, value=MIN_VOLUME)),
        tempo=_TEMPO,
        tick_bound=_TICKS,
        value_bound=_VOLUME,
        sustain=EnvelopeSpan(begin=len(shape) - 1, end=len(shape) - 1),
    )


def _played_db(envelope: Envelope, moments: Series) -> Series:
    """What the format actually plays at each moment: its steps read straight through, in decibels."""
    ticks = np.asarray([point.tick for point in envelope.points], dtype=np.float64)
    values = np.asarray([point.value for point in envelope.points], dtype=np.float64)
    return gain_to_db(np.asarray(np.interp(moments / _TICK_S, ticks, values), dtype=np.float64) / MAX_VOLUME)


def _gaps_db(envelope: Envelope, curve: PiecewiseCurve) -> Series:
    """How far the written curve stands from the shape, over every moment the shape spans."""
    moments = np.linspace(0.0, float(curve.seconds[-1]), _MOMENTS)
    target = np.maximum(curve.at(moments) - np.max(curve.values), _FLOOR_DB)
    return np.abs(_played_db(envelope, moments) - target)


def _shape_points(envelope: Envelope) -> tuple[int, ...]:
    """The levels the shape itself turns through, which is every point the release does not spend."""
    return tuple(point.value for point in envelope.points[:-1])


# --- how many corners a format leaves the shape -----------------------------------------------------------


@pytest.mark.parametrize(("points", "corners"), [(_IT_POINTS, 24), (_XM_POINTS, 11)])
def test_a_shape_is_given_every_point_the_release_does_not_spend(points: Bound, corners: int) -> None:
    assert shape_nodes(points) == corners


# --- what the written curve says --------------------------------------------------------------------------


def test_the_loudest_moment_written_takes_the_unity_step() -> None:
    """A volume envelope multiplies, so the level it is written against is unity and the rest attenuates."""
    written = _written(_STRUCK)

    assert written.points[0].value == MAX_VOLUME
    assert max(_shape_points(written)) == MAX_VOLUME


def test_the_curve_opens_at_the_onset() -> None:
    """A voice starts at the first tick, so the curve has to state a level there rather than half a window in."""
    assert _written(_STRUCK).points[0].tick == 0


def test_each_corner_lands_on_the_tick_it_falls_on() -> None:
    """Time is what the format counts in ticks, so a corner keeps the moment it was fitted at."""
    written = _written(_STRUCK)

    placed = [point.tick for point in written.points[1:-1]]

    assert placed == [round(moment / _TICK_S) for moment in _STRUCK.seconds[1:]]


def test_a_released_note_is_let_go_rather_than_left_ringing() -> None:
    """An instrument file carries no pattern, so silence has to be stated by the curve itself."""
    release = _written(_STRUCK).points[-1]

    assert release.value == MIN_VOLUME
    assert release.tick == pytest.approx((float(_STRUCK.seconds[-1]) + _RELEASE_S) / _TICK_S, abs=1)


def test_a_held_note_stays_where_the_shape_left_it() -> None:
    """Sustaining on the last corner is what keeps a held note from starting its release early."""
    written = _written(_STRUCK)

    assert written.sustain is not None
    assert written.sustain.begin == written.sustain.end == len(written.points) - 2


def test_the_breakpoints_ascend_even_where_two_corners_fall_inside_one_tick() -> None:
    """A shape read at a finer window than the tick grid still has to name distinct ticks."""
    crowded = PiecewiseCurve(
        nodes=tuple(CurveNode(seconds=moment * _TICK_S / 100.0, value=-3.0 * moment) for moment in range(5))
    )

    ticks = [point.tick for point in _written(crowded).points]

    assert ticks == sorted(set(ticks))


def test_a_curve_running_past_the_last_tick_keeps_every_breakpoint() -> None:
    """Losing a node would lose where the decline ends, so the curve is pulled back into the ticks left."""
    narrowest = len(_STRUCK.nodes)  # the last tick a curve of this many corners and its release can reach

    ticks = [point.tick for point in _written(_STRUCK, ticks=Bound(minimum=0, maximum=narrowest)).points]

    assert ticks == sorted(set(ticks))
    assert max(ticks) <= narrowest


# --- what the repair sweep buys -----------------------------------------------------------------------------


@pytest.mark.parametrize("curve", [_STRUCK, _DEEP], ids=["a struck note", "a fall past the grid's own floor"])
def test_repairing_the_rounding_runs_the_played_curve_closer_to_the_shape(curve: PiecewiseCurve) -> None:
    """Rounding each level on its own ignores the runs between them, which is what one sweep answers."""
    repaired = _gaps_db(_written(curve), curve)
    plain = _gaps_db(_rounded(curve), curve)

    assert np.max(repaired) < np.max(plain)
    assert np.mean(repaired) < np.mean(plain)


def test_a_level_the_grid_cannot_reach_is_written_at_the_quietest_step_it_holds() -> None:
    """Sixty-four amplitude steps reach 36 dB, so a note falling further is held at the floor and released."""
    shape = _shape_points(_written(_DEEP))

    assert min(shape) == _QUIETEST_STEP
    assert np.max(_gaps_db(_written(_DEEP), _DEEP)) < abs(_FLOOR_DB)


def test_a_shape_that_never_declines_is_written_at_unity_throughout() -> None:
    """An instrument whose samples carry every level they play at asks the envelope for nothing."""
    flat = PiecewiseCurve(nodes=(CurveNode(seconds=0.0, value=-4.0), CurveNode(seconds=2.0, value=-4.0)))

    assert _shape_points(_written(flat)) == (MAX_VOLUME, MAX_VOLUME)


def test_a_corner_its_neighbors_leave_no_moment_between_keeps_the_step_it_was_rounded_onto() -> None:
    """A curve running for minutes is priced at a thinned grid, which two corners can fall between."""
    span_s = 200.0
    crowded = PiecewiseCurve(
        nodes=(
            CurveNode(seconds=0.0, value=0.0),
            *(CurveNode(seconds=100.0 + step * _TICK_S, value=-12.0 - step) for step in range(3)),
            CurveNode(seconds=span_s, value=-24.0),
        )
    )

    shape = _shape_points(_written(crowded))

    assert shape[2] == round(MAX_VOLUME * 10.0 ** (-13.0 / 20.0))


# --- what a written curve multiplies a held voice by ----------------------------------------------------------


def _held_envelope() -> Envelope:
    """A curve falling over four ticks and sustaining there, which is the shape a struck note is written as."""
    return timed_envelope(
        (
            Breakpoint(seconds=0.0, value=MAX_VOLUME),
            Breakpoint(seconds=4 * _TICK_S, value=MAX_VOLUME // 4),
            Breakpoint(seconds=8 * _TICK_S, value=MIN_VOLUME),
        ),
        tempo=_TEMPO,
        tick_bound=_TICKS,
        value_bound=_VOLUME,
        sustain=EnvelopeSpan(begin=1, end=1),
    )


def test_the_gain_walks_straight_in_amplitude_between_two_corners() -> None:
    """A tracker interpolates its envelope in amplitude, so the gain read back is the straight line."""
    rate = 1000
    frames = round(4 * _TICK_S * rate)
    gain = played_gain(_held_envelope(), tempo=_TEMPO, frames=frames, sample_rate=rate)

    assert gain[0] == pytest.approx(1.0)
    assert gain[-1] == pytest.approx(0.25, abs=0.01)
    assert np.all(np.diff(gain) <= 0.0)
    assert gain == pytest.approx(np.linspace(gain[0], gain[-1], frames), abs=0.01)


def test_a_voice_held_past_the_shape_stays_where_the_shape_left_it() -> None:
    """The breakpoint past the sustain carries a released note, so a held one never reaches it."""
    rate = 1000
    reached = round(4 * _TICK_S * rate)
    gain = played_gain(_held_envelope(), tempo=_TEMPO, frames=reached * 2, sample_rate=rate)

    assert gain[reached:] == pytest.approx(gain[-1])
    assert gain[-1] == pytest.approx(0.25, abs=0.01)


def test_a_curve_written_for_a_faster_clock_falls_over_less_time() -> None:
    """Ticks are what a format counts envelope time in, so the curve holds only for the tempo it was written for."""
    rate = 1000
    frames = round(4 * tick_seconds(_TEMPO) * rate)
    slower = played_gain(_held_envelope(), tempo=_TEMPO, frames=frames, sample_rate=rate)
    faster = played_gain(_held_envelope(), tempo=_TEMPO * 2, frames=frames, sample_rate=rate)

    assert faster[-1] < slower[-1]


def test_the_sounding_gain_stays_above_the_step_that_silences_a_voice() -> None:
    """A moment the curve silences plays as silence, so dividing a recording by it stays finite."""
    rate = 1000
    silencing = _silencing()
    frames = round(8 * _TICK_S * rate)
    played = played_gain(silencing, tempo=_TEMPO, frames=frames, sample_rate=rate)
    sounding = sounding_gain(silencing, tempo=_TEMPO, frames=frames, sample_rate=rate)

    assert np.min(played) == pytest.approx(0.0)
    assert np.all(sounding >= _QUIETEST_STEP / MAX_VOLUME)
    assert np.all(np.isfinite(1.0 / sounding))


# --- the written curve as the level algebra carries it ----------------------------------------------------


def test_a_written_curve_is_read_on_the_clock_a_tracker_walks_it_on() -> None:
    """A tracker walks ticks whatever key is struck, so the curve stands on the played clock."""
    assert envelope_level(_written(_STRUCK), tempo=_TEMPO).clock is Clock.PLAYED


def test_an_instrument_carrying_no_curve_leaves_its_voices_where_their_material_holds_them() -> None:
    assert envelope_level(NO_ENVELOPE, tempo=_TEMPO).transparent


def test_the_level_and_the_played_gain_agree_at_every_tick_the_curve_spans() -> None:
    """A tracker sets a voice's volume once a tick, so both readings state the same level at each of them."""
    envelope = _written(_STRUCK)
    rate = 44_100
    ticks = np.arange(envelope.points[0].tick, envelope.points[-1].tick + 1, dtype=np.float64)
    seconds = ticks * _TICK_S
    frames = round(float(seconds[-1]) * rate) + 1
    walked = played_gain(envelope, tempo=_TEMPO, frames=frames, sample_rate=rate)
    landed = np.asarray([walked[min(round(moment * rate), frames - 1)] for moment in seconds], dtype=np.float64)

    assert envelope_level(envelope, tempo=_TEMPO).gain(seconds) == pytest.approx(landed, abs=1e-9)


def test_between_two_ticks_the_two_readings_stay_inside_the_step_the_grid_moves_by() -> None:
    """Both smooth a staircase the tracker climbs once a tick, so they part by less than one of its steps."""
    envelope = _written(_STRUCK)
    rate = 44_100
    frames = round(envelope.points[-1].tick * _TICK_S * rate)
    walked = played_gain(envelope, tempo=_TEMPO, frames=frames, sample_rate=rate)
    carried = envelope_level(envelope, tempo=_TEMPO).frame_gains(frames, rate)
    apart = np.abs(gain_to_db(np.maximum(walked, _QUIETEST_STEP / MAX_VOLUME)) - gain_to_db(carried))

    assert float(np.max(apart)) < abs(gain_to_db(_QUIETEST_STEP / (_QUIETEST_STEP + 1)))


def test_the_sounding_level_stays_above_the_step_that_silences_a_voice() -> None:
    """A carrier was divided against that step, so the level playing it back stands against it too."""
    silencing = _silencing()

    assert envelope_level(silencing, tempo=_TEMPO).readings.values.min() < _FLOOR_DB
    assert sounding_level(silencing, tempo=_TEMPO).readings.values.min() == pytest.approx(_FLOOR_DB)


def test_an_instrument_carrying_no_curve_sounds_its_voices_where_their_material_holds_them() -> None:
    assert sounding_level(NO_ENVELOPE, tempo=_TEMPO).transparent


@pytest.mark.parametrize("curve", [_STRUCK, _DEEP])
def test_a_carrier_played_back_through_its_own_level_is_the_material_it_was_taken_from(
    curve: PiecewiseCurve,
) -> None:
    """The round trip step 4 rests on: what the division handed to the envelope, the level hands back.

    A stored carrier travels with this level and the surrogate renderer plays it through, so the two
    standing inverse is what has a carrier scored as the recording it stands for rather than as the
    level-flat waveform it is stored as. The pair walks the ticks between two nodes in amplitude on the
    dividing side and in decibels on the returning one, so the material comes back within the decibels a
    single tick turns through.
    """
    envelope = _written(curve)
    rate = 22_050
    frames = round(envelope.points[-1].tick * _TICK_S * rate)
    material = np.cos(np.arange(frames, dtype=np.float64) * 0.05) * np.linspace(1.0, 0.2, frames)

    carrier = carried_signal(material, envelope, tempo=_TEMPO, sample_rate=rate)
    sounded = carrier * sounding_level(envelope, tempo=_TEMPO).frame_gains(frames, rate)

    assert float(np.max(np.abs(gain_to_db(np.abs(sounded)) - gain_to_db(np.abs(material))))) < _ONE_TICK_DB
