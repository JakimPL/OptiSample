from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.dsp.levels import gain_to_db
from optisample.dsp.piecewise import CurveNode, PiecewiseCurve
from optisample.dsp.series import Series
from optisample.dsp.trajectory import SharedTrajectory
from optisample.optimize.export.build import loudest_db, slot_envelope
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.envelope import NO_ENVELOPE, EnvelopeGrid, shape_nodes, volume_envelope
from optisample.optimize.export.voices import NO_SHAPE
from optisample.optimize.plans import InstrumentPlan
from trackmod.core.envelopes.curve import Breakpoint, timed_envelope
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.span import EnvelopeSpan
from trackmod.core.timing.clock import tick_seconds
from trackmod.limits.bound import Bound
from trackmod.module.protocol import TrackerModule
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
_LOOPED_BUDGET_KB = 40.0  # tight enough that a plan buys the bytes a loop saves, and wide enough to be feasible

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


def _shared(curve: PiecewiseCurve) -> SharedTrajectory:
    """One instrument's fitted shape, which is all a plan-wide level is read off."""
    return SharedTrajectory(curve=curve, offsets_db=((0.0,),), gaps_db=(0.0,))


def _grid(ticks: Bound = _TICKS) -> EnvelopeGrid:
    return EnvelopeGrid(tempo=_TEMPO, release_s=_RELEASE_S, tick_bound=ticks, value_bound=_VOLUME)


def _written(curve: PiecewiseCurve, *, ticks: Bound = _TICKS, peak_db: float | None = None) -> Envelope:
    return volume_envelope(curve, _grid(ticks), peak_db=curve.peak if peak_db is None else peak_db)


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


def test_a_corner_its_neighbours_leave_no_moment_between_keeps_the_step_it_was_rounded_onto() -> None:
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


# --- what one level for a whole plan keeps ---------------------------------------------------------------------


def test_a_plan_is_written_against_the_loudest_moment_any_of_its_instruments_reaches() -> None:
    """One level has to stand for the unity step, and it is the loudest the plan states anywhere."""
    assert loudest_db((_shared(_STRUCK), _shared(_QUIET), NO_SHAPE)) == pytest.approx(_STRUCK.peak)


def test_a_plan_whose_instruments_carry_no_shape_names_unity_itself() -> None:
    assert loudest_db((NO_SHAPE,)) == pytest.approx(0.0)


def test_two_instruments_written_against_one_level_stay_as_far_apart_as_their_shapes() -> None:
    """Normalising each curve against its own peak would move two velocity layers together by their difference."""
    peak_db = loudest_db((_shared(_STRUCK), _shared(_QUIET)))

    louder, quieter = (_written(curve, peak_db=peak_db).points[0].value for curve in (_STRUCK, _QUIET))

    assert gain_to_db(quieter / louder) == pytest.approx(_QUIET.peak - _STRUCK.peak, abs=0.2)


# --- what the written module carries -------------------------------------------------------------------------


def test_an_instrument_carrying_no_shape_is_written_without_an_envelope(export_context: ExportContext) -> None:
    """A slot the material plays no recorded key of leaves its voices at the level their material carries."""
    assert slot_envelope(NO_SHAPE, export_context, peak_db=0.0) is NO_ENVELOPE


def test_the_written_module_plays_a_looped_note_down_rather_than_ringing(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    """The whole gap this closes: a looped sample holds one level, so the module has to state the decline.

    The budget is tight enough that the bytes a loop saves are worth what holding a region at one level
    costs, which is the trade the solver declines wherever the material fits whole. Every instrument the
    material plays carries a curve, and the ones holding a loop are played down by theirs.
    """
    _, module = build(budget_kb=_LOOPED_BUDGET_KB)
    song = module.song
    envelopes = [instrument.volume_envelope for instrument in song.instruments]
    loops = [sample.loop is not None for sample in song.samples]

    assert any(loops), "the demo plan stores no loop, so this test would prove nothing"
    assert all(envelope is not None for envelope in envelopes)
    assert any(min(_shape_points(envelope)) < MAX_VOLUME for envelope in envelopes if envelope is not None)
