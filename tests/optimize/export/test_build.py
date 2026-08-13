from __future__ import annotations

from collections.abc import Callable

import pytest

from optisample.dsp.level import Clock, curve_level, gain_to_db, loudest_db, written_level
from optisample.dsp.piecewise import CurveNode, PiecewiseCurve
from optisample.dsp.trajectory import SharedTrajectory
from optisample.io.tracker.envelope import NO_ENVELOPE, EnvelopeGrid, volume_envelope
from optisample.optimize.export.build import slot_envelope, slot_level
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.voices import NO_SHAPE
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
_QUIET = PiecewiseCurve(  # a second instrument of the same plan, sitting well under the first
    nodes=(CurveNode(seconds=0.0, value=-18.0), CurveNode(seconds=1.0, value=-24.0))
)


def _shared(curve: PiecewiseCurve) -> SharedTrajectory:
    """One instrument's fitted shape, which is all a plan-wide level is read off."""
    return SharedTrajectory(curve=curve, offsets_db=((0.0,),), gaps_db=(0.0,))


def _grid() -> EnvelopeGrid:
    return EnvelopeGrid(tempo=_TEMPO, release_s=_RELEASE_S, tick_bound=_TICKS, value_bound=_VOLUME)


def _written(curve: PiecewiseCurve, *, reference_db: float) -> Envelope:
    return volume_envelope(written_level(curve_level(curve, Clock.PLAYED), reference_db=reference_db), _grid())


def _shape_points(envelope: Envelope) -> list[int]:
    """The steps the shape itself turns through, leaving out the breakpoint that lets a note go."""
    held = envelope.length if envelope.sustain is None else envelope.sustain.begin + 1
    return [point.value for point in envelope.points[:held]]


# --- what one level for a whole plan keeps ---------------------------------------------------------------------


def test_a_plan_is_written_against_the_loudest_moment_any_of_its_instruments_reaches() -> None:
    """One level has to stand for the unity step, and it is the loudest the plan states anywhere."""
    levels = [slot_level(shape) for shape in (_shared(_STRUCK), _shared(_QUIET), NO_SHAPE)]

    assert loudest_db([level for level in levels if level is not NO_SHAPE]) == pytest.approx(_STRUCK.peak)


def test_a_plan_whose_instruments_carry_no_shape_names_unity_itself() -> None:
    assert slot_level(NO_SHAPE) is NO_SHAPE
    assert loudest_db([]) == pytest.approx(0.0)


def test_two_instruments_written_against_one_level_stay_as_far_apart_as_their_shapes() -> None:
    """Normalising each curve against its own peak would move two velocity layers together by their difference."""
    reference_db = loudest_db([curve_level(curve, Clock.PLAYED) for curve in (_STRUCK, _QUIET)])

    louder, quieter = (_written(curve, reference_db=reference_db).points[0].value for curve in (_STRUCK, _QUIET))

    assert gain_to_db(quieter / louder) == pytest.approx(_QUIET.peak - _STRUCK.peak, abs=0.2)


# --- what the written module carries -------------------------------------------------------------------------


def test_an_instrument_carrying_no_shape_is_written_without_an_envelope(export_context: ExportContext) -> None:
    """A slot the material plays no recorded key of leaves its voices at the level their material carries."""
    assert slot_envelope(NO_SHAPE, export_context, reference_db=0.0) is NO_ENVELOPE


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
