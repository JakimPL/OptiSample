from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.dsp.levels import db_to_gain, gain_to_db
from optisample.dsp.piecewise import PiecewiseCurve
from optisample.dsp.series import Series
from optisample.io.tracker.target import ExportTarget
from trackmod.core.envelopes.curve import Breakpoint, placed_ticks, timed_envelope
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.span import EnvelopeSpan
from trackmod.core.timing.clock import tick_seconds
from trackmod.limits.bound import Bound
from trackmod.spec.levels import MAX_VOLUME, MIN_VOLUME

NO_ENVELOPE: Final = None  # what an instrument whose samples carry every level they play at leaves behind

_ONSET_S: Final = 0.0
_RELEASE_POINTS: Final = 1  # points of a format's own budget the breakpoint carrying a note to silence spends
_MOST_MOMENTS: Final = 2048  # moments a written curve is priced at, which bounds what one repair sweep reads
_MOST_SWEEPS: Final = 8  # sweeps the nodes are given to settle onto their steps, which a couple of them spend
_STEPS: Final = tuple(range(MIN_VOLUME, MAX_VOLUME + 1))  # every level one envelope node states
QUIETEST_STEP: Final = 1  # the softest step a node holds short of silencing the voice outright

_FLOOR_DB: Final = gain_to_db(QUIETEST_STEP / MAX_VOLUME)  # how far under unity a written curve reaches


@dataclass(frozen=True)
class EnvelopeGrid:
    """What a format gives a written curve: the clock it counts in, its two grids, and how a note is let go.

    ``tempo`` settles how long a tick lasts, so a curve holds only for the clock it was written for.
    ``release_s`` is the one part of the curve a recording never states, since a recording is a note held
    to its end.
    """

    tempo: int
    release_s: float
    tick_bound: Bound
    value_bound: Bound


def envelope_grid(target: ExportTarget, *, tempo: int, release_s: float) -> EnvelopeGrid:
    """What a curve written for ``target`` is held to: its two grids, the clock, and how a note is let go.

    The format decides the grids and the caller the clock, so one instrument written beside a module and
    one written on its own from the same curve land on the same ticks and the same steps.
    """
    return EnvelopeGrid(
        tempo=tempo,
        release_s=release_s,
        tick_bound=target.envelope_tick_bound,
        value_bound=target.envelope_value_bound,
    )


def shape_nodes(point_bound: Bound) -> int:
    """How many corners a fitted shape may turn through in a format numbering ``point_bound`` envelope points.

    The written curve is the shape plus the breakpoint carrying a released note to silence, so the shape
    itself is given every point the format numbers less that one -- twenty-four for Impulse Tracker,
    eleven for FastTracker 2.
    """
    return point_bound.maximum - _RELEASE_POINTS


def played_gain(envelope: Envelope, *, tempo: int, frames: int, sample_rate: int) -> Series:
    """What ``envelope`` multiplies a held voice by at each of ``frames``, on the clock ``tempo`` runs.

    A tracker walks straight between two breakpoints in amplitude and holds the last one it reaches for
    as long as the note is held, which is what a note sounding past the corner the shape closes on plays
    at. Reading the curve this way states the gain the format actually applies, so a waveform divided by
    it comes back out at the level the recording held.

    The breakpoints up to the sustain point are the shape itself; the one past it carries a released note
    to silence, so a note still sounding never reaches it.
    """
    shape = envelope.points[: envelope.length if envelope.sustain is None else envelope.sustain.begin + 1]
    length = tick_seconds(tempo)
    seconds = np.arange(frames, dtype=np.float64) / sample_rate
    moments = np.asarray([point.tick * length for point in shape], dtype=np.float64)
    values = np.asarray([point.value / MAX_VOLUME for point in shape], dtype=np.float64)
    return np.asarray(np.interp(seconds, moments, values), dtype=np.float64)


def sounding_gain(envelope: Envelope, *, tempo: int, frames: int, sample_rate: int) -> Series:
    """:func:`played_gain` held at the quietest step that still sounds, so a division by it stays finite.

    A moment the curve silences outright plays as silence whatever the waveform holds there, so a
    recording divided by the gain its envelope applies is divided by this instead and the waveform stays
    finite throughout.
    """
    gain = played_gain(envelope, tempo=tempo, frames=frames, sample_rate=sample_rate)
    return np.maximum(gain, QUIETEST_STEP / MAX_VOLUME)


def _grid_value(level_db: float) -> int:
    """The step a node holds ``level_db`` below unity at, on the format's own 0-64 amplitude grid.

    The grid counts in amplitude, so its steps stand a fraction of a decibel apart at the top and several
    decibels apart near the floor -- which is what the repair sweep is for.
    """
    return min(MAX_VOLUME, max(MIN_VOLUME, round(MAX_VOLUME * db_to_gain(level_db))))


def _played_db(ticks: Series, values: Series, moments: Series) -> Series:
    """The level a curve of ``values`` at ``ticks`` puts out at each moment, in decibels.

    A tracker walks between two nodes in amplitude while a fitted shape turns in decibels, so what the
    format actually plays is read back here and the nodes are chosen against it.
    """
    return gain_to_db(np.asarray(np.interp(moments, ticks, values), dtype=np.float64) / MAX_VOLUME)


def _moments(ticks: Series) -> Series:
    """The ticks a written curve is priced at: the ones its shape spans, thinned to what a sweep reads."""
    span = int(ticks[-1] - ticks[0]) + 1
    return np.linspace(ticks[0], ticks[-1], num=min(span, _MOST_MOMENTS), dtype=np.float64)


def _settled_value(index: int, ticks: Series, values: Series, target_db: Series, moments: Series) -> int:
    """The step the node at ``index`` holds to run the played curve closest to the shape it stands for.

    Moving one node reaches only the two straight runs meeting at it, so the choice is priced over that
    stretch alone and every step the grid offers is tried on it. A node its neighbours leave no moment
    between stays where it is, there being nothing there to price it against.
    """
    opening, closing = max(index - 1, 0), min(index + 1, values.size - 1)
    held = (moments >= ticks[opening]) & (moments <= ticks[closing])
    if not np.any(held):
        return int(values[index])

    reach, stated = ticks[opening : closing + 1], values[opening : closing + 1].copy()
    errors = []
    for step in _STEPS:
        stated[index - opening] = step
        errors.append(float(np.sum((_played_db(reach, stated, moments[held]) - target_db[held]) ** 2)))

    return MIN_VOLUME + int(np.argmin(errors))


def _settled_values(ticks: Series, values: Series, target_db: Series, moments: Series, *, loudest: int) -> Series:
    """Every node moved onto the step the format offers it, one at a time, until a sweep leaves them alone.

    Two things separate a written curve from the shape it stands for: the 0-64 grid a level is rounded
    onto, which is coarse in decibels wherever the curve is quiet, and the straight run a tracker walks
    between two nodes in amplitude where the shape turns in decibels. Both are answered the same way --
    with every other node held still, one node has a best step and trying them all finds it -- and each
    step lowers the same squared error, so the set settles.

    The node at ``loudest`` keeps the step it was rounded onto throughout. That is where the instrument
    states its own level, which the gains beside the envelope were staged against, so the sweep leaves it
    alone and spends the steps under it on the shape.
    """
    settled = values.copy()
    for _ in range(_MOST_SWEEPS):
        moved = False
        for index in range(settled.size):
            if index == loudest:
                continue

            step = _settled_value(index, ticks, settled, target_db, moments)
            moved = moved or step != settled[index]
            settled[index] = step

        if not moved:
            break

    return settled


def _shape_points(curve: PiecewiseCurve, peak_db: float) -> tuple[Breakpoint, ...]:
    """``curve``'s corners as breakpoints, stated against ``peak_db`` and read from the onset.

    A volume envelope multiplies, so the loudest moment written anywhere in a plan is the unity step and
    every other moment attenuates from it. Stating every instrument against that one level is what keeps
    the balance the gains staged: two curves normalised each against its own peak would move apart by
    exactly the difference between them. The first corner sits at the onset, since the curve holds its
    opening value for everything before it.
    """
    levels_db = curve.values - peak_db
    seconds = (_ONSET_S, *(float(moment) for moment in curve.seconds[1:]))
    return tuple(
        Breakpoint(seconds=moment, value=_grid_value(float(level))) for moment, level in zip(seconds, levels_db)
    )


def volume_envelope(curve: PiecewiseCurve, grid: EnvelopeGrid, *, peak_db: float) -> Envelope:
    """``curve`` written as the envelope an instrument plays its voices down by, on ``grid``.

    The shape arrives in decibels on the played clock (:func:`~optisample.dsp.piecewise.fit_piecewise`)
    and the format stores whole ticks and 0-64 amplitude steps, so writing it is two roundings: each
    corner onto the tick it falls on, each level onto the step nearest it. What those roundings cost is
    then repaired by moving one node at a time onto the step running the played curve closest to the
    shape (:func:`_settled_values`), which also answers the straight runs a tracker walks in amplitude
    where the shape turns in decibels.

    The last corner is the sustain point, so a held note stays at the level the shape reached and a
    released one goes on to a final breakpoint ``release_s`` later and dies. Both formats sustain on a
    single point, so the span names one.

    The release is spelled as a breakpoint rather than as an instrument fadeout because the two formats
    begin a fade in different places -- FastTracker 2 at the key off, Impulse Tracker where the volume
    envelope ends -- while a curve reaching zero says the same thing to both
    (:mod:`trackmod.core.instruments.fade`).

    Ticks are what a format counts envelope time in, so the curve holds only for the tempo it was written
    for -- which is why that tempo travels with a bank
    (:class:`~optisample.artifacts.documents.bank.BankDocument`).

    Raises:
        ValueError: when the corners and the release outnumber the ticks ``grid`` counts.
    """
    shape = _shape_points(curve, peak_db)
    release = Breakpoint(seconds=float(curve.seconds[-1]) + grid.release_s, value=MIN_VOLUME)
    placed = placed_ticks((*shape, release), tempo=grid.tempo, bound=grid.tick_bound)
    ticks = np.asarray(placed, dtype=np.float64)[: len(shape)]
    moments = _moments(ticks)
    target_db = np.maximum(curve.at(moments * tick_seconds(grid.tempo)) - peak_db, _FLOOR_DB)
    settled = _settled_values(
        ticks,
        np.asarray([point.value for point in shape], dtype=np.float64),
        target_db,
        moments,
        loudest=int(np.argmax(curve.values)),
    )
    held = len(shape) - 1
    return timed_envelope(
        (*(Breakpoint(seconds=point.seconds, value=int(step)) for point, step in zip(shape, settled)), release),
        tempo=grid.tempo,
        tick_bound=grid.tick_bound,
        value_bound=grid.value_bound,
        sustain=EnvelopeSpan(begin=held, end=held),
    )
