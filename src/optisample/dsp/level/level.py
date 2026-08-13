from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.dsp.level.clock import Clock, played_scale
from optisample.dsp.level.decibels import Signal, db_to_gain, gain_to_db
from optisample.dsp.piecewise import PiecewiseCurve, fit_piecewise
from optisample.dsp.series import Readings, Series

UNITY_DB: Final = 0.0  # the level a gain of one reads at, which is what a transparent level holds throughout

_ONE_MOMENT: Final = 0.0  # the moment a level holding one value throughout states that value at


@dataclass(frozen=True)
class Level:
    """A gain over time, held in decibels on the clock its moments are counted on.

    Everything in this pipeline that scales audio is one of these: the smooth amplitude a recording moves
    through, the curve an instrument plays its voices down by, the step written beside a sample, the volume
    a pattern states for a velocity. Holding them as one value is what lets them compose instead of being
    re-derived -- a waveform divided by a level and multiplied by it again is the waveform it started as.

    ``readings`` states the level at the moments it turns at, running straight between each pair and level
    past both ends, which is exactly what a tracker's envelope does with its own breakpoints. A level
    holding one value throughout states it at a single moment, so the common case stays one number rather
    than an array per frame -- what keeps carrying a level everywhere affordable.

    Decibels are the domain because that is where gains add, where a ringing note falls straight, and where
    every reading of this material is already taken. Composition is therefore addition, and it is total:
    every level reaching here is floored strictly positive by the reading that produced it, so there is no
    silence to take a logarithm of.
    """

    readings: Readings
    clock: Clock

    @property
    def peak_db(self) -> float:
        """The loudest level reached, which is what a written form states its shape against."""
        return float(np.max(self.readings.values))

    @property
    def transparent(self) -> bool:
        """Whether this level leaves what it multiplies exactly as it stands."""
        return bool(np.all(self.readings.values == UNITY_DB))

    def db(self, seconds: Series) -> Series:
        """The level in decibels at each moment in ``seconds``, held at its end values on either side."""
        return np.asarray(
            np.interp(seconds, self.readings.seconds, self.readings.values),
            dtype=np.float64,
        )

    def gain(self, seconds: Series) -> Series:
        """The amplitude this level multiplies by at each moment in ``seconds``."""
        return db_to_gain(self.db(seconds))

    def frame_gains(self, frames: int, sample_rate: int) -> Series:
        """The amplitude each of the first ``frames`` frames is multiplied by, at ``sample_rate``.

        The accessor a waveform is scaled through, so a caller dividing a recording by its level and one
        playing a sample back through it read the very same curve.
        """
        return self.gain(np.arange(frames, dtype=np.float64) / sample_rate)

    def times(self, other: Level) -> Level:
        """This level multiplied by ``other``, which is their decibels added.

        Both run straight between their own moments, so the product turns at the moments either of them
        does and the result is exact rather than resampled.

        Raises:
            ValueError: when the two are counted on different clocks, whose moments state different things.
        """
        moments = _shared_moments(self, other)
        return Level(
            readings=Readings(values=self.db(moments) + other.db(moments), seconds=moments),
            clock=self.clock,
        )

    def over(self, other: Level) -> Level:
        """This level divided by ``other`` -- what is left for something else to supply.

        Raises:
            ValueError: when the two are counted on different clocks, whose moments state different things.
        """
        moments = _shared_moments(self, other)
        return Level(
            readings=Readings(values=self.db(moments) - other.db(moments), seconds=moments),
            clock=self.clock,
        )

    def scaled_db(self, delta_db: float) -> Level:
        """This level shifted by a constant number of decibels, turning where it already turns."""
        return Level(
            readings=Readings(values=self.readings.values + delta_db, seconds=self.readings.seconds),
            clock=self.clock,
        )

    def at_pitch(self, *, root_pitch: int, pitch: int) -> Level:
        """This recorded level as the played note sounding it at ``pitch`` makes it.

        Repitching stretches the material's own timeline (:func:`~optisample.dsp.level.clock.played_scale`)
        while leaving every level it passes through as it was, so the moments move and the values stand.

        Raises:
            ValueError: when the level is already counted on the played clock, which repitching leaves
                where it is.
        """
        if self.clock is not Clock.RECORDED:
            raise ValueError(f"a played level is sounded as it stands, against the {self.clock.value} one asked for")

        scale = played_scale(root_pitch=root_pitch, pitch=pitch)
        return Level(
            readings=Readings(values=self.readings.values, seconds=self.readings.seconds * scale),
            clock=Clock.PLAYED,
        )

    def fitted(self, *, nodes: int) -> Level:
        """This level stated in at most ``nodes`` corners, which is what a format has room to write.

        Raises:
            ValueError: when fewer than two nodes are asked for, when the level holds nothing to fit, or
                when it turns at more moments than :data:`~optisample.dsp.piecewise.MOST_READINGS`.
        """
        return curve_level(fit_piecewise(self.readings, nodes=nodes), self.clock)

    def distance_db(self, other: Level) -> float:
        """How far this level stands from ``other`` at the moment they stand furthest apart.

        Both run straight between their moments, so their difference does too and its widest point is one
        of those moments -- which makes the reading exact rather than sampled. This is what one shared
        curve costs the levels written under it.

        Raises:
            ValueError: when the two are counted on different clocks, whose moments state different things.
        """
        return float(np.max(np.abs(self.over(other).readings.values)))


def _shared_moments(one: Level, other: Level) -> Series:
    """Every moment either level turns at, ascending, which is where their composition turns.

    Raises:
        ValueError: when the two are counted on different clocks, whose moments state different things.
    """
    if one.clock is not other.clock:
        raise ValueError(f"levels compose on one clock, against the {one.clock.value} and {other.clock.value} stated")

    return np.asarray(np.union1d(one.readings.seconds, other.readings.seconds), dtype=np.float64)


def constant_level(level_db: float, clock: Clock) -> Level:
    """A level holding ``level_db`` for as long as anything asks it, stated at a single moment."""
    return Level(
        readings=Readings(
            values=np.asarray([level_db], dtype=np.float64),
            seconds=np.asarray([_ONE_MOMENT], dtype=np.float64),
        ),
        clock=clock,
    )


def unit_level(clock: Clock) -> Level:
    """The level that leaves what it multiplies as it stands, which is what a caller starts composing from."""
    return constant_level(UNITY_DB, clock)


def gain_level(gain: float, clock: Clock) -> Level:
    """A constant level stated as the amplitude it multiplies by rather than as decibels."""
    return constant_level(gain_to_db(gain), clock)


def read_level(readings: Readings, clock: Clock) -> Level:
    """Level readings already taken in decibels, carried as the level they state."""
    return Level(readings=readings, clock=clock)


def curve_level(curve: PiecewiseCurve, clock: Clock) -> Level:
    """A fitted curve of decibel corners carried as the level it states."""
    return Level(readings=Readings(values=curve.values, seconds=curve.seconds), clock=clock)


def sampled_level(amplitudes: Signal, sample_rate: int, clock: Clock) -> Level:
    """A per-frame amplitude curve carried as the level it states, one moment per frame.

    This is how the smooth amplitude a recording moves through
    (:attr:`~optisample.dsp.envelope.Decomposition.level`) enters the algebra. It turns at every frame, so
    a caller holding it for longer than one composition is better served fitting it first
    (:meth:`Level.fitted`).
    """
    data = np.asarray(amplitudes, dtype=np.float64)
    return Level(
        readings=Readings(
            values=gain_to_db(data),
            seconds=np.arange(data.size, dtype=np.float64) / sample_rate,
        ),
        clock=clock,
    )


def product(levels: Sequence[Level]) -> Level:
    """Every level in ``levels`` multiplied together, which is what a voice arrives at.

    A tracker multiplies several levels onto one note -- the waveform's own, the step beside its sample,
    the volume the pattern states, the curve its instrument plays -- so what the listener hears is this.

    Raises:
        ValueError: when the set holds nothing, or when its levels are counted on different clocks.
    """
    if not levels:
        raise ValueError("a product states at least one level")

    composed = levels[0]
    for level in levels[1:]:
        composed = composed.times(level)

    return composed
