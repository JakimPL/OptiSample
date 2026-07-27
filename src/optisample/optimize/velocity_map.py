from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.optimize import VelocityConfig
from optisample.dsp.levels import db_to_gain
from optisample.metrics.base import Signal
from optisample.metrics.preprocess import integrated_loudness
from optisample.music import MIDI_MAX_VELOCITY
from trackmod.spec.levels import MAX_VOLUME

_MIDI_VELOCITIES: Final = MIDI_MAX_VELOCITY + 1


@dataclass(frozen=True)
class VelocityAnchor:
    """A measured velocity: its integrated loudness and the note volume the map assigns it."""

    velocity: int
    loudness_lufs: float
    volume: int


@dataclass(frozen=True)
class VelocityVolumeMap:
    """MIDI velocity (0..127) -> IT note volume (0..64), with the measured anchors kept for reporting."""

    volumes: tuple[int, ...]  # length 128, indexed by MIDI velocity
    anchors: tuple[VelocityAnchor, ...]

    def volume(self, velocity: int) -> int:
        """Note volume for ``velocity`` (0..127)."""
        if not 0 <= velocity < _MIDI_VELOCITIES:
            raise ValueError(f"velocity must be in [0, {MIDI_MAX_VELOCITY}], got {velocity}")

        return self.volumes[velocity]


def loudness_by_velocity(clips: Sequence[tuple[int, Signal]], sample_rate: int) -> dict[int, float]:
    """Mean integrated loudness (LUFS) of the recordings at each velocity (silence -> ``-inf``)."""
    grouped: dict[int, list[float]] = {}
    for velocity, signal in clips:
        grouped.setdefault(velocity, []).append(integrated_loudness(signal, sample_rate))

    result: dict[int, float] = {}
    for velocity, values in grouped.items():
        finite = [value for value in values if np.isfinite(value)]
        result[velocity] = float(np.mean(finite)) if finite else -np.inf

    return result


def _silent_map(anchors: Sequence[tuple[int, float]]) -> VelocityVolumeMap:
    """Everything at zero volume -- used when no velocity has measurable loudness."""
    volumes = tuple(0 for _ in range(_MIDI_VELOCITIES))
    return VelocityVolumeMap(volumes, tuple(VelocityAnchor(int(v), float(loud), 0) for v, loud in anchors))


def _reference_loudness(measured: NDArray[np.float64]) -> float:
    """The loudness the map anchors to full volume: the loudest measured velocity."""
    return float(np.max(measured[np.isfinite(measured)]))


def _clamp_to_floor(
    measured: NDArray[np.float64],
    reference: float,
    floor_lu: float,
) -> NDArray[np.float64]:
    """Raise every anchor to at least ``reference - floor_lu`` so silence maps to a defined quietest level.

    Silent velocities measure ``-inf``; the floor lifts them (and any near-silent ones) to a bounded
    quietest level, keeping the map within ``floor_lu`` of the loudest recording.
    """
    return np.maximum(measured, reference - floor_lu)


def _interpolate_over_velocities(
    velocities: NDArray[np.float64],
    loudness: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Loudness for every MIDI velocity 0..127, linearly interpolated in dB between the sparse anchors.

    ``np.interp`` extrapolates flat beyond the measured anchors (holding the nearest endpoint), so
    velocities outside the recorded range reuse the closest measured loudness, holding the edges at a
    plausible level.
    """
    grid = np.arange(_MIDI_VELOCITIES, dtype=np.float64)
    return np.interp(grid, velocities, loudness)


def _gains_to_volumes(
    loudness: NDArray[np.float64],
    reference: float,
    max_volume: int,
) -> tuple[int, ...]:
    """Turn per-velocity loudness (dB) into IT note volumes matched to the reference's amplitude.

    Note volume is linear in amplitude, so a velocity ``d`` dB below the reference (``d = loudness -
    reference <= 0``) must play at the amplitude ratio :func:`~optisample.dsp.levels.db_to_gain`
    gives; that ratio scaled to ``max_volume`` and rounded is the volume written into the pattern.
    """
    gains = db_to_gain(loudness - reference)
    return tuple(int(np.clip(round(float(gain) * max_volume), 0, max_volume)) for gain in gains)


def derive_velocity_map(
    loudness: Mapping[int, float],
    config: VelocityConfig,
    *,
    max_volume: int = MAX_VOLUME,
) -> VelocityVolumeMap:
    """Build a loudness-matched velocity->volume map from per-velocity loudness measurements.

    Steps: anchor the loudest velocity at full volume (:func:`_reference_loudness`), floor the silent
    ones (:func:`_clamp_to_floor`), fill in every velocity 0..127 (:func:`_interpolate_over_velocities`),
    and convert the resulting dB curve to note volumes (:func:`_gains_to_volumes`).
    """
    if not loudness:
        raise ValueError("need at least one velocity measurement")

    anchors_in = sorted(loudness.items())
    velocities = np.array([velocity for velocity, _ in anchors_in], dtype=np.float64)
    measured = np.array([loud for _, loud in anchors_in], dtype=np.float64)
    if not np.any(np.isfinite(measured)):
        return _silent_map(anchors_in)

    reference = _reference_loudness(measured)
    clamped = _clamp_to_floor(measured, reference, config.loudness_floor_lu)
    interpolated = _interpolate_over_velocities(velocities, clamped)
    volumes = _gains_to_volumes(interpolated, reference, max_volume)
    anchors = tuple(VelocityAnchor(int(velocity), float(loud), volumes[int(velocity)]) for velocity, loud in anchors_in)
    return VelocityVolumeMap(volumes, anchors)
