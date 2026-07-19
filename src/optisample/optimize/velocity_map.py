"""Derive the velocity->volume map: a conversion-time artifact, not stored in the IT file.

IT has no velocity layers -- one stored sample serves every dynamic of a key, and loudness comes
from the note volume (``0..64``, *linear* in amplitude). So the map answers: to make MIDI velocity
``v`` play back as loud as the recording at ``v``, what note volume do we write into the pattern?

We measure each recording's integrated loudness (BS.1770 / LUFS), anchor the loudest velocity at
full volume (64), and set the rest by the amplitude ratio ``10**((L_v - L_ref)/20)``. Because that
ratio is exponential in a linear-in-dB loudness curve, the map is inherently non-linear. The sparse
per-velocity anchors are interpolated (in dB, with flat extrapolation) across the full 0..127 range.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.optimize import VelocityConfig
from optisample.dsp.surrogate import MAX_VOLUME
from optisample.metrics.base import Signal
from optisample.metrics.preprocess import integrated_loudness

_MIDI_VELOCITIES = 128


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
            raise ValueError(f"velocity must be in [0, 127], got {velocity}")
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


def derive_velocity_map(
    loudness: Mapping[int, float], config: VelocityConfig, *, max_volume: int = MAX_VOLUME
) -> VelocityVolumeMap:
    """Build a loudness-matched velocity->volume map from per-velocity loudness measurements."""
    if not loudness:
        raise ValueError("need at least one velocity measurement")
    anchors_in = sorted(loudness.items())
    velocities = np.array([velocity for velocity, _ in anchors_in], dtype=np.float64)
    measured = np.array([loud for _, loud in anchors_in], dtype=np.float64)
    if not np.any(np.isfinite(measured)):
        return _silent_map(anchors_in)

    reference = float(np.max(measured[np.isfinite(measured)]))
    clamped = np.maximum(measured, reference - config.loudness_floor_lu)  # -inf (silence) -> the floor
    grid = np.arange(_MIDI_VELOCITIES, dtype=np.float64)
    interpolated = np.interp(grid, velocities, clamped)  # flat extrapolation beyond the anchors
    gains = 10.0 ** ((interpolated - reference) / 20.0)
    volumes = tuple(int(np.clip(round(float(gain) * max_volume), 0, max_volume)) for gain in gains)
    anchors = tuple(VelocityAnchor(int(velocity), float(loud), volumes[int(velocity)]) for velocity, loud in anchors_in)
    return VelocityVolumeMap(volumes, anchors)
