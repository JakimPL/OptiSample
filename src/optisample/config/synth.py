"""Synthetic-dataset configuration: archetype synthesis parameters and instrument presets.

The synthetic generator is how the pipeline is exercised without real recordings, so its shape is a
tunable too: the ``sustained`` and ``piano`` archetype coefficients (partials, vibrato/evolution,
spectral rolloff, decay) and the ``presets`` (pitches, velocities, sample duration, budget, material).
"""

from __future__ import annotations

from typing import Literal

from optisample.config.base import ConfigModel

Archetype = Literal["sustained", "piano"]


class SustainedConfig(ConfigModel):  # pylint: disable=too-many-instance-attributes
    """Strings/pad archetype: partial count, vibrato, slow spectral evolution, spectral rolloff, A/R."""

    n_partials: int
    vibrato_hz: float
    vibrato_depth: float
    rolloff_base: float
    rolloff_vel: float
    rolloff_controller: float
    evolution_depth: float
    evolution_hz: float
    evolution_min_partial: int
    attack_s: float
    release_s: float


class PianoConfig(ConfigModel):  # pylint: disable=too-many-instance-attributes
    """Decaying-one-shot archetype: partial count, per-partial decay, inharmonicity, rolloff, attack."""

    n_partials: int
    base_tau_scale: float
    base_tau_exp: float
    base_tau_min: float
    base_tau_max: float
    rolloff_base: float
    rolloff_vel: float
    rolloff_controller: float
    decay_partial_factor: float
    inharmonicity: float
    attack_s: float


class PresetConfig(ConfigModel):  # pylint: disable=too-many-instance-attributes
    """One demo instrument: its archetype, recorded (pitch, velocity) grid, durations, budget, material."""

    id: str
    archetype: Archetype
    pitches: tuple[int, ...]
    velocities: tuple[int, ...]
    sample_dur: float
    budget_kb: float
    material: tuple[tuple[int, int, float, int], ...]  # (pitch, velocity, duration_s, count)


class SynthConfig(ConfigModel):  # pylint: disable=too-many-instance-attributes
    """The synthetic generator: render rate, anti-alias/velocity mapping, archetypes, and presets."""

    sample_rate: int
    nyquist_fraction: float
    velocity_peak_floor: float
    velocity_peak_scale: float
    sustained: SustainedConfig
    piano: PianoConfig
    presets: tuple[PresetConfig, ...]
