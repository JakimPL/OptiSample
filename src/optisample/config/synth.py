from typing import Literal

from optisample.config.base import ConfigModel

Archetype = Literal["sustained", "piano"]


class SustainedConfig(ConfigModel):
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


class PianoConfig(ConfigModel):
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


class MaterialEvent(ConfigModel):
    """One material row: a (pitch, velocity, duration, repeat count) the demo song plays."""

    pitch: int
    velocity: int
    duration_s: float
    count: int


class PresetConfig(ConfigModel):
    """One demo instrument: its archetype, recorded (pitch, velocity) grid, durations, budget, material."""

    id: str
    archetype: Archetype
    pitches: tuple[int, ...]
    velocities: tuple[int, ...]
    sample_duration: float
    budget_kb: float
    material: tuple[MaterialEvent, ...]


class SynthConfig(ConfigModel):
    """The synthetic generator: render rate, anti-alias/velocity mapping, archetypes, and presets."""

    sample_rate: int
    nyquist_fraction: float
    velocity_peak_floor: float
    velocity_peak_scale: float
    sustained: SustainedConfig
    piano: PianoConfig
    presets: tuple[PresetConfig, ...]
