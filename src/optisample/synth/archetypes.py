from dataclasses import dataclass
from typing import Final, Literal, assert_never

import numpy as np
from numpy.typing import NDArray

from optisample.config.synth import SynthConfig
from optisample.music import A4_FREQ_HZ, MIDI_MAX_VELOCITY, midi_to_freq

Archetype = Literal["sustained", "piano"]

_MIN_PARTIALS: Final = 1


@dataclass(frozen=True)
class NoteSpec:
    """A single note to render, independent of archetype."""

    pitch: int
    velocity: int
    controller: float
    duration_s: float
    sample_rate: int

    @property
    def normalized_velocity(self) -> float:
        """Velocity mapped into ``[0, 1]``."""
        return self.velocity / MIDI_MAX_VELOCITY

    def time_axis(self) -> NDArray[np.float64]:
        """The per-sample time grid (seconds) spanning the note's duration."""
        return np.arange(int(self.duration_s * self.sample_rate), dtype=np.float64) / self.sample_rate


def _n_partials(
    fundamental: float,
    sample_rate: int,
    requested: int,
    nyquist_fraction: float,
) -> int:
    """Cap the partial count so the highest partial stays below Nyquist (no reference aliasing)."""
    ceiling = int(nyquist_fraction * sample_rate / fundamental)
    return max(_MIN_PARTIALS, min(requested, ceiling))


def _velocity_peak(velocity: int, floor: float, scale: float) -> float:
    """Map velocity → target peak amplitude (monotonic, stays below full scale)."""
    return floor + scale * (velocity / MIDI_MAX_VELOCITY)


def _normalize_peak(
    signal: NDArray[np.float64],
    peak: float,
) -> NDArray[np.float64]:
    """Scale ``signal`` so its largest magnitude equals ``peak`` (silence passes through unchanged)."""
    largest = float(np.max(np.abs(signal)))
    if largest == 0.0:
        return signal

    return np.asarray(signal / largest * peak, dtype=np.float64)


def _attack_release(
    time_axis: NDArray[np.float64],
    attack_s: float,
    release_s: float,
) -> NDArray[np.float64]:
    """Linear attack/release envelope over ``time_axis`` (unity across the sustain in between)."""
    attack = np.minimum(1.0, time_axis / attack_s)
    total = float(time_axis[-1]) if time_axis.size else 0.0
    release = np.minimum(1.0, np.maximum(0.0, (total - time_axis) / release_s))
    return np.asarray(attack * release, dtype=np.float64)


def render_sustained(
    spec: NoteSpec,
    rng: np.random.Generator,
    config: SynthConfig,
) -> NDArray[np.float64]:
    """Synthesize a sustained tone: vibrato-modulated partials rolled off by velocity and controller.

    Partials at or above ``evolution_min_partial`` get a slow independent wax/wane, so the sustain
    drifts continuously; an attack/release envelope then shapes it and it is peak-normalized to the
    velocity's target level.
    """
    archetype_config = config.sustained
    fundamental = midi_to_freq(spec.pitch)
    partials = _n_partials(fundamental, spec.sample_rate, archetype_config.n_partials, config.nyquist_fraction)
    time_axis = spec.time_axis()

    vibrato = 1.0 + archetype_config.vibrato_depth * np.sin(2.0 * np.pi * archetype_config.vibrato_hz * time_axis)
    rolloff = (archetype_config.rolloff_base + archetype_config.rolloff_vel * spec.normalized_velocity) * (
        1.0 + archetype_config.rolloff_controller * spec.controller
    )

    signal = np.zeros_like(time_axis)
    for partial in range(1, partials + 1):
        phase = 2.0 * np.pi * np.cumsum(partial * fundamental * vibrato) / spec.sample_rate + rng.uniform(
            0.0, 2.0 * np.pi
        )
        depth = archetype_config.evolution_depth if partial >= archetype_config.evolution_min_partial else 0.0
        evolution = 1.0 + depth * np.sin(2.0 * np.pi * archetype_config.evolution_hz * time_axis + partial)
        signal += rolloff ** (partial - 1) / partial * evolution * np.sin(phase)

    signal *= _attack_release(time_axis, attack_s=archetype_config.attack_s, release_s=archetype_config.release_s)
    return _normalize_peak(
        signal,
        _velocity_peak(
            spec.velocity,
            config.velocity_peak_floor,
            config.velocity_peak_scale,
        ),
    )


def render_piano(spec: NoteSpec, rng: np.random.Generator, config: SynthConfig) -> NDArray[np.float64]:
    """Synthesize a struck-string tone: inharmonic partials with frequency-dependent decay.

    Partial ``partial`` is stretched slightly sharp by ``inharmonicity`` (the stiff-string effect) and
    rings with a time constant that shortens for higher partials, so the tone darkens as it decays; a
    short attack ramp and peak-normalization to the velocity's target level finish it.
    """
    archetype_config = config.piano
    fundamental = midi_to_freq(spec.pitch)
    partials = _n_partials(
        fundamental,
        spec.sample_rate,
        archetype_config.n_partials,
        config.nyquist_fraction,
    )
    time_axis = spec.time_axis()

    base_tau = float(
        np.clip(
            archetype_config.base_tau_scale * (A4_FREQ_HZ / fundamental) ** archetype_config.base_tau_exp,
            archetype_config.base_tau_min,
            archetype_config.base_tau_max,
        )
    )
    rolloff = (archetype_config.rolloff_base + archetype_config.rolloff_vel * spec.normalized_velocity) * (
        1.0 + archetype_config.rolloff_controller * spec.controller
    )
    signal = np.zeros_like(time_axis)
    for partial in range(1, partials + 1):
        freq = partial * fundamental * np.sqrt(1.0 + archetype_config.inharmonicity * partial * partial)
        tau = base_tau / (1.0 + archetype_config.decay_partial_factor * (partial - 1))
        phase = 2.0 * np.pi * freq * time_axis + rng.uniform(0.0, 2.0 * np.pi)
        signal += rolloff ** (partial - 1) / partial * np.exp(-time_axis / tau) * np.sin(phase)

    signal *= np.minimum(1.0, time_axis / archetype_config.attack_s)
    return _normalize_peak(
        signal,
        _velocity_peak(
            spec.velocity,
            config.velocity_peak_floor,
            config.velocity_peak_scale,
        ),
    )


def render_sample(
    archetype: Archetype, spec: NoteSpec, rng: np.random.Generator, config: SynthConfig
) -> NDArray[np.float64]:
    """Render ``spec`` with the synthesis routine for ``archetype``."""
    match archetype:
        case "sustained":
            return render_sustained(spec, rng, config)
        case "piano":
            return render_piano(spec, rng, config)
        case _:  # pragma: no cover - archetype is an exhaustive Literal
            assert_never(archetype)
