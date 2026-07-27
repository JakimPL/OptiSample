from dataclasses import dataclass
from typing import Final, Literal, assert_never

import numpy as np
from numpy.typing import NDArray

from optisample.config.synth import SynthConfig
from optisample.music import A4_FREQ_HZ, MIDI_MAX_VELOCITY, midi_to_freq

Archetype = Literal["sustained", "piano"]

_MIN_PARTIALS: Final = 1
_FULL_TURN: Final = 2.0 * np.pi


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
    phases: NDArray[np.float64],
    config: SynthConfig,
) -> NDArray[np.float64]:
    """Synthesize a sustained tone: vibrato-modulated partials rolled off by velocity and controller.

    One partial is summed per entry of ``phases``, each starting at the angle that entry names. Partials
    at or above ``evolution_min_partial`` get a slow independent wax/wane, so the sustain drifts
    continuously; an attack/release envelope then shapes it and it is peak-normalized to the velocity's
    target level.
    """
    archetype_config = config.sustained
    fundamental = midi_to_freq(spec.pitch)
    time_axis = spec.time_axis()

    vibrato = 1.0 + archetype_config.vibrato_depth * np.sin(_FULL_TURN * archetype_config.vibrato_hz * time_axis)
    rolloff = (archetype_config.rolloff_base + archetype_config.rolloff_vel * spec.normalized_velocity) * (
        1.0 + archetype_config.rolloff_controller * spec.controller
    )

    signal = np.zeros_like(time_axis)
    for partial, offset in enumerate(phases, start=1):
        phase = _FULL_TURN * np.cumsum(partial * fundamental * vibrato) / spec.sample_rate + offset
        depth = archetype_config.evolution_depth if partial >= archetype_config.evolution_min_partial else 0.0
        evolution = 1.0 + depth * np.sin(_FULL_TURN * archetype_config.evolution_hz * time_axis + partial)
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


def render_piano(spec: NoteSpec, phases: NDArray[np.float64], config: SynthConfig) -> NDArray[np.float64]:
    """Synthesize a struck-string tone: inharmonic partials with frequency-dependent decay.

    One partial is summed per entry of ``phases``, each starting at the angle that entry names. Partial
    ``partial`` is stretched slightly sharp by ``inharmonicity`` (the stiff-string effect) and rings with
    a time constant that shortens for higher partials, so the tone darkens as it decays; a short attack
    ramp and peak-normalization to the velocity's target level finish it.
    """
    archetype_config = config.piano
    fundamental = midi_to_freq(spec.pitch)
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
    for partial, offset in enumerate(phases, start=1):
        freq = partial * fundamental * np.sqrt(1.0 + archetype_config.inharmonicity * partial * partial)
        tau = base_tau / (1.0 + archetype_config.decay_partial_factor * (partial - 1))
        phase = _FULL_TURN * freq * time_axis + offset
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
    archetype: Archetype, spec: NoteSpec, phases: NDArray[np.float64], config: SynthConfig
) -> NDArray[np.float64]:
    """Render ``spec`` with the synthesis routine for ``archetype``, starting from ``phases``."""
    match archetype:
        case "sustained":
            return render_sustained(spec, phases, config)
        case "piano":
            return render_piano(spec, phases, config)
        case _:  # pragma: no cover - archetype is an exhaustive Literal
            assert_never(archetype)


def _partial_count(archetype: Archetype, spec: NoteSpec, config: SynthConfig) -> int:
    """How many partials ``archetype`` sums for ``spec``, under its own requested count."""
    match archetype:
        case "sustained":
            requested = config.sustained.n_partials
        case "piano":
            requested = config.piano.n_partials
        case _:  # pragma: no cover - archetype is an exhaustive Literal
            assert_never(archetype)

    return _n_partials(midi_to_freq(spec.pitch), spec.sample_rate, requested, config.nyquist_fraction)


def draw_phases(
    archetype: Archetype, spec: NoteSpec, rng: np.random.Generator, config: SynthConfig
) -> NDArray[np.float64]:
    """The starting angle of every partial ``archetype`` sums for ``spec``, drawn from ``rng``.

    Drawing a note's phases up front leaves :func:`render_sample` a pure function of its inputs, so a
    note sounds the same in whichever process and whichever order it is rendered. Drawing them all from
    one stream, in the order the notes are listed, keeps a whole dataset reproducible from one seed.
    """
    return rng.uniform(0.0, _FULL_TURN, size=_partial_count(archetype, spec, config))


def synthesize(
    archetype: Archetype, spec: NoteSpec, rng: np.random.Generator, config: SynthConfig
) -> NDArray[np.float64]:
    """One note drawn from ``rng`` and rendered, for a caller wanting a note rather than a dataset.

    A dataset draws every note's phases first and renders them apart, which is what lets the rendering
    be shared out; a caller after a single note has the two steps here in one place.
    """
    return render_sample(archetype, spec, draw_phases(archetype, spec, rng, config), config)
