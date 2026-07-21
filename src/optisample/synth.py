"""Deterministic synthetic instruments so the pipeline is testable without real recordings.

Two archetypes are produced, matching the POC targets:

* ``sustained`` (strings/pad-like): slow attack, an evolving sustain with slight vibrato,
  release. The sustain *evolves* over time on purpose, so a too-short loop is measurably
  static later on.
* ``piano`` (decaying one-shot): fast attack, per-partial exponential decay (high partials
  decay faster → the tone darkens as it rings), mild inharmonicity.

For both, higher velocity is rendered *louder and brighter* — so emulating velocity by
volume alone is genuinely lossy, which is exactly the trade-off the optimizer must weigh.

Every archetype coefficient and preset lives in :class:`~optisample.config.synth.SynthConfig`
(the ``opticonfig/synth.yaml`` values); the render functions take it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optisample.config.synth import PresetConfig, SynthConfig
from optisample.io.audio import write_wav
from optisample.io.manifest import dump_manifest
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample
from optisample.music import A4_FREQ_HZ, MIDI_MAX_VELOCITY, midi_to_freq

Archetype = Literal["sustained", "piano"]


@dataclass(frozen=True)
class NoteSpec:
    """A single note to render, independent of archetype."""

    pitch: int
    velocity: int
    controller: float
    duration_s: float
    sample_rate: int

    @property
    def vel(self) -> float:
        """Normalized velocity in [0, 1]."""
        return self.velocity / MIDI_MAX_VELOCITY

    def time_axis(self) -> NDArray[np.float64]:
        return np.arange(int(self.duration_s * self.sample_rate), dtype=np.float64) / self.sample_rate


def _n_partials(fundamental: float, sample_rate: int, requested: int, nyquist_fraction: float) -> int:
    """Cap the partial count so the highest partial stays below Nyquist (no reference aliasing)."""
    ceiling = int(nyquist_fraction * sample_rate / fundamental)
    return max(1, min(requested, ceiling))


def _velocity_peak(velocity: int, floor: float, scale: float) -> float:
    """Map velocity → target peak amplitude (monotonic, stays below full scale)."""
    return floor + scale * (velocity / MIDI_MAX_VELOCITY)


def _normalize_peak(signal: NDArray[np.float64], peak: float) -> NDArray[np.float64]:
    largest = float(np.max(np.abs(signal)))
    if largest == 0.0:
        return signal
    return np.asarray(signal / largest * peak, dtype=np.float64)


def _attack_release(t: NDArray[np.float64], attack_s: float, release_s: float) -> NDArray[np.float64]:
    attack = np.minimum(1.0, t / attack_s)
    total = float(t[-1]) if t.size else 0.0
    release = np.minimum(1.0, np.maximum(0.0, (total - t) / release_s))
    return np.asarray(attack * release, dtype=np.float64)


def render_sustained(spec: NoteSpec, rng: np.random.Generator, config: SynthConfig) -> NDArray[np.float64]:
    """Synthesize a sustained tone: vibrato-modulated partials rolled off by velocity and controller.

    Partials at or above ``evolution_min_partial`` get a slow independent wax/wane, so the sustain
    drifts rather than repeating exactly; an attack/release envelope then shapes it and it is
    peak-normalized to the velocity's target level.
    """
    cfg = config.sustained
    fundamental = midi_to_freq(spec.pitch)
    partials = _n_partials(fundamental, spec.sample_rate, cfg.n_partials, config.nyquist_fraction)
    t = spec.time_axis()

    vibrato = 1.0 + cfg.vibrato_depth * np.sin(2.0 * np.pi * cfg.vibrato_hz * t)
    rolloff = (cfg.rolloff_base + cfg.rolloff_vel * spec.vel) * (1.0 + cfg.rolloff_controller * spec.controller)
    signal = np.zeros_like(t)
    for n in range(1, partials + 1):
        phase = 2.0 * np.pi * np.cumsum(n * fundamental * vibrato) / spec.sample_rate + rng.uniform(0.0, 2.0 * np.pi)
        depth = cfg.evolution_depth if n >= cfg.evolution_min_partial else 0.0
        evolution = 1.0 + depth * np.sin(2.0 * np.pi * cfg.evolution_hz * t + n)
        signal += rolloff ** (n - 1) / n * evolution * np.sin(phase)

    signal *= _attack_release(t, attack_s=cfg.attack_s, release_s=cfg.release_s)
    return _normalize_peak(
        signal, _velocity_peak(spec.velocity, config.velocity_peak_floor, config.velocity_peak_scale)
    )


def render_piano(spec: NoteSpec, rng: np.random.Generator, config: SynthConfig) -> NDArray[np.float64]:
    """Synthesize a struck-string tone: inharmonic partials with frequency-dependent decay.

    Partial ``n`` is stretched slightly sharp by ``inharmonicity`` (the stiff-string effect) and rings
    with a time constant that shortens for higher partials, so the tone darkens as it decays; a short
    attack ramp and peak-normalization to the velocity's target level finish it.
    """
    cfg = config.piano
    fundamental = midi_to_freq(spec.pitch)
    partials = _n_partials(fundamental, spec.sample_rate, cfg.n_partials, config.nyquist_fraction)
    t = spec.time_axis()

    base_tau = float(
        np.clip(cfg.base_tau_scale * (A4_FREQ_HZ / fundamental) ** cfg.base_tau_exp, cfg.base_tau_min, cfg.base_tau_max)
    )
    rolloff = (cfg.rolloff_base + cfg.rolloff_vel * spec.vel) * (1.0 + cfg.rolloff_controller * spec.controller)
    signal = np.zeros_like(t)
    for n in range(1, partials + 1):
        freq = n * fundamental * np.sqrt(1.0 + cfg.inharmonicity * n * n)
        tau = base_tau / (1.0 + cfg.decay_partial_factor * (n - 1))
        phase = 2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi)
        signal += rolloff ** (n - 1) / n * np.exp(-t / tau) * np.sin(phase)

    signal *= np.minimum(1.0, t / cfg.attack_s)
    return _normalize_peak(
        signal, _velocity_peak(spec.velocity, config.velocity_peak_floor, config.velocity_peak_scale)
    )


def render_sample(
    archetype: Archetype, spec: NoteSpec, rng: np.random.Generator, config: SynthConfig
) -> NDArray[np.float64]:
    if archetype == "sustained":
        return render_sustained(spec, rng, config)
    return render_piano(spec, rng, config)


def _render_instrument(
    outdir: Path, preset: PresetConfig, config: SynthConfig, rate: int, rng: np.random.Generator
) -> InstrumentSpec:
    """Render one preset's (pitch, velocity) grid to WAVs under ``outdir`` and assemble its spec."""
    (outdir / preset.id).mkdir(parents=True, exist_ok=True)
    samples: list[SourceSample] = []
    for pitch in preset.pitches:
        for velocity in preset.velocities:
            spec = NoteSpec(pitch, velocity, 0.0, preset.sample_dur, rate)
            rel = Path(preset.id) / f"p{pitch}_v{velocity}_c0.wav"
            write_wav(outdir / rel, render_sample(preset.archetype, spec, rng, config), rate)
            samples.append(SourceSample(file=rel, pitch=pitch, velocity=velocity, controller=0.0))
    material = [NoteEvent(pitch=p, velocity=v, duration_s=d, count=c) for p, v, d, c in preset.material]
    return InstrumentSpec(id=preset.id, budget_kb=preset.budget_kb, samples=samples, material=material)


def generate_demo(outdir: Path | str, config: SynthConfig, *, sample_rate: int | None = None, seed: int = 0) -> Path:
    """Render every preset instrument in ``config`` to ``outdir`` and write a ``manifest.yaml``.

    ``sample_rate`` overrides the render rate for a quick low-rate run; ``None`` uses
    ``config.sample_rate``. Returns the path to the written manifest. Deterministic for a given ``seed``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rate = config.sample_rate if sample_rate is None else sample_rate
    rng = np.random.default_rng(seed)

    instruments = [_render_instrument(outdir, preset, config, rate, rng) for preset in config.presets]
    manifest = Manifest(project=ProjectSpec(name="demo"), instruments=instruments)
    manifest_path = outdir / "manifest.yaml"
    dump_manifest(manifest, manifest_path)
    return manifest_path
