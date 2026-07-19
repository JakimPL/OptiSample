"""Deterministic synthetic instruments so the pipeline is testable without real recordings.

Two archetypes are produced, matching the POC targets:

* ``sustained`` (strings/pad-like): slow attack, an evolving sustain with slight vibrato,
  release. The sustain *evolves* over time on purpose, so a too-short loop is measurably
  static later on.
* ``piano`` (decaying one-shot): fast attack, per-partial exponential decay (high partials
  decay faster → the tone darkens as it rings), mild inharmonicity.

For both, higher velocity is rendered *louder and brighter* — so emulating velocity by
volume alone is genuinely lossy, which is exactly the trade-off the optimizer must weigh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optisample.io.audio import write_wav
from optisample.io.manifest import dump_manifest
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample

SAMPLE_RATE = 44_100

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
        return self.velocity / 127.0

    def time_axis(self) -> NDArray[np.float64]:
        return np.arange(int(self.duration_s * self.sample_rate), dtype=np.float64) / self.sample_rate


def midi_to_freq(pitch: int) -> float:
    """MIDI note number → fundamental frequency in Hz (A4/69 = 440 Hz)."""
    return float(440.0 * 2.0 ** ((pitch - 69) / 12.0))


def _n_partials(fundamental: float, sample_rate: int, requested: int) -> int:
    """Cap the partial count so the highest partial stays below Nyquist (no reference aliasing)."""
    ceiling = int(0.45 * sample_rate / fundamental)
    return max(1, min(requested, ceiling))


def _velocity_peak(velocity: int) -> float:
    """Map velocity → target peak amplitude (monotonic, stays below full scale)."""
    return 0.1 + 0.85 * (velocity / 127.0)


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


def render_sustained(spec: NoteSpec, rng: np.random.Generator) -> NDArray[np.float64]:
    fundamental = midi_to_freq(spec.pitch)
    partials = _n_partials(fundamental, spec.sample_rate, 10)
    t = spec.time_axis()

    vibrato = 1.0 + 0.004 * np.sin(2.0 * np.pi * 5.2 * t)
    rolloff = (0.5 + 0.4 * spec.vel) * (1.0 + 0.002 * spec.controller)  # brighter when louder
    signal = np.zeros_like(t)
    for n in range(1, partials + 1):
        phase = 2.0 * np.pi * np.cumsum(n * fundamental * vibrato) / spec.sample_rate + rng.uniform(0.0, 2.0 * np.pi)
        # High partials slowly wax/wane so the sustain is not perfectly periodic.
        evolution = 1.0 + (0.2 if n >= 4 else 0.0) * np.sin(2.0 * np.pi * 0.5 * t + n)
        signal += rolloff ** (n - 1) / n * evolution * np.sin(phase)

    signal *= _attack_release(t, attack_s=0.08, release_s=0.25)
    return _normalize_peak(signal, _velocity_peak(spec.velocity))


def render_piano(spec: NoteSpec, rng: np.random.Generator) -> NDArray[np.float64]:
    fundamental = midi_to_freq(spec.pitch)
    partials = _n_partials(fundamental, spec.sample_rate, 12)
    t = spec.time_axis()

    base_tau = float(np.clip(0.9 * (440.0 / fundamental) ** 0.5, 0.2, 1.2))
    rolloff = (0.45 + 0.45 * spec.vel) * (1.0 + 0.001 * spec.controller)
    signal = np.zeros_like(t)
    for n in range(1, partials + 1):
        freq = n * fundamental * np.sqrt(1.0 + 0.0004 * n * n)  # mild inharmonicity
        tau = base_tau / (1.0 + 0.6 * (n - 1))  # high partials decay faster → tone darkens
        phase = 2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi)
        signal += rolloff ** (n - 1) / n * np.exp(-t / tau) * np.sin(phase)

    signal *= np.minimum(1.0, t / 0.003)
    return _normalize_peak(signal, _velocity_peak(spec.velocity))


def render_sample(archetype: Archetype, spec: NoteSpec, rng: np.random.Generator) -> NDArray[np.float64]:
    if archetype == "sustained":
        return render_sustained(spec, rng)
    return render_piano(spec, rng)


@dataclass(frozen=True)
class _Preset:
    id: str
    archetype: Archetype
    pitches: tuple[int, ...]
    velocities: tuple[int, ...]
    sample_dur: float
    budget_kb: float
    material: tuple[tuple[int, int, float, int], ...]  # (pitch, velocity, duration_s, count)


_PRESETS: tuple[_Preset, ...] = (
    _Preset(
        id="strings",
        archetype="sustained",
        pitches=(48, 55, 60, 67, 72),
        velocities=(40, 80, 115),
        sample_dur=4.5,  # >= the longest material note (4.0 s), so a real sustain exists to loop/match
        budget_kb=128.0,
        material=(
            (60, 80, 2.5, 10),
            (55, 80, 3.0, 6),
            (67, 115, 1.5, 8),
            (48, 40, 4.0, 3),
            (72, 115, 1.0, 5),
            (60, 40, 2.0, 4),
        ),
    ),
    _Preset(
        id="piano",
        archetype="piano",
        pitches=(48, 55, 60, 67, 72),
        velocities=(50, 100),
        sample_dur=1.6,
        budget_kb=96.0,
        material=(
            (60, 100, 0.8, 12),
            (48, 50, 1.2, 5),
            (67, 100, 0.5, 10),
            (55, 50, 0.9, 7),
            (72, 100, 0.4, 6),
        ),
    ),
)


def generate_demo(outdir: Path | str, *, sample_rate: int = SAMPLE_RATE, seed: int = 0) -> Path:
    """Render both demo instruments to ``outdir`` and write a ``manifest.yaml``.

    Returns the path to the written manifest. Deterministic for a given ``seed``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    instruments: list[InstrumentSpec] = []
    for preset in _PRESETS:
        (outdir / preset.id).mkdir(parents=True, exist_ok=True)
        samples: list[SourceSample] = []
        for pitch in preset.pitches:
            for velocity in preset.velocities:
                spec = NoteSpec(pitch, velocity, 0.0, preset.sample_dur, sample_rate)
                rel = Path(preset.id) / f"p{pitch}_v{velocity}_c0.wav"
                write_wav(outdir / rel, render_sample(preset.archetype, spec, rng), sample_rate)
                samples.append(SourceSample(file=rel, pitch=pitch, velocity=velocity, controller=0.0))
        material = [NoteEvent(pitch=p, velocity=v, duration_s=d, count=c) for p, v, d, c in preset.material]
        instruments.append(InstrumentSpec(id=preset.id, budget_kb=preset.budget_kb, samples=samples, material=material))

    manifest = Manifest(project=ProjectSpec(name="demo"), instruments=instruments)
    manifest_path = outdir / "manifest.yaml"
    dump_manifest(manifest, manifest_path)
    return manifest_path
