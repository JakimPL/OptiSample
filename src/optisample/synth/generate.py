"""Serialize synthesized archetypes into a demo dataset: per-note WAVs plus a ``manifest.yaml``.

:func:`generate_demo` walks every preset in the config, renders each preset's (pitch, velocity) grid
via :mod:`optisample.synth.archetypes`, writes the WAVs, and assembles the manifest the optimizer
consumes. Deterministic for a given seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from optisample.config.synth import PresetConfig, SynthConfig
from optisample.io.audio import write_wav
from optisample.io.manifest import dump_manifest
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample
from optisample.synth.archetypes import NoteSpec, render_sample

DEFAULT_SEED: Final = 0  # default RNG seed for reproducible demo generation.


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
    material = [
        NoteEvent(pitch=event.pitch, velocity=event.velocity, duration_s=event.duration_s, count=event.count)
        for event in preset.material
    ]
    return InstrumentSpec(id=preset.id, budget_kb=preset.budget_kb, samples=samples, material=material)


def generate_demo(
    outdir: Path | str, config: SynthConfig, *, sample_rate: int | None = None, seed: int = DEFAULT_SEED
) -> Path:
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
