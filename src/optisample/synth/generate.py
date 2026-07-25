"""Serialize synthesized archetypes into a demo dataset: per-note WAVs plus a NoteExtractor ``.notes.json``.

:func:`generate_demo` walks every preset in the config, count-expands each preset's material song into
individual played notes, renders one WAV per note via :mod:`optisample.synth.archetypes`, and writes each
preset's samples directory alongside its ``.notes.json`` -- the same on-disk pair NoteExtractor produces
and :func:`optisample.io.note_extractor.load_notes` consumes. Deterministic for a given seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from optisample.config.synth import PresetConfig, SynthConfig
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NoteRecord, dump_notes
from optisample.synth.archetypes import NoteSpec, render_sample

DEFAULT_SEED: Final = 0
_NO_CONTROLLER: Final = 0.0


def _render_preset(
    outdir: Path, preset: PresetConfig, config: SynthConfig, rate: int, rng: np.random.Generator
) -> tuple[Path, Path]:
    """Render one preset's material song to per-note WAVs and write its ``.notes.json``.

    Each material event is expanded by its ``count`` into individual played notes -- one WAV each, named
    with a leading render index -- so the samples deduplicate to the played ``(pitch, velocity)`` grid
    while the notes reproduce the song. Returns the written ``(notes_json, samples_dir)`` pair.
    """
    samples_dir = outdir / preset.id
    samples_dir.mkdir(parents=True, exist_ok=True)
    records: list[NoteRecord] = []
    for event in preset.material:
        for _ in range(event.count):
            index = len(records)
            spec = NoteSpec(event.pitch, event.velocity, _NO_CONTROLLER, event.duration_s, rate)
            name = f"{index:04d}_p{event.pitch}_v{event.velocity}.wav"
            write_wav(samples_dir / name, render_sample(preset.archetype, spec, rng, config), rate)
            records.append(
                NoteRecord(index=index, pitch=event.pitch, velocity=event.velocity, duration_s=event.duration_s)
            )
    notes_json = outdir / f"{preset.id}.notes.json"
    dump_notes(records, notes_json)
    return notes_json, samples_dir


def generate_demo(
    outdir: Path | str, config: SynthConfig, *, sample_rate: int | None = None, seed: int = DEFAULT_SEED
) -> list[tuple[Path, Path]]:
    """Render every preset in ``config`` to ``outdir`` as a NoteExtractor-style samples dir + ``.notes.json``.

    ``sample_rate`` overrides the render rate for a quick low-rate run; ``None`` uses ``config.sample_rate``.
    Returns one ``(notes_json, samples_dir)`` pair per preset. Deterministic for a given ``seed``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rate = config.sample_rate if sample_rate is None else sample_rate
    rng = np.random.default_rng(seed)
    return [_render_preset(outdir, preset, config, rate, rng) for preset in config.presets]
