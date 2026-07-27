from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from optisample.config.synth import PresetConfig, SynthConfig
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NoteRecord, dump_notes
from optisample.progress import ProgressSink
from optisample.synth.archetypes import NoteSpec, render_sample

DEFAULT_SEED: Final = 137
_NO_CONTROLLER: Final = 0.0


@dataclass(frozen=True)
class _Rendering:
    """What every note of every preset is rendered through: the archetype config, rate, RNG and sink."""

    config: SynthConfig
    sample_rate: int
    rng: np.random.Generator
    progress: ProgressSink


def _note_specs(preset: PresetConfig, sample_rate: int) -> list[NoteSpec]:
    """Every note the preset's material song plays, its events expanded by their repeat counts.

    Listing them before rendering gives the run a note total to report against, and their order is the
    render index each WAV is named by.
    """
    return [
        NoteSpec(event.pitch, event.velocity, _NO_CONTROLLER, event.duration_s, sample_rate)
        for event in preset.material
        for _ in range(event.count)
    ]


def _render_preset(outdir: Path, preset: PresetConfig, rendering: _Rendering) -> tuple[Path, Path]:
    """Render one preset's material song to per-note WAVs and write its ``.notes.json``.

    Each material event is expanded by its ``count`` into individual played notes -- one WAV each, named
    with a leading render index -- so the samples deduplicate to the played ``(pitch, velocity)`` grid
    while the notes reproduce the song. Returns the written ``(notes_json, samples_dir)`` pair.
    """
    samples_dir = outdir / preset.id
    samples_dir.mkdir(parents=True, exist_ok=True)
    specs = _note_specs(preset, rendering.sample_rate)
    records: list[NoteRecord] = []
    tracked = rendering.progress.track(specs, label=f"Rendering {preset.id}", total=len(specs))
    for index, spec in enumerate(tracked):
        write_wav(
            samples_dir / f"{index:04d}_p{spec.pitch}_v{spec.velocity}.wav",
            render_sample(preset.archetype, spec, rendering.rng, rendering.config),
            rendering.sample_rate,
        )
        records.append(
            NoteRecord(
                index=index,
                pitch=spec.pitch,
                velocity=spec.velocity,
                duration_s=spec.duration_s,
            )
        )

    notes_json = outdir / f"{preset.id}.notes.json"
    dump_notes(records, notes_json)
    return notes_json, samples_dir


def generate_demo(
    outdir: Path | str,
    config: SynthConfig,
    *,
    sample_rate: int | None = None,
    seed: int = DEFAULT_SEED,
    progress: ProgressSink,
) -> list[tuple[Path, Path]]:
    """Render every preset in ``config`` to ``outdir`` as a NoteExtractor-style samples dir + ``.notes.json``.

    ``sample_rate`` overrides the render rate for a quick low-rate run; ``None`` uses ``config.sample_rate``.
    Returns one ``(notes_json, samples_dir)`` pair per preset. Deterministic for a given ``seed``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rendering = _Rendering(
        config=config,
        sample_rate=config.sample_rate if sample_rate is None else sample_rate,
        rng=np.random.default_rng(seed),
        progress=progress,
    )
    return [_render_preset(outdir, preset, rendering) for preset in config.presets]
