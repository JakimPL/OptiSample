from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.synth import PresetConfig, SynthConfig
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NoteRecord, dump_notes
from optisample.parallel import IN_PROCESS, map_workers
from optisample.progress import NO_PROGRESS, ProgressSink
from optisample.seed import DEFAULT_SEED
from optisample.synth.archetypes import Archetype, NoteSpec, draw_phases, render_sample

_NO_CONTROLLER: Final = 0.0


@dataclass(frozen=True)
class DemoSettings:
    """How a demo dataset is produced, as opposed to which instruments it holds.

    ``sample_rate`` is the rate every preset renders at, which a quick run drops below the configured
    one. ``seed`` is the entropy the whole dataset is reproducible from, ``workers`` how many processes
    share the notes out between them, and ``progress`` where each preset reports how far through it is.
    """

    sample_rate: int
    seed: int = DEFAULT_SEED
    workers: int = IN_PROCESS
    progress: ProgressSink = NO_PROGRESS


@dataclass(frozen=True)
class _Rendering:
    """What every note of every preset is rendered through: the archetype config, the RNG and the run."""

    config: SynthConfig
    rng: np.random.Generator
    settings: DemoSettings


@dataclass(frozen=True)
class _PlayedNote:
    """One note as a worker receives it: where it lands, what to synthesize, and where it starts from."""

    path: Path
    archetype: Archetype
    spec: NoteSpec
    phases: NDArray[np.float64]


def _write_note(note: _PlayedNote, config: SynthConfig) -> None:
    """Synthesize one note and write it as a WAV, which is the whole of a worker's share of a preset."""
    write_wav(note.path, render_sample(note.archetype, note.spec, note.phases, config), note.spec.sample_rate)


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


def _played_notes(
    samples_dir: Path,
    preset: PresetConfig,
    specs: Sequence[NoteSpec],
    rendering: _Rendering,
) -> list[_PlayedNote]:
    """Pair every note with the file it lands in and the phases it starts from, in render-index order.

    The whole preset's phases are drawn here, one note after another, so a seed reproduces the dataset
    however the notes are then shared out to be rendered.
    """
    return [
        _PlayedNote(
            path=samples_dir / f"{index:04d}_p{spec.pitch}_v{spec.velocity}.wav",
            archetype=preset.archetype,
            spec=spec,
            phases=draw_phases(preset.archetype, spec, rendering.rng, rendering.config),
        )
        for index, spec in enumerate(specs)
    ]


def _render_preset(outdir: Path, preset: PresetConfig, rendering: _Rendering) -> tuple[Path, Path]:
    """Render one preset's material song to per-note WAVs and write its ``.notes.json``.

    Each material event is expanded by its ``count`` into individual played notes -- one WAV each, named
    with a leading render index -- so the samples deduplicate to the played ``(pitch, velocity)`` grid
    while the notes reproduce the song. Returns the written ``(notes_json, samples_dir)`` pair.
    """
    samples_dir = outdir / preset.id
    samples_dir.mkdir(parents=True, exist_ok=True)
    specs = _note_specs(preset, rendering.settings.sample_rate)
    map_workers(
        partial(_write_note, config=rendering.config),
        _played_notes(samples_dir, preset, specs, rendering),
        workers=rendering.settings.workers,
        label=f"Rendering {preset.id}",
        progress=rendering.settings.progress,
    )
    records = [
        NoteRecord(index=index, pitch=spec.pitch, velocity=spec.velocity, duration_s=spec.duration_s)
        for index, spec in enumerate(specs)
    ]
    notes_json = outdir / f"{preset.id}.notes.json"
    dump_notes(records, notes_json)
    return notes_json, samples_dir


def generate_demo(outdir: Path | str, config: SynthConfig, settings: DemoSettings) -> list[tuple[Path, Path]]:
    """Render every preset in ``config`` to ``outdir`` as a NoteExtractor-style samples dir + ``.notes.json``.

    Returns one ``(notes_json, samples_dir)`` pair per preset. Deterministic for a given
    ``settings.seed``, whichever fan-out the notes were rendered under.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rendering = _Rendering(config=config, rng=np.random.default_rng(settings.seed), settings=settings)
    return [_render_preset(outdir, preset, rendering) for preset in config.presets]
