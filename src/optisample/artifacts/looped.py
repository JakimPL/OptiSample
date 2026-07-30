from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np

from optisample.artifacts.dataset import note_records, tracked_ccs, write_recording
from optisample.artifacts.paths import LoopedPaths, looped_paths
from optisample.artifacts.serialize import LoopsDocument, WrittenSampleRecord, loops_document, write_json
from optisample.dsp.decay import LinearDecay
from optisample.dsp.loop import crossfade_loop, seam_frames
from optisample.io.audio import write_wav
from optisample.io.note_extractor import dump_notes
from optisample.keys import SampleKey
from optisample.loop.settle import StoredLoop
from optisample.metrics.base import Signal
from optisample.model import Manifest
from optisample.optimize.orchestrate.audio import load_run_audio
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.progress import ProgressSink

_AUDITION_LABEL: Final = "Rendering loop auditions"
_RECORDING_STEM: Final = "recording"
_LOOPED_STEM: Final = "looped"
_HELD_ROUNDS: Final = 4  # times an audition wraps the loop, enough to hear a seam and a level step repeat


@dataclass(frozen=True)
class LoopedInstrumentArtifacts:
    """One instrument's looped dataset: where each part landed, and how much of it there is.

    ``looped`` counts the recordings that ended up stored around a loop, against ``recordings`` in total, so
    a reader sees how much of the material the stage found a loop for.
    """

    instrument_id: str
    paths: LoopedPaths
    recordings: int
    looped: int
    auditions: int
    elapsed_s: float


@dataclass(frozen=True)
class _Written:
    """The recordings a looped dataset holds: what was written, and the index each key joins on."""

    records: tuple[WrittenSampleRecord, ...]
    indices: dict[SampleKey, int]


def _write_recordings(looped: LoopedInstrument, samples_dir: Path) -> _Written:
    """Write each recording as the stage analysed it, so the frames a loop names index into the file.

    The audio written here is what the ingest produced: onset aligned, at the run's one rate, and trimmed
    to the span worth storing. Recordings are numbered in key order, so the same dataset reduces to the same
    filenames on every run.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    loaded = looped.loaded
    records: list[WrittenSampleRecord] = []
    indices: dict[SampleKey, int] = {}
    for index, key in enumerate(sorted(loaded.audio)):
        records.append(write_recording(loaded.audio[key], key, index, loaded.sample_rate, samples_dir))
        indices[key] = index

    return _Written(records=tuple(records), indices=indices)


def held_audition(signal: Signal, stored: StoredLoop, sample_rate: int, seam: int) -> Signal:
    """The recording played out through its loop: the attack, the loop wrapped a few times, then the decline.

    The loop is crossfaded first, so what is heard is the wrap a player makes rather than the raw splice,
    and the fitted decay is put over the whole span, so the audition carries the level a held note falls to
    as well as the seam it falls through. Wrapping :data:`_HELD_ROUNDS` times is enough for a seam step or a
    level pulse to become a rhythm rather than a single click.
    """
    faded = crossfade_loop(signal, stored.loop, fade_len=seam)
    region = faded[stored.loop.start : stored.loop.end]
    played = np.concatenate([faded[: stored.loop.end], np.tile(region, _HELD_ROUNDS)])
    return _declined(played, stored.decay, sample_rate)


def _declined(played: Signal, decay: LinearDecay | None, sample_rate: int) -> Signal:
    """``played`` brought down by the ramp the recording states, where it states one to make."""
    if decay is None:
        return played

    return np.asarray(played * decay.envelope(played.size, sample_rate), dtype=np.float64)


def _write_auditions(looped: LoopedInstrument, out_dir: Path, seam: int, progress: ProgressSink) -> int:
    """Write the recording beside its loop played out, one folder per recording that earned a loop.

    A recording the stage settled no loop for has nothing to audition against itself, so it contributes no
    folder and the ones present are exactly the loops a listener has to judge.
    """
    loaded = looped.loaded
    judged = [
        (key, settlement.stored)
        for key, settlement in sorted(looped.settlements.items())
        if settlement.stored is not None
    ]
    written = 0
    for key, stored in progress.track(judged, label=_AUDITION_LABEL, total=len(judged)):
        folder = out_dir / key.label
        folder.mkdir(parents=True, exist_ok=True)
        signal = loaded.audio[key]
        write_wav(folder / f"{_RECORDING_STEM}.wav", signal, loaded.sample_rate)
        held = held_audition(signal, stored, loaded.sample_rate, seam)
        write_wav(folder / f"{_LOOPED_STEM}.wav", held, loaded.sample_rate)
        written += 2

    return written


def _searched(looped: LoopedInstrument) -> dict[SampleKey, float]:
    """The stretch each recording's candidates were measured over, which the document states per recording."""
    return {key: settlement.search_s for key, settlement in looped.settlements.items()}


def looped_document(looped: LoopedInstrument) -> LoopsDocument:
    """What the stage decided for this instrument, as the document written beside its dataset."""
    return loops_document(
        looped.loaded.instrument.id,
        looped.loaded.sample_rate,
        looped.settlements,
        _searched(looped),
    )


def dump_looped(
    looped: LoopedInstrument,
    out_dir: Path | str,
    settings: OptimizeSettings,
) -> LoopedInstrumentArtifacts:
    """Write one instrument's settled loops and the dataset they were measured over under ``out_dir``.

    The output root is itself a NoteExtractor dataset -- one WAV per recording as the stage analysed it, and
    the material routed onto those recordings -- so the reduction picks up from here over audio the loop
    frames already index into. Beside it, ``loops.json`` states the loop each recording keeps and the candidates climbed past, and
    ``auditions/`` holds each loop played out against the recording it was taken from, which is what makes
    the stage judgeable by ear on its own.
    """
    started_at = perf_counter()
    loaded = looped.loaded
    paths = looped_paths(Path(out_dir), loaded.instrument.id)
    material = loaded.instrument.material
    written = _write_recordings(looped, paths.samples_dir)
    dump_notes(
        note_records(material, loaded.audio, written.indices),
        paths.notes_json,
        tracked_ccs=tracked_ccs(material),
    )
    seam = seam_frames(settings.loop.seam, loaded.sample_rate)
    auditions = _write_auditions(looped, paths.auditions_dir, seam, settings.progress)
    paths.loops_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(paths.loops_json, looped_document(looped))
    return LoopedInstrumentArtifacts(
        instrument_id=loaded.instrument.id,
        paths=paths,
        recordings=len(written.records),
        looped=looped.looped_recordings,
        auditions=auditions,
        elapsed_s=perf_counter() - started_at,
    )


def loop_project(
    manifest: Manifest,
    out_dir: Path | str,
    settings: OptimizeSettings,
) -> list[LoopedInstrumentArtifacts]:
    """Settle every instrument's loops and write each one's looped dataset under ``out_dir``.

    Instruments share the root, each contributing its own ``<id>.notes.json`` and ``<id>/`` pair, so one
    loop run over a project yields one looped dataset per instrument in the shape ingest expects.
    """
    out_dir = Path(out_dir)
    return [
        dump_looped(run_loops(load_run_audio(instrument, settings), settings), out_dir, settings)
        for instrument in manifest.instruments
    ]
