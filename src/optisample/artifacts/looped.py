from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np

from optisample.artifacts.dataset import note_records, recording_stem, tracked_ccs, write_recording
from optisample.artifacts.documents.loops import LoopsDocument, loops_document
from optisample.artifacts.documents.reduction import WrittenSampleRecord
from optisample.artifacts.documents.sample import ProvenanceRecord, SampleDocument, sample_document
from optisample.artifacts.instruments.dump import InstrumentSettings, WrittenInstruments, write_dataset_instruments
from optisample.artifacts.paths import LoopedPaths, looped_paths
from optisample.artifacts.serialize import write_json, write_msgpack
from optisample.config.loop import LoopConfig, SeamConfig
from optisample.dsp.decay import LinearDecay
from optisample.dsp.envelope import LevelReading, decompose, level_reading
from optisample.dsp.loop import prepare_loop
from optisample.io.audio import write_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import dump_notes
from optisample.keys import SampleKey
from optisample.loop.settle import Settlement, StoredLoop
from optisample.metrics.base import Signal
from optisample.model import Manifest
from optisample.music import midi_to_freq
from optisample.optimize.orchestrate.audio import LoadedInstrument, load_run_audio
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.progress import ProgressSink

_AUDITION_LABEL: Final = "Rendering loop auditions"
_RECORDING_STEM: Final = "recording"
_LOOPED_STEM: Final = "looped"  # suffixed with the offer's own index, which is what an encoding names it by
_HELD_ROUNDS: Final = 4  # times an audition wraps the loop, enough to hear a seam and a level step repeat
_STAGE: Final = "loop"  # the run behind every container this stage writes, stamped into its provenance
_NO_SETTLEMENT: Final = None  # what a run storing no loops leaves a recording with, which offers none


@dataclass(frozen=True)
class LoopedInstrumentArtifacts:
    """One instrument's looped dataset: where each part landed, and how much of it there is.

    ``looped`` counts the recordings that offer at least one loop, against ``recordings`` in total, so a
    reader sees how much of the material the stage found a loop for. Each of those ``recordings`` lands as a
    WAV, as the ``.sample`` carrying its calibrated form, and as the standalone ``instruments`` a player
    loads it through. ``auditions`` counts the files written, which is one per recording plus one per loop
    it offers.
    """

    instrument_id: str
    paths: LoopedPaths
    recordings: int
    looped: int
    auditions: int
    instruments: WrittenInstruments
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


def _calibrated_sample(
    loaded: LoadedInstrument,
    key: SampleKey,
    index: int,
    settlement: Settlement | None,
    config: LoopConfig,
) -> SampleDocument:
    """One recording as the calibrated unit it is carried on: its split, its loops and where it came from.

    The level is read at the pitch the key sounds, which is the reading the loop stage settled this
    recording's loops under, so the split a container carries is the one every measurement of that
    recording was taken through. A run storing no loops leaves the container stating none, and the pair it
    holds still puts the recording back together.
    """
    reading = level_reading(loaded.sample_rate, config.envelope, midi_to_freq(key.pitch))
    return sample_document(
        decompose(loaded.audio[key], reading),
        settlement.offered if settlement is not None else (),
        key=key,
        sample_rate=loaded.sample_rate,
        provenance=ProvenanceRecord(
            stage=_STAGE,
            instrument_id=loaded.instrument.id,
            index=index,
            source=f"{recording_stem(key, index)}.wav",
        ),
    )


def _write_calibrated(looped: LoopedInstrument, written: _Written, paths: LoopedPaths, config: LoopConfig) -> None:
    """Write every recording as a ``.sample`` beside the WAV of the same stem.

    A container states the recording split into the level it moves through and the carrier that level
    scales, so a stage reading one back stores the carrier over the whole depth of its grid and plays the
    level as a curve. Its loops travel with it, which is what carries the stage's decision forward in the
    company of the very audio it was measured over.
    """
    for key, index in sorted(written.indices.items()):
        document = _calibrated_sample(looped.loaded, key, index, looped.settlements.get(key, _NO_SETTLEMENT), config)
        write_msgpack(paths.calibrated(recording_stem(key, index)), document)


def held_audition(
    signal: Signal,
    stored: StoredLoop,
    sample_rate: int,
    seam: SeamConfig,
    reading: LevelReading,
) -> Signal:
    """The recording played out through its loop: the attack, the loop wrapped a few times, then the decline.

    The region is prepared first -- held at one level and blended at the seam -- so what is heard is the wrap
    a player makes over the waveform a sample stores, and the fitted decay is put over the whole span, so the
    audition carries the level a held note falls to as well as the seam it falls through. Wrapping
    :data:`_HELD_ROUNDS` times is enough for a seam step or a level pulse to become a rhythm a listener
    catches rather than a single click.
    """
    prepared = prepare_loop(signal, stored.loop, sample_rate, seam, reading)
    region = prepared[stored.loop.start : stored.loop.end]
    played = np.concatenate([prepared[: stored.loop.end], np.tile(region, _HELD_ROUNDS)])
    return _declined(played, stored.decay, sample_rate)


def _declined(played: Signal, decay: LinearDecay | None, sample_rate: int) -> Signal:
    """``played`` brought down by the ramp the recording states, where it states one to make."""
    if decay is None:
        return played

    return np.asarray(played * decay.envelope(played.size, sample_rate), dtype=np.float64)


def _write_key_auditions(
    loaded: LoadedInstrument,
    key: SampleKey,
    offered: Sequence[StoredLoop],
    folder: Path,
    config: LoopConfig,
) -> int:
    """One recording and each loop it offers played out, written into ``folder``; answers the files written.

    Every offer is written, named by the index an encoding reaches it under, so the folder holds the whole
    stretch of stored length the sweep prices and what a longer region buys is audible against what it
    costs. The level is read at the pitch the key sounds, the same way the loop stage read it, so each
    audition wraps the waveform the stage measured.
    """
    folder.mkdir(parents=True, exist_ok=True)
    signal = loaded.audio[key]
    write_wav(folder / f"{_RECORDING_STEM}.wav", signal, loaded.sample_rate)
    reading = level_reading(loaded.sample_rate, config.envelope, midi_to_freq(key.pitch))
    for index, stored in enumerate(offered):
        held = held_audition(signal, stored, loaded.sample_rate, config.seam, reading)
        write_wav(folder / f"{_LOOPED_STEM}{index}.wav", held, loaded.sample_rate)

    return 1 + len(offered)


def _write_auditions(
    looped: LoopedInstrument,
    out_dir: Path,
    config: LoopConfig,
    progress: ProgressSink,
) -> int:
    """Write the recording beside each loop it offers played out, one folder per recording that earned one.

    A recording the stage found no loop for has nothing to audition against itself, so it contributes no
    folder and the ones present are exactly the loops a listener has to judge.
    """
    judged = [(key, settlement.offered) for key, settlement in sorted(looped.settlements.items()) if settlement.loops]
    return sum(
        _write_key_auditions(looped.loaded, key, offered, out_dir / key.label, config)
        for key, offered in progress.track(judged, label=_AUDITION_LABEL, total=len(judged))
    )


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
    instruments: InstrumentSettings,
) -> LoopedInstrumentArtifacts:
    """Write one instrument's settled loops and the dataset they were measured over under ``out_dir``.

    The output root is itself a NoteExtractor dataset -- one WAV per recording as the stage analysed it, and
    the material routed onto those recordings -- so the reduction picks up from here over audio the loop
    frames already index into. Each recording is written a second time as a ``.sample`` of the same stem,
    the calibrated unit carrying its level/carrier split and the loops settled over it, and a third time as
    the standalone instruments of every format (:func:`~optisample.artifacts.instruments.dump
    .write_dataset_instruments`), so the stage's own audio is playable in a tracker as it stands. Beside
    them, ``loops.json`` states the loops each recording offers and the candidates turned down, and
    ``auditions/`` holds each of those loops played out against the recording it was taken from, which is
    what makes the stage judgeable by ear on its own.
    """
    started_at = perf_counter()
    loaded = looped.loaded
    paths = looped_paths(Path(out_dir), loaded.instrument.id)
    material = loaded.instrument.material
    written = _write_recordings(looped, paths.samples_dir)
    _write_calibrated(looped, written, paths, settings.loop)
    dump_notes(
        note_records(material, loaded.audio, written.indices),
        paths.notes_json,
        tracked_ccs=tracked_ccs(material),
        tempo_bpm=loaded.instrument.tempo_bpm,
    )
    played = write_dataset_instruments(
        SourceDataset(path=paths.notes_json, samples_dir=paths.samples_dir),
        settings=instruments,
    )
    auditions = _write_auditions(looped, paths.auditions_dir, settings.loop, settings.progress)
    paths.loops_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(paths.loops_json, looped_document(looped))
    return LoopedInstrumentArtifacts(
        instrument_id=loaded.instrument.id,
        paths=paths,
        recordings=len(written.records),
        looped=looped.looped_recordings,
        auditions=auditions,
        instruments=played,
        elapsed_s=perf_counter() - started_at,
    )


def loop_project(
    manifest: Manifest,
    out_dir: Path | str,
    settings: OptimizeSettings,
    instruments: InstrumentSettings,
) -> list[LoopedInstrumentArtifacts]:
    """Settle every instrument's loops and write each one's looped dataset under ``out_dir``.

    Instruments share the root, each contributing its own ``<id>.notes.json`` and ``<id>/`` pair, so one
    loop run over a project yields one looped dataset per instrument in the shape ingest expects.
    """
    out_dir = Path(out_dir)
    return [
        dump_looped(run_loops(load_run_audio(instrument, settings), settings), out_dir, settings, instruments)
        for instrument in manifest.instruments
    ]
