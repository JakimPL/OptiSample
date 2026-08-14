from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.artifacts.calibrated import CalibratedRecording, CalibratedRun, write_calibrated
from optisample.artifacts.dataset import note_records, tracked_ccs, write_recording
from optisample.artifacts.documents.reduction import (
    ReducedDocument,
    WrittenSampleRecord,
    reduction_document,
    screen_record,
)
from optisample.artifacts.instruments.dump import InstrumentSettings, WrittenInstruments, write_dataset_instruments
from optisample.artifacts.paths import ReducedPaths, reduced_paths
from optisample.artifacts.serialize import write_json
from optisample.config.loop import LoopConfig
from optisample.dsp.surrogate import EncodingParams
from optisample.io.audio import write_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import dump_notes
from optisample.keys import SampleKey
from optisample.loop.settle import Settlement, StoredLoop
from optisample.metrics.base import Signal
from optisample.model import Manifest
from optisample.music import pitch_label
from optisample.optimize.orchestrate import RunInputs, prepare_run
from optisample.optimize.orchestrate.audio import LoadedInstrument, load_run_audio
from optisample.optimize.orchestrate.looping import LoopedInstrument, Settlements, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.summary import KeptRecording
from optisample.optimize.reduce.trim import RecordingScreen
from optisample.optimize.tasks import (
    AudioMap,
    EvalContext,
    Event,
    PitchTask,
    audition_sample,
    render_event,
)
from optisample.progress import ProgressSink

_REFERENCE_STEM: Final = "reference"
_AUDITION_LABEL: Final = "Rendering auditions"
_STAGE: Final = "reduce"  # the run behind every container this stage writes, stamped into its provenance
_NO_SETTLEMENT: Final = None  # what a run storing no loops leaves a recording with, which offers none
_NOTHING_OFFERED: Final = ()  # the loops a survivor no settlement was reached for hands its container
_REFERENCE_FILES: Final = 1  # each pitch's audition folder opens with the recording the rest are judged against


@dataclass(frozen=True)
class ReducedInstrument:
    """One instrument's reduced dataset: where each part landed and how much of it there is.

    ``screen`` states what admitting the recordings cost, so a reader sees the recordings the dataset
    leaves out beside the ones it wrote, and ``instruments`` where each survivor landed as the standalone
    instrument a player loads it through. ``looped`` counts the survivors carrying a loop the trimmed span
    still holds, which is how many of them a player sustains past the end of the audio.
    """

    instrument_id: str
    paths: ReducedPaths
    survivors: int
    looped: int
    notes: int
    auditions: int
    instruments: WrittenInstruments
    screen: RecordingScreen
    elapsed_s: float


@dataclass(frozen=True)
class _Survivors:
    """The recordings a reduced dataset holds: what was written, where each key joins, and what wraps.

    ``looped`` counts the survivors whose written span still reaches a loop the stage settled over them,
    which is how many of the dataset's instruments sustain a note past the end of their own audio.
    """

    records: tuple[WrittenSampleRecord, ...]
    indices: dict[SampleKey, int]
    looped: int


def stored_frames(recording: KeptRecording, sample_rate: int) -> int:
    """Frames a survivor is written with: enough to hold everything its pitch asks of it.

    Rounding the requirement up keeps the written recording at least as long as
    :func:`~optisample.optimize.reduce.dedupe.required_duration_s` demands, so a dataset read back
    reports the coverage it was written with. A recording holding less than that is written whole,
    keeping as much of the note as was recorded.
    """
    return ceil(recording.required_duration_s * sample_rate)


def held_loops(settlement: Settlement | None, frames: int) -> tuple[StoredLoop, ...]:
    """The loops a settlement offers that the ``frames`` a survivor is written with still reach.

    A survivor is trimmed to the span its material asks of it (:func:`stored_frames`), so a loop settled
    over the whole recording may sit past where the written one ends. What a container states is the
    offers a player loading that file can reach, which keeps the loops beside the audio holding them and
    leaves the trimmed dataset saying what it can play. A survivor no loop was settled for offers none.
    """
    if settlement is None:
        return _NOTHING_OFFERED

    return tuple(stored for stored in settlement.offered if stored.loop.end <= frames)


def _survivors(
    audio: AudioMap,
    recordings: Sequence[KeptRecording],
    settlements: Settlements,
    sample_rate: int,
) -> tuple[CalibratedRecording, ...]:
    """Each survivor as the trimmed recording the stage writes, with the loops that span still holds.

    Survivors are numbered in the order the reduction kept them, which is key order, so the same
    recordings reduce to the same filenames on every run and one recording's WAV, container and
    instruments all share a stem.
    """
    kept: list[CalibratedRecording] = []
    for index, recording in enumerate(recordings):
        stored = audio[recording.key][: stored_frames(recording, sample_rate)]
        kept.append(
            CalibratedRecording(
                key=recording.key,
                index=index,
                signal=stored,
                offered=held_loops(settlements.get(recording.key, _NO_SETTLEMENT), int(stored.size)),
            )
        )

    return tuple(kept)


def _write_survivors(recordings: Sequence[CalibratedRecording], run: CalibratedRun) -> _Survivors:
    """Write each survivor as a WAV and the container carrying it, and state how the notes join to them.

    Files are named ``{index:04d}_{key label}``: the leading index is the render index a later ingest
    reads back, and the rest of the name says which recording it holds. Both files of one recording share
    that stem, so the container states the very audio standing beside it.
    """
    run.samples_dir.mkdir(parents=True, exist_ok=True)
    records = tuple(
        write_recording(recording.signal, recording.key, recording.index, run.sample_rate, run.samples_dir)
        for recording in recordings
    )
    write_calibrated(recordings, run)
    return _Survivors(
        records=records,
        indices={recording.key: recording.index for recording in recordings},
        looped=sum(1 for recording in recordings if recording.offered),
    )


def _calibrated_run(loaded: LoadedInstrument, paths: ReducedPaths, config: LoopConfig) -> CalibratedRun:
    """What every container this stage writes shares: where they land, the run behind them, and the rate."""
    return CalibratedRun(
        samples_dir=paths.samples_dir,
        instrument_id=loaded.instrument.id,
        stage=_STAGE,
        sample_rate=loaded.sample_rate,
        config=config,
    )


def _encoding_stem(params: EncodingParams) -> str:
    """The audition filename for one swept encoding: every axis it asks for, in the sweep's order."""
    parts = [f"r{params.target_rate}", f"d{params.depth_bits}"]
    if params.compress:
        parts.append("c")
    if params.loop_index is not None:
        parts.append(f"loop{params.loop_index}")

    return "_".join(parts)


def _audition(task: PitchTask, event: Event, params: EncodingParams, context: EvalContext) -> Signal:
    """The audio one swept encoding produces for a pitch's representative note.

    Stored and played back the way the sweep scores it, so what lands on disk is the reconstruction the
    objective measures.
    """
    return render_event(
        audition_sample(task, params, context), event, pitch=task.pitch, sample_rate=context.sample_rate
    )


def _write_auditions(
    task: PitchTask,
    encodings: Sequence[EncodingParams],
    out_dir: Path,
    context: EvalContext,
) -> int:
    """Render one pitch's representative note through every encoding it is swept over, beside its reference.

    Writing the reference alongside them makes the folder a listening test: the recording as it stands,
    next to each encoding the allocation may buy, so the format the reduction settled is judged by ear
    while the sweep is still ahead. Returns how many files were written.
    """
    directory = out_dir / pitch_label(task.pitch)
    directory.mkdir(parents=True, exist_ok=True)
    event = task.representative_event
    write_wav(directory / f"{_REFERENCE_STEM}.wav", event.scored_span(context.sample_rate), context.sample_rate)
    for params in encodings:
        write_wav(
            directory / f"{_encoding_stem(params)}.wav", _audition(task, event, params, context), context.sample_rate
        )

    return len(encodings) + _REFERENCE_FILES


def _write_all_auditions(inputs: RunInputs, out_dir: Path, progress: ProgressSink) -> int:
    """Render every played pitch's swept encodings and return how many audition files were written."""
    encodings = inputs.reduction.encodings()
    tracked = progress.track(inputs.tasks, label=_AUDITION_LABEL, total=len(inputs.tasks))
    return sum(_write_auditions(task, encodings[task.pitch], out_dir, inputs.context) for task in tracked)


def _reduced_document(
    loaded: LoadedInstrument,
    survivors: _Survivors,
    inputs: RunInputs,
    settings: OptimizeSettings,
) -> ReducedDocument:
    """What the reduce run decided, as the document written beside the dataset it produced."""
    return ReducedDocument(
        instrument_id=loaded.instrument.id,
        dedupe_key=settings.reduce.dedupe.key,
        sample_rate=loaded.sample_rate,
        samples=list(survivors.records),
        screen=screen_record(loaded.screen),
        reduction=reduction_document(inputs.reduction),
    )


def dump_reduced(
    looped: LoopedInstrument,
    out_dir: Path | str,
    settings: OptimizeSettings,
    instruments: InstrumentSettings,
) -> ReducedInstrument:
    """Run the pre-optimization stage for one instrument and write what it decided as a dataset.

    The output root is itself a NoteExtractor dataset -- one WAV per surviving recording and one note
    per note the material plays -- so an allocation run picks up from here and reaches the sweep having
    paid only the ingest. Every survivor is written a second time as the ``.sample`` carrying its
    calibrated form (:func:`~optisample.artifacts.calibrated.write_calibrated`), holding the loops settled
    over it that the trimmed span still reaches, and a third time as the standalone instruments of each
    format (:func:`~optisample.artifacts.instruments.dump.write_dataset_instruments`), which read those
    containers back for the region each survivor wraps on, so what the stage kept is playable in a tracker
    as it stands. Beside them, ``reduction.json`` states what each axis came down to and ``auditions/``
    holds every swept encoding rendered to audio, so the stage is inspectable and audible on its own.

    A dataset reproduces its survivors exactly under the key it was reduced with: each note carries the
    identity of the recording serving it, so re-running dedup over those recordings keeps the same one
    per slot. A coarser key projects several identities onto one survivor, which then reports the
    identity of the first note reaching it.

    What the written dataset holds is what the screen admitted: the recordings that carried signal,
    trimmed to the span worth storing, and the notes those recordings can serve.
    """
    started_at = perf_counter()
    loaded = looped.loaded
    instrument, audio, sample_rate = loaded.instrument, loaded.audio, loaded.sample_rate
    paths = reduced_paths(Path(out_dir), instrument.id)
    inputs = prepare_run(instrument, looped.recordings, settings)
    survivors = _write_survivors(
        _survivors(audio, inputs.reduction.recordings, looped.settlements, sample_rate),
        _calibrated_run(loaded, paths, settings.loop),
    )
    notes = note_records(instrument.material, audio, survivors.indices)
    dump_notes(
        notes,
        paths.notes_json,
        tracked_ccs=tracked_ccs(instrument.material),
        tempo_bpm=instrument.tempo_bpm,
    )
    played = write_dataset_instruments(
        SourceDataset(path=paths.notes_json, samples_dir=paths.samples_dir),
        settings=instruments,
    )
    auditions = _write_all_auditions(inputs, paths.auditions_dir, settings.progress)
    paths.reduction_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(paths.reduction_json, _reduced_document(loaded, survivors, inputs, settings))
    return ReducedInstrument(
        instrument_id=instrument.id,
        paths=paths,
        survivors=len(survivors.records),
        looped=survivors.looped,
        notes=len(notes),
        auditions=auditions,
        instruments=played,
        screen=loaded.screen,
        elapsed_s=perf_counter() - started_at,
    )


def reduce_project(
    manifest: Manifest,
    out_dir: Path | str,
    settings: OptimizeSettings,
    instruments: InstrumentSettings,
) -> list[ReducedInstrument]:
    """Reduce every instrument of a loaded manifest and write each one's dataset under ``out_dir``.

    Instruments share the root, each contributing its own ``<id>.notes.json`` and ``<id>/`` pair, so one
    reduce run over a project yields one reduced dataset per instrument in the shape ingest expects.
    """
    out_dir = Path(out_dir)
    results: list[ReducedInstrument] = []
    for instrument in manifest.instruments:
        loaded = load_run_audio(instrument, settings)
        looped = run_loops(loaded, settings)
        results.append(dump_reduced(looped, out_dir, settings, instruments))

    return results
