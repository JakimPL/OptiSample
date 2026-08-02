from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.artifacts.dataset import note_records, tracked_ccs, write_recording
from optisample.artifacts.documents.reduction import (
    ReducedDocument,
    WrittenSampleRecord,
    reduction_document,
    screen_record,
)
from optisample.artifacts.paths import ReducedPaths, reduced_paths
from optisample.artifacts.serialize import write_json
from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.io.audio import write_wav
from optisample.io.note_extractor import dump_notes
from optisample.keys import SampleKey
from optisample.metrics.base import Signal
from optisample.model import Manifest
from optisample.music import pitch_label
from optisample.optimize.orchestrate import RunInputs, prepare_run
from optisample.optimize.orchestrate.audio import LoadedInstrument, load_run_audio
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.summary import KeptRecording
from optisample.optimize.reduce.trim import RecordingScreen
from optisample.optimize.tasks import (
    AudioMap,
    EvalContext,
    Event,
    PitchTask,
    render_event,
)
from optisample.progress import ProgressSink

_REFERENCE_STEM: Final = "reference"
_AUDITION_LABEL: Final = "Rendering auditions"
_REFERENCE_FILES: Final = 1  # each pitch's audition folder opens with the recording the rest are judged against


@dataclass(frozen=True)
class ReducedInstrument:
    """One instrument's reduced dataset: where each part landed and how much of it there is.

    ``screen`` states what admitting the recordings cost, so a reader sees the recordings the dataset
    leaves out beside the ones it wrote.
    """

    instrument_id: str
    paths: ReducedPaths
    survivors: int
    notes: int
    auditions: int
    screen: RecordingScreen
    elapsed_s: float


@dataclass(frozen=True)
class _Survivors:
    """The recordings a reduced dataset holds: what was written, and the index each key joins on."""

    records: tuple[WrittenSampleRecord, ...]
    indices: dict[SampleKey, int]


def stored_frames(recording: KeptRecording, sample_rate: int) -> int:
    """Frames a survivor is written with: enough to hold everything its pitch asks of it.

    Rounding the requirement up keeps the written recording at least as long as
    :func:`~optisample.optimize.reduce.dedupe.required_duration_s` demands, so a dataset read back
    reports the coverage it was written with. A recording holding less than that is written whole,
    keeping as much of the note as was recorded.
    """
    return ceil(recording.required_duration_s * sample_rate)


def _write_survivors(
    audio: AudioMap,
    recordings: Sequence[KeptRecording],
    sample_rate: int,
    samples_dir: Path,
) -> _Survivors:
    """Write one WAV per surviving recording under ``samples_dir`` and state how the notes join to them.

    Files are named ``{index:04d}_{key label}.wav``: the leading index is the render index a later
    ingest reads back, and the rest of the name says which recording it holds. Survivors are numbered in
    key order, so the same recordings reduce to the same filenames on every run.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    records: list[WrittenSampleRecord] = []
    indices: dict[SampleKey, int] = {}
    for index, recording in enumerate(recordings):
        stored = audio[recording.key][: stored_frames(recording, sample_rate)]
        records.append(write_recording(stored, recording.key, index, sample_rate, samples_dir))
        indices[recording.key] = index

    return _Survivors(records=tuple(records), indices=indices)


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
    objective measures. The dither runs off the surrogate's own fixed seed, so a pitch renders the same
    audition however many others were rendered before it.
    """
    stored = encode(
        task.representative,
        context.sample_rate,
        params,
        EncodeContext(root_pitch=task.pitch, config=context.encode, settled=task.settled),
    )
    return render_event(stored, event, pitch=task.pitch, sample_rate=context.sample_rate)


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
) -> ReducedInstrument:
    """Run the pre-optimization stage for one instrument and write what it decided as a dataset.

    The output root is itself a NoteExtractor dataset -- one WAV per surviving recording and one note
    per note the material plays -- so an allocation run picks up from here and reaches the sweep having
    paid only the ingest. Beside it, ``reduction.json`` states what each axis came down to and
    ``auditions/`` holds every swept encoding rendered to audio, so the stage is inspectable and audible
    on its own.

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
    survivors = _write_survivors(audio, inputs.reduction.recordings, sample_rate, paths.samples_dir)
    notes = note_records(instrument.material, audio, survivors.indices)
    dump_notes(notes, paths.notes_json, tracked_ccs=tracked_ccs(instrument.material))
    auditions = _write_all_auditions(inputs, paths.auditions_dir, settings.progress)
    paths.reduction_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(paths.reduction_json, _reduced_document(loaded, survivors, inputs, settings))
    return ReducedInstrument(
        instrument_id=instrument.id,
        paths=paths,
        survivors=len(survivors.records),
        notes=len(notes),
        auditions=auditions,
        screen=loaded.screen,
        elapsed_s=perf_counter() - started_at,
    )


def reduce_project(manifest: Manifest, out_dir: Path | str, settings: OptimizeSettings) -> list[ReducedInstrument]:
    """Reduce every instrument of a loaded manifest and write each one's dataset under ``out_dir``.

    Instruments share the root, each contributing its own ``<id>.notes.json`` and ``<id>/`` pair, so one
    reduce run over a project yields one reduced dataset per instrument in the shape ingest expects.
    """
    out_dir = Path(out_dir)
    results: list[ReducedInstrument] = []
    for instrument in manifest.instruments:
        loaded = load_run_audio(instrument, settings)
        looped = run_loops(loaded, settings)
        results.append(dump_reduced(looped, out_dir, settings))

    return results
