from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.artifacts.serialize import (
    ReducedDocument,
    WrittenSampleRecord,
    reduction_document,
    write_json,
)
from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NOTES_SUFFIX, NoteRecord, dump_notes
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, Manifest, NoteEvent
from optisample.music import pitch_label
from optisample.optimize.orchestrate import RunInputs, prepare_run
from optisample.optimize.orchestrate.audio import load_run_audio
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.keys import SampleKey, nearest_key
from optisample.optimize.reduce.summary import KeptRecording
from optisample.optimize.tasks import (
    AudioMap,
    EvalContext,
    Event,
    PitchTask,
    render_event,
)
from optisample.progress import ProgressSink

_AUDITIONS_DIR: Final = "auditions"
_REDUCTION_DIR: Final = "reduction"
_REDUCTION_JSON: Final = "reduction.json"
_REFERENCE_STEM: Final = "reference"
_AUDITION_LABEL: Final = "Rendering auditions"
_REFERENCE_FILES: Final = 1  # each pitch's audition folder opens with the recording the rest are judged against


@dataclass(frozen=True)
class ReducedPaths:
    """Where one instrument's reduced dataset and its inspection material land under the output root.

    ``notes_json`` and ``samples_dir`` are the sibling pair a later ingest resolves by default, so the
    output root is itself a NoteExtractor dataset. What the stage decided sits apart under
    ``reduction_json`` and ``auditions_dir``, leaving the dataset holding recordings alone.
    """

    notes_json: Path
    samples_dir: Path
    reduction_json: Path
    auditions_dir: Path


@dataclass(frozen=True)
class ReducedInstrument:
    """One instrument's reduced dataset: where each part landed and how much of it there is."""

    instrument_id: str
    paths: ReducedPaths
    survivors: int
    notes: int
    auditions: int
    elapsed_s: float


@dataclass(frozen=True)
class _Survivors:
    """The recordings a reduced dataset holds: what was written, and the index each key joins on."""

    records: tuple[WrittenSampleRecord, ...]
    indices: dict[SampleKey, int]


def reduced_paths(out_dir: Path, instrument_id: str) -> ReducedPaths:
    """The tree one instrument's reduce run writes under ``out_dir``."""
    reduction_dir = out_dir / _REDUCTION_DIR / instrument_id
    return ReducedPaths(
        notes_json=out_dir / f"{instrument_id}{NOTES_SUFFIX}",
        samples_dir=out_dir / instrument_id,
        reduction_json=reduction_dir / _REDUCTION_JSON,
        auditions_dir=reduction_dir / _AUDITIONS_DIR,
    )


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
        name = f"{index:04d}_{recording.key.label}.wav"
        write_wav(samples_dir / name, stored, sample_rate)
        indices[recording.key] = index
        records.append(
            WrittenSampleRecord(
                index=index,
                key=recording.key.label,
                file=name,
                frames=int(stored.size),
                duration_s=int(stored.size) / sample_rate,
            )
        )

    return _Survivors(records=tuple(records), indices=indices)


def _keys_by_pitch(audio: AudioMap) -> dict[int, list[SampleKey]]:
    """The survivors available at each pitch, in key order, which is what a played note is routed among."""
    available: dict[int, list[SampleKey]] = {}
    for key in sorted(audio):
        available.setdefault(key.pitch, []).append(key)

    return available


def _note_records(
    material: Sequence[NoteEvent],
    audio: AudioMap,
    survivors: _Survivors,
) -> list[NoteRecord]:
    """Every played note as a dataset entry, pointing at the survivor it is scored against.

    A note is routed by :func:`~optisample.optimize.reduce.keys.nearest_key`, the same lookup
    :func:`~optisample.optimize.reduce.events.merge_events` scores it through, so the reduced dataset
    pairs each note with exactly the recording the objective already measured it against. An event
    standing for several played notes is written once per note, so the material keeps its weight.
    """
    available = _keys_by_pitch(audio)
    return [
        NoteRecord(
            index=survivors.indices[nearest_key(available[event.pitch], event.velocity)],
            pitch=event.pitch,
            velocity=event.velocity,
            duration_s=event.duration_s,
            cc_averages=event.cc_averages,
        )
        for event in material
        for _ in range(event.count)
    ]


def _tracked_ccs(material: Sequence[NoteEvent]) -> list[int]:
    """Every controller the material reports an average for, which the dataset declares up front."""
    return sorted({controller for event in material for controller in event.cc_averages})


def _encoding_stem(params: EncodingParams) -> str:
    """The audition filename for one shortlisted encoding: every axis it asks for, in the sweep's order."""
    parts = [f"r{params.target_rate}", f"d{params.depth_bits}"]
    if params.compress:
        parts.append("c")
    if params.loop:
        parts.append("loop")

    return "_".join(parts)


def _audition(task: PitchTask, event: Event, params: EncodingParams, context: EvalContext) -> Signal:
    """The audio one shortlisted encoding produces for a pitch's representative note.

    Stored and played back the way the sweep scores it, so what lands on disk is the reconstruction the
    objective measures. The dither runs off the surrogate's own fixed seed, so a pitch renders the same
    audition however many others were rendered before it.
    """
    stored = encode(
        task.representative,
        context.sample_rate,
        params,
        EncodeContext(root_pitch=task.pitch, config=context.encode),
    )
    return render_event(stored, event, pitch=task.pitch, sample_rate=context.sample_rate)


def _write_auditions(
    task: PitchTask,
    shortlist: Sequence[EncodingParams],
    out_dir: Path,
    context: EvalContext,
) -> int:
    """Render one pitch's representative note through every shortlisted encoding, beside its reference.

    Writing the reference alongside them makes the folder a listening test: the recording as it stands,
    next to each encoding the allocation may buy, so the shortlist is judged by ear while the sweep is
    still ahead. Returns how many files were written.
    """
    directory = out_dir / pitch_label(task.pitch)
    directory.mkdir(parents=True, exist_ok=True)
    event = task.representative_event
    write_wav(directory / f"{_REFERENCE_STEM}.wav", event.scored_reference(context.sample_rate), context.sample_rate)
    for params in shortlist:
        write_wav(
            directory / f"{_encoding_stem(params)}.wav", _audition(task, event, params, context), context.sample_rate
        )

    return len(shortlist) + _REFERENCE_FILES


def _write_all_auditions(inputs: RunInputs, out_dir: Path, progress: ProgressSink) -> int:
    """Render the shortlist of every played pitch and return how many audition files were written."""
    shortlists = inputs.reduction.shortlists()
    tracked = progress.track(inputs.tasks, label=_AUDITION_LABEL, total=len(inputs.tasks))
    return sum(_write_auditions(task, shortlists[task.pitch], out_dir, inputs.context) for task in tracked)


def _reduced_document(
    instrument_id: str,
    sample_rate: int,
    survivors: _Survivors,
    inputs: RunInputs,
    settings: OptimizeSettings,
) -> ReducedDocument:
    """What the reduce run decided, as the document written beside the dataset it produced."""
    return ReducedDocument(
        instrument_id=instrument_id,
        dedupe_key=settings.reduce.dedupe.key,
        sample_rate=sample_rate,
        samples=list(survivors.records),
        reduction=reduction_document(inputs.reduction),
    )


def dump_reduced(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    out_dir: Path | str,
    settings: OptimizeSettings,
) -> ReducedInstrument:
    """Run the pre-optimization stage for one instrument and write what it decided as a dataset.

    The output root is itself a NoteExtractor dataset -- one WAV per surviving recording and one note
    per note the material plays -- so an allocation run picks up from here and reaches the sweep having
    paid only the ingest. Beside it, ``reduction.json`` states what each axis came down to and
    ``auditions/`` holds every shortlisted encoding rendered to audio, so the stage is inspectable and
    audible on its own.

    A dataset reproduces its survivors exactly under the key it was reduced with: each note carries the
    identity of the recording serving it, so re-running dedup over those recordings keeps the same one
    per slot. A coarser key projects several identities onto one survivor, which then reports the
    identity of the first note reaching it.
    """
    started_at = perf_counter()
    paths = reduced_paths(Path(out_dir), instrument.id)
    inputs = prepare_run(instrument, audio, sample_rate, settings)
    survivors = _write_survivors(audio, inputs.reduction.recordings, sample_rate, paths.samples_dir)
    notes = _note_records(instrument.material, audio, survivors)
    dump_notes(notes, paths.notes_json, tracked_ccs=_tracked_ccs(instrument.material))
    auditions = _write_all_auditions(inputs, paths.auditions_dir, settings.progress)
    paths.reduction_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(paths.reduction_json, _reduced_document(instrument.id, sample_rate, survivors, inputs, settings))
    return ReducedInstrument(
        instrument_id=instrument.id,
        paths=paths,
        survivors=len(survivors.records),
        notes=len(notes),
        auditions=auditions,
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
        audio, sample_rate = load_run_audio(instrument, settings)
        results.append(dump_reduced(instrument, audio, sample_rate, out_dir, settings))

    return results
