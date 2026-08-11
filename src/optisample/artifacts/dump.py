from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.artifacts.container import bank_contents, write_container
from optisample.artifacts.context import (
    NO_INSTRUMENTS,
    DumpContext,
    DumpResult,
    DumpSettings,
    PlanArtifacts,
)
from optisample.artifacts.documents.metrics import (
    NoteMetricRecord,
    RenderedNote,
    RepresentativeEventRecord,
    event_records,
    metrics_document,
    note_record,
)
from optisample.artifacts.instruments.dump import Recording, WrittenInstruments, write_instruments
from optisample.artifacts.paths import PlanPaths, plan_paths
from optisample.artifacts.serialize import write_json, write_text
from optisample.artifacts.units import PlanKind, Unit, make_kind
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NO_TEMPO
from optisample.io.render import openmpt123_available, render_module
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, Manifest, NoteEvent
from optisample.music import pitch_label
from optisample.optimize.dp import AllocationInfeasibleError
from optisample.optimize.grouping.optimize import allocate_instrument_grouped
from optisample.optimize.orchestrate import RunInputs, allocate_instrument, prepare_run
from optisample.optimize.orchestrate.audio import load_instrument_audio
from optisample.optimize.orchestrate.looping import run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import Event, PitchTask, StoredRecordings, render_event

_Allocator = Callable[[InstrumentSpec, RunInputs, OptimizeSettings], InstrumentPlan | GroupedInstrumentPlan]


@dataclass(frozen=True)
class _Strategy:
    """One allocation strategy: its subdirectory name and the allocator that produces its plan.

    Both allocators read the run the dumper prepared once, so the two strategies share the velocity map,
    the pitch tasks and the bandwidth pre-pass rather than each deriving its own.
    """

    name: str
    allocate: _Allocator


_UNGROUPED: Final = _Strategy("ungrouped", allocate_instrument)
_GROUPED: Final = _Strategy("grouped", allocate_instrument_grouped)
_NOTE_LABEL: Final = "Rendering note pairs"


def _rendered_note(
    dump_context: DumpContext,
    kind: PlanKind,
    unit: Unit,
    task: PitchTask,
    event: Event,
) -> tuple[Signal, int, str]:
    """Audio the module produces for one note: the real engine when available, else the surrogate."""
    if dump_context.settings.render_ground_truth and openmpt123_available():
        note = NoteEvent(pitch=task.pitch, velocity=event.velocity, duration_s=event.duration_s)
        audio, rate = render_module(kind.make_module([note]), dump_context.settings.render)
        return audio, rate, "openmpt123"
    candidate = render_event(unit.stored, event, pitch=task.pitch, sample_rate=dump_context.sample_rate)
    return candidate, dump_context.sample_rate, "surrogate"


def _note_record(
    kind: PlanKind,
    unit: Unit,
    task: PitchTask,
    paths: PlanPaths,
    dump_context: DumpContext,
) -> NoteMetricRecord:
    """Write the reference/rendered A/B pair for one pitch of one layer and return its metric record."""
    events, contribution = event_records(unit.stored, task, dump_context.eval_context)
    rep = task.representative_event
    band = kind.layers.bands[unit.layer].label
    stem = pitch_label(task.pitch)
    reference = rep.scored_reference(unit.stored, dump_context.sample_rate, pitch=task.pitch)
    write_wav(paths.reference_wav(band, stem), reference, dump_context.sample_rate)
    rendered, rate, source = _rendered_note(dump_context, kind, unit, task, rep)
    write_wav(paths.rendered_wav(band, stem), rendered, rate)
    outcome = RenderedNote(
        served_by=unit.label,
        layer=band,
        representative=unit.representative,
        source=source,
        rate=rate,
        event=RepresentativeEventRecord(velocity=rep.velocity, duration_s=rep.duration_s),
    )
    return note_record(task, events, contribution, outcome)


def _write_plan_docs(kind: PlanKind, paths: PlanPaths) -> None:
    """Write the human report and the plan and reduction JSON documents.

    The plan carries the velocity map the allocation measured, and the bank carries it beside the very
    waveforms it was measured from, so both readings of it state the same table.
    """
    write_text(paths.report, kind.report_text)
    write_json(paths.plan_json, kind.plan_document)
    write_json(paths.reduction_json, kind.plan_document.reduction)


def _write_sample_wavs(kind: PlanKind, paths: PlanPaths) -> None:
    """Decode every stored sample back to a float WAV under ``samples/`` (bit-identical to the module)."""
    for unit in kind.units:
        write_wav(paths.sample_wav(unit.label), unit.stored.pcm, unit.stored.sample_rate)


def _plan_recordings(kind: PlanKind) -> tuple[Recording, ...]:
    """Every stored sample as the recording an instrument of its own is written from.

    A unit is rooted at the pitch its recording was played at, which is the key the instrument written
    from it sounds, and carries the label the plan files all of its artifacts under.
    """
    return tuple(
        Recording(
            name=unit.label,
            root_pitch=unit.representative,
            sample_rate=unit.stored.sample_rate,
            signal=unit.stored.pcm,
        )
        for unit in kind.units
    )


def _write_sample_instruments(kind: PlanKind, paths: PlanPaths, settings: DumpSettings) -> WrittenInstruments:
    """Write every stored sample as the standalone instrument of each format, beside its own WAV.

    The bank holds the plan's instruments as the allocation grouped them -- one per velocity band, each
    reaching the keys its zones cover. These stand for the waveforms themselves: one instrument per stored
    sample, played down by the curve that sample's own level was fitted to, so a single voice out of a
    plan is auditioned in a tracker on its own.
    """
    return write_instruments(
        _plan_recordings(kind),
        paths.samples_dir,
        settings=settings.instruments,
        recorded_tempo_bpm=NO_TEMPO,
    )


def _write_bank(kind: PlanKind, paths: PlanPaths, target: ExportTarget) -> None:
    """Write the bank the whole plan is played through, as one archive and spread over the tree beside it.

    The archive is what a producer of sampled instruments ships: the manifest and every voice it names in
    one file, so the map of which dynamic plays which waveform travels with the waveforms. The same
    instruments land loose under ``instruments/`` as well, which is the form a tracker loading a single
    voice off the tree reads, and both are written from one description of the bank.
    """
    name = kind.plan_document.instrument_id
    contents = bank_contents(name, kind.module.song, kind.layout, kind.plan_document.velocity_map, target)
    write_container(paths.container(name), contents)
    for entry in contents.entries:
        path = paths.stored(entry.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry.data)


def _render_module(kind: PlanKind, paths: PlanPaths, dump_context: DumpContext) -> bool:
    """Render the module the plan auditions itself with, when openmpt123 is available and asked for.

    The module is the plan playing its own material, which is the audible check on the whole allocation.
    :func:`~optisample.io.render.render_module` hands it to the renderer through a path of its own, so
    what lands in the tree is the audio it produced.
    """
    if not (dump_context.settings.render_ground_truth and openmpt123_available()):
        return False

    paths.render_dir.mkdir(parents=True, exist_ok=True)
    audio, rate = render_module(kind.module, dump_context.settings.render)
    write_wav(paths.module_render, audio, rate)
    return True


def _write_metrics(
    kind: PlanKind,
    paths: PlanPaths,
    dump_context: DumpContext,
) -> None:
    """Score and A/B-render every covered pitch; ``metrics.json``'s objective reproduces ``plan.objective``."""
    covered = [(unit, task) for unit in kind.units for task in unit.tasks]
    progress = dump_context.settings.progress
    notes = [
        _note_record(kind, unit, task, paths, dump_context)
        for unit, task in progress.track(
            covered,
            label=_NOTE_LABEL,
            total=len(covered),
        )
    ]
    document = metrics_document(
        kind.name,
        kind.plan_document.instrument_id,
        dump_context.sample_rate,
        kind.plan_document.objective,
        notes,
    )
    write_json(paths.metrics_json, document)


def _dump_plan(
    kind: PlanKind,
    paths: PlanPaths,
    dump_context: DumpContext,
    started_at: float,
) -> PlanArtifacts:
    """Write every artifact for one strategy and return a summary of what landed on disk.

    ``started_at`` is the :func:`time.perf_counter` reading taken before the optimize call, so the
    reported ``elapsed_s`` spans the whole strategy (optimize + this dump), not just the I/O here.
    """
    paths.samples_dir.mkdir(parents=True, exist_ok=True)
    for band in kind.layers.bands:
        paths.layer_dir(band.label).mkdir(parents=True, exist_ok=True)

    _write_plan_docs(kind, paths)
    _write_sample_wavs(kind, paths)
    instruments = _write_sample_instruments(kind, paths, dump_context.settings)
    _write_bank(kind, paths, dump_context.settings.optimize.target)
    rendered = _render_module(kind, paths, dump_context)
    _write_metrics(kind, paths, dump_context)
    return PlanArtifacts(
        name=kind.name,
        reason=None,
        rendered=rendered,
        objective=kind.plan_document.objective,
        used_bytes=kind.plan_document.budget.used_bytes,
        instruments=instruments,
        elapsed_s=perf_counter() - started_at,
    )


def _optimize_and_dump(
    instrument: InstrumentSpec,
    paths: PlanPaths,
    dump_context: DumpContext,
    strategy: _Strategy,
) -> PlanArtifacts:
    """Allocate one strategy and dump it; on an allocation out of reach, record why instead of raising."""
    paths.directory.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    try:
        plan = strategy.allocate(instrument, dump_context.inputs, dump_context.settings.optimize)
        kind = make_kind(plan, dump_context)
    except AllocationInfeasibleError as exc:
        write_text(paths.infeasible, f"{strategy.name} allocation is infeasible at this budget:\n{exc}\n")
        return PlanArtifacts(
            strategy.name,
            reason=str(exc),
            rendered=False,
            objective=None,
            used_bytes=None,
            instruments=NO_INSTRUMENTS,
            elapsed_s=perf_counter() - started_at,
        )

    return _dump_plan(kind, paths, dump_context, started_at)


def dump_instrument(
    instrument: InstrumentSpec,
    recordings: StoredRecordings,
    out_dir: Path | str,
    settings: DumpSettings,
) -> DumpResult:
    """Optimize one instrument (both strategies) and write every inspection artifact under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_context = DumpContext(
        instrument=instrument,
        recordings=recordings,
        inputs=prepare_run(instrument, recordings, settings.optimize),
        settings=settings,
    )
    strategies = [
        strategy
        for strategy, enabled in (
            (_UNGROUPED, settings.ungrouped),
            (_GROUPED, settings.grouped),
        )
        if enabled
    ]
    plans = [
        _optimize_and_dump(
            instrument,
            plan_paths(out_dir, strategy.name),
            dump_context,
            strategy,
        )
        for strategy in strategies
    ]
    return DumpResult(
        instrument_id=instrument.id,
        directory=out_dir,
        plans=tuple(plans),
    )


def dump_project(
    manifest: Manifest,
    out_dir: Path | str,
    settings: DumpSettings,
) -> list[DumpResult]:
    """Run :func:`dump_instrument` for every instrument in a loaded manifest under ``out_dir``.

    Each instrument's own stages report their progress as they run, so this walks them plainly and lets
    those bars have the terminal line to themselves.
    """
    out_dir = Path(out_dir)
    results: list[DumpResult] = []
    for instrument in manifest.instruments:
        loaded = load_instrument_audio(
            instrument,
            settings.optimize.reduce,
            settings.optimize.loop,
            settings.progress,
        )
        looped = run_loops(loaded, settings.optimize)
        results.append(
            dump_instrument(
                loaded.instrument,
                looped.recordings,
                out_dir / instrument.id,
                settings,
            ),
        )

    return results
