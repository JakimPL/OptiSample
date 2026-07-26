from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.artifacts.context import (
    DumpContext,
    DumpResult,
    DumpSettings,
    PlanArtifacts,
)
from optisample.artifacts.serialize import (
    NoteMetricRecord,
    RenderedNote,
    RepresentativeEventRecord,
    event_records,
    metrics_document,
    note_record,
    write_json,
    write_text,
)
from optisample.artifacts.units import PlanKind, Unit, make_kind
from optisample.dsp.surrogate import render
from optisample.dsp.timebase import seconds_to_frames
from optisample.io.audio import write_wav
from optisample.io.render import openmpt123_available, render_module
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, Manifest, NoteEvent
from optisample.music import note_name
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping import optimize_instrument_grouped
from optisample.optimize.orchestrate import optimize_instrument, prepare_run
from optisample.optimize.orchestrate.audio import load_instrument_audio
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import AudioMap, Event, PitchTask

_Optimizer = Callable[[InstrumentSpec, AudioMap, int, OptimizeSettings], InstrumentPlan | GroupedInstrumentPlan]


@dataclass(frozen=True)
class _Strategy:
    """One allocation strategy: its subdirectory name and the optimizer that produces its plan."""

    name: str
    optimize: _Optimizer


_UNGROUPED: Final = _Strategy("ungrouped", optimize_instrument)
_GROUPED: Final = _Strategy("grouped", optimize_instrument_grouped)
_MODULE_STEM: Final = "module"


def _representative_event(task: PitchTask) -> Event:
    """The note worth listening to for a pitch: the most-played dynamic (ties -> the longest)."""
    return max(task.events, key=lambda event: (event.weight, event.duration_s))


def _rendered_note(
    dump_context: DumpContext, kind: PlanKind, unit: Unit, task: PitchTask, event: Event
) -> tuple[Signal, int, str]:
    """Audio the module produces for one note: the real engine when available, else the surrogate."""
    if dump_context.settings.render_ground_truth and openmpt123_available():
        note = NoteEvent(pitch=task.pitch, velocity=event.velocity, duration_s=event.duration_s)
        audio, rate = render_module(kind.make_module([note]), dump_context.settings.render)
        return audio, rate, "openmpt123"
    volume = dump_context.eval_context.velocity_map.volume(event.velocity)
    candidate = render(
        unit.stored, dump_context.sample_rate, pitch=task.pitch, volume=volume, duration_s=event.duration_s
    )
    return candidate, dump_context.sample_rate, "surrogate"


def _note_record(
    kind: PlanKind, unit: Unit, task: PitchTask, out_dir: Path, dump_context: DumpContext
) -> NoteMetricRecord:
    """Write the reference/rendered A/B pair for one pitch and return its metric record."""
    events, contribution = event_records(unit.stored, task, dump_context.eval_context)
    rep = _representative_event(task)
    stem = f"p{task.pitch:03d}_{note_name(task.pitch)}"
    reference = rep.reference[: seconds_to_frames(rep.duration_s, dump_context.sample_rate)]
    write_wav(out_dir / "compare" / f"{stem}_ref.wav", reference, dump_context.sample_rate)
    rendered, rate, source = _rendered_note(dump_context, kind, unit, task, rep)
    write_wav(out_dir / "compare" / f"{stem}_render.wav", rendered, rate)
    outcome = RenderedNote(
        served_by=unit.label,
        representative=unit.representative,
        source=source,
        rate=rate,
        event=RepresentativeEventRecord(velocity=rep.velocity, duration_s=rep.duration_s),
    )
    return note_record(task, events, contribution, outcome)


def _write_plan_docs(kind: PlanKind, out_dir: Path) -> None:
    """Write the human report and the plan + velocity-map JSON documents."""
    write_text(out_dir / "report.txt", kind.report_text)
    write_json(out_dir / "plan.json", kind.plan_document)
    write_json(out_dir / "velocity_map.json", kind.plan_document.velocity_map)


def _write_sample_wavs(kind: PlanKind, out_dir: Path) -> None:
    """Decode every stored sample back to a float WAV under ``samples/`` (bit-identical to the module)."""
    for unit in kind.units:
        write_wav(out_dir / "samples" / f"{unit.label}.wav", unit.stored.pcm, unit.stored.sample_rate)


def _write_module_and_render(kind: PlanKind, out_dir: Path, dump_context: DumpContext) -> bool:
    """Write the module in its own format and, when openmpt123 is available and asked for, render it."""
    kind.module.save(out_dir / f"{_MODULE_STEM}{kind.module.extension}")
    if not (dump_context.settings.render_ground_truth and openmpt123_available()):
        return False
    (out_dir / "render").mkdir(parents=True, exist_ok=True)
    audio, rate = render_module(kind.module, dump_context.settings.render)
    write_wav(out_dir / "render" / f"{_MODULE_STEM}.wav", audio, rate)
    return True


def _write_metrics(kind: PlanKind, out_dir: Path, dump_context: DumpContext) -> None:
    """Score and A/B-render every covered pitch; ``metrics.json``'s objective reproduces ``plan.objective``."""
    notes = [_note_record(kind, unit, task, out_dir, dump_context) for unit in kind.units for task in unit.tasks]
    document = metrics_document(
        kind.name, kind.plan_document.instrument_id, dump_context.sample_rate, kind.plan_document.objective, notes
    )
    write_json(out_dir / "metrics.json", document)


def _dump_plan(kind: PlanKind, out_dir: Path, dump_context: DumpContext, started_at: float) -> PlanArtifacts:
    """Write every artifact for one strategy and return a summary of what landed on disk.

    ``started_at`` is the :func:`time.perf_counter` reading taken before the optimize call, so the
    reported ``elapsed_s`` spans the whole strategy (optimize + this dump), not just the I/O here.
    """
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)
    _write_plan_docs(kind, out_dir)
    _write_sample_wavs(kind, out_dir)
    rendered = _write_module_and_render(kind, out_dir, dump_context)
    _write_metrics(kind, out_dir, dump_context)
    return PlanArtifacts(
        name=kind.name,
        reason=None,
        rendered=rendered,
        objective=kind.plan_document.objective,
        used_bytes=kind.plan_document.budget.used_bytes,
        elapsed_s=perf_counter() - started_at,
    )


def _optimize_and_dump(
    instrument: InstrumentSpec, out_dir: Path, dump_context: DumpContext, strategy: _Strategy
) -> PlanArtifacts:
    """Optimize one strategy and dump it; on an infeasible budget, record why instead of raising."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    try:
        plan = strategy.optimize(
            instrument, dump_context.audio, dump_context.sample_rate, dump_context.settings.optimize
        )
        kind = make_kind(plan, dump_context)
    except BudgetInfeasibleError as exc:
        write_text(out_dir / "INFEASIBLE.txt", f"{strategy.name} allocation is infeasible at this budget:\n{exc}\n")
        return PlanArtifacts(
            strategy.name,
            reason=str(exc),
            rendered=False,
            objective=None,
            used_bytes=None,
            elapsed_s=perf_counter() - started_at,
        )
    return _dump_plan(kind, out_dir, dump_context, started_at)


def dump_instrument(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    out_dir: Path | str,
    settings: DumpSettings,
) -> DumpResult:
    """Optimize one instrument (both strategies) and write every inspection artifact under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, context, tasks = prepare_run(instrument, audio, sample_rate, settings.optimize)
    dump_context = DumpContext(
        audio=audio,
        sample_rate=sample_rate,
        material=tuple(instrument.material or []),
        eval_context=context,
        tasks_by_pitch={task.pitch: task for task in tasks},
        settings=settings,
    )
    strategies = [
        strategy for strategy, enabled in ((_UNGROUPED, settings.ungrouped), (_GROUPED, settings.grouped)) if enabled
    ]
    plans = [_optimize_and_dump(instrument, out_dir / strategy.name, dump_context, strategy) for strategy in strategies]
    return DumpResult(instrument_id=instrument.id, directory=out_dir, plans=tuple(plans))


def dump_project(manifest: Manifest, out_dir: Path | str, settings: DumpSettings) -> list[DumpResult]:
    """Run :func:`dump_instrument` for every instrument in a loaded manifest under ``out_dir``."""
    out_dir = Path(out_dir)
    results: list[DumpResult] = []
    for instrument in manifest.instruments:
        audio, sample_rate = load_instrument_audio(
            instrument, settings.optimize.reduce.dedupe, settings.optimize.encode.loop
        )
        results.append(dump_instrument(instrument, audio, sample_rate, out_dir / instrument.id, settings))
    return results
