"""Dump inspectable artifacts for an optimized instrument, for offline quality analysis.

The optimizer's decisions and its objective are otherwise only visible as numbers in a report; this
module writes the *audible and machine-readable* evidence behind them to a directory tree, so a human
can listen to what was stored, compare it against the real recordings, and read the exact per-note
scores that drove the allocation. Nothing here changes the optimizer -- it re-runs it and serializes
the result: this module orchestrates and does file I/O, while :mod:`.serialize`/:mod:`.units` build the
typed documents and re-encoded samples it writes.

Per instrument, each strategy (``ungrouped`` = one sample per key, ``grouped`` = pitch zones) gets its
own subtree::

    <out>/<instrument>/<strategy>/
      module.it            the exported IT module
      report.txt           the human-readable optimizer report
      plan.json            zones/pitches, chosen params, bytes, objective, note map
      velocity_map.json    the velocity->volume map (a conversion-time artifact, not in the IT)
      samples/             each stored sample decoded back to float WAV (bit-identical to module.it)
      render/module.wav    openmpt123 render of the whole module (ground truth), if available
      compare/             per-pitch reference-vs-rendered WAV pairs (A/B by ear)
      metrics.json         per-note composite fidelity + sub-scores; its objective == plan's

The A/B render is the *real engine's* output (``openmpt123`` on a one-note module) when the binary is
installed, falling back to the numpy surrogate otherwise; ``metrics.json`` always reports the surrogate
scores, because those are exactly the objective the optimizer minimized (so its total reproduces
``plan.objective``). If a strategy is infeasible at the budget it writes ``INFEASIBLE.txt`` instead of a
plan -- useful precisely because grouping can fit where the ungrouped allocation cannot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from optisample.artifacts.context import DumpContext, DumpResult, DumpSettings, PlanArtifacts
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
from optisample.io.it_writer import write_it
from optisample.io.render import openmpt123_available, render_module
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, Manifest, NoteEvent
from optisample.music import note_name
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping import optimize_instrument_grouped
from optisample.optimize.orchestrate import OptimizeSettings, load_instrument_audio, optimize_instrument, prepare_run
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import AudioMap, Event, PitchTask

_Optimizer = Callable[[InstrumentSpec, AudioMap, int, OptimizeSettings], InstrumentPlan | GroupedInstrumentPlan]


@dataclass(frozen=True)
class _Strategy:
    """One allocation strategy: its subdirectory name and the optimizer that produces its plan."""

    name: str
    optimize: _Optimizer


_UNGROUPED = _Strategy("ungrouped", optimize_instrument)
_GROUPED = _Strategy("grouped", optimize_instrument_grouped)


def _representative_event(task: PitchTask) -> Event:
    """The note worth listening to for a pitch: the most-played dynamic (ties -> the longest)."""
    return max(task.events, key=lambda event: (event.weight, event.duration_s))


def _rendered_note(
    dctx: DumpContext, kind: PlanKind, unit: Unit, task: PitchTask, event: Event
) -> tuple[Signal, int, str]:
    """Audio the module produces for one note: the real engine when available, else the surrogate."""
    if dctx.settings.render_ground_truth and openmpt123_available():
        note = NoteEvent(pitch=task.pitch, velocity=event.velocity, duration_s=event.duration_s)
        audio, rate = render_module(kind.make_module([note]), dctx.settings.render)
        return audio, rate, "openmpt123"
    volume = dctx.ctx.velocity_map.volume(event.velocity)
    candidate = render(unit.stored, dctx.sample_rate, pitch=task.pitch, volume=volume, duration_s=event.duration_s)
    return candidate, dctx.sample_rate, "surrogate"


def _note_record(kind: PlanKind, unit: Unit, task: PitchTask, out_dir: Path, dctx: DumpContext) -> NoteMetricRecord:
    """Write the reference/rendered A/B pair for one pitch and return its metric record."""
    events, contribution = event_records(unit.stored, task, dctx.ctx)
    rep = _representative_event(task)
    stem = f"p{task.pitch:03d}_{note_name(task.pitch)}"
    reference = rep.reference[: seconds_to_frames(rep.duration_s, dctx.sample_rate)]
    write_wav(out_dir / "compare" / f"{stem}_ref.wav", reference, dctx.sample_rate)
    rendered, rate, source = _rendered_note(dctx, kind, unit, task, rep)
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
    """Decode every stored sample back to a float WAV under ``samples/`` (bit-identical to module.it)."""
    for unit in kind.units:
        write_wav(out_dir / "samples" / f"{unit.label}.wav", unit.stored.pcm, unit.stored.sample_rate)


def _write_module_and_render(kind: PlanKind, out_dir: Path, dctx: DumpContext) -> bool:
    """Write ``module.it`` and, when openmpt123 is available and requested, the ground-truth render."""
    module = kind.make_module(dctx.material)
    write_it(out_dir / "module.it", module)
    if not (dctx.settings.render_ground_truth and openmpt123_available()):
        return False
    (out_dir / "render").mkdir(parents=True, exist_ok=True)
    audio, rate = render_module(module, dctx.settings.render)
    write_wav(out_dir / "render" / "module.wav", audio, rate)
    return True


def _write_metrics(kind: PlanKind, out_dir: Path, dctx: DumpContext) -> None:
    """Score and A/B-render every covered pitch; ``metrics.json``'s objective reproduces ``plan.objective``."""
    notes = [_note_record(kind, unit, task, out_dir, dctx) for unit in kind.units for task in unit.tasks]
    document = metrics_document(
        kind.name, kind.plan_document.instrument_id, dctx.sample_rate, kind.plan_document.objective, notes
    )
    write_json(out_dir / "metrics.json", document)


def _dump_plan(kind: PlanKind, out_dir: Path, dctx: DumpContext, started_at: float) -> PlanArtifacts:
    """Write every artifact for one strategy and return a summary of what landed on disk.

    ``started_at`` is the :func:`time.perf_counter` reading taken before the optimize call, so the
    reported ``elapsed_s`` spans the whole strategy (optimize + this dump), not just the I/O here.
    """
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)
    _write_plan_docs(kind, out_dir)
    _write_sample_wavs(kind, out_dir)
    rendered = _write_module_and_render(kind, out_dir, dctx)
    _write_metrics(kind, out_dir, dctx)
    return PlanArtifacts(
        name=kind.name,
        feasible=True,
        reason=None,
        rendered=rendered,
        objective=kind.plan_document.objective,
        used_bytes=kind.plan_document.budget.used_bytes,
        elapsed_s=perf_counter() - started_at,
    )


def _optimize_and_dump(
    instrument: InstrumentSpec, out_dir: Path, dctx: DumpContext, strategy: _Strategy
) -> PlanArtifacts:
    """Optimize one strategy and dump it; on an infeasible budget, record why instead of raising."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    try:
        plan = strategy.optimize(instrument, dctx.audio, dctx.sample_rate, dctx.settings.optimize)
        kind = make_kind(plan, dctx)
    except BudgetInfeasibleError as exc:
        write_text(out_dir / "INFEASIBLE.txt", f"{strategy.name} allocation is infeasible at this budget:\n{exc}\n")
        return PlanArtifacts(
            strategy.name,
            feasible=False,
            reason=str(exc),
            rendered=False,
            objective=None,
            used_bytes=None,
            elapsed_s=perf_counter() - started_at,
        )
    return _dump_plan(kind, out_dir, dctx, started_at)


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
    _, ctx, tasks = prepare_run(instrument, audio, sample_rate, settings.optimize)
    dctx = DumpContext(
        audio=audio,
        sample_rate=sample_rate,
        material=tuple(instrument.material or []),
        ctx=ctx,
        tasks_by_pitch={task.pitch: task for task in tasks},
        settings=settings,
    )
    strategies = [
        strategy for strategy, enabled in ((_UNGROUPED, settings.ungrouped), (_GROUPED, settings.grouped)) if enabled
    ]
    plans = [_optimize_and_dump(instrument, out_dir / strategy.name, dctx, strategy) for strategy in strategies]
    return DumpResult(instrument_id=instrument.id, directory=out_dir, plans=tuple(plans))


def dump_project(manifest: Manifest, out_dir: Path | str, settings: DumpSettings) -> list[DumpResult]:
    """Run :func:`dump_instrument` for every instrument in a loaded manifest under ``out_dir``."""
    out_dir = Path(out_dir)
    results: list[DumpResult] = []
    for instrument in manifest.instruments:
        audio, sample_rate = load_instrument_audio(instrument)
        results.append(dump_instrument(instrument, audio, sample_rate, out_dir / instrument.id, settings))
    return results
