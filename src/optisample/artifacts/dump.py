"""Dump inspectable artifacts for an optimized instrument, for offline quality analysis.

The optimizer's decisions and its objective are otherwise only visible as numbers in a report; this
module writes the *audible and machine-readable* evidence behind them to a directory tree, so a human
can listen to what was stored, compare it against the real recordings, and read the exact per-note
scores that drove the allocation. Nothing here changes the optimizer -- it re-runs it and serializes
the result.

Per instrument, each strategy (``ungrouped`` = one sample per key, ``grouped`` = P5 pitch zones) gets
its own subtree::

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
installed, falling back to the numpy surrogate otherwise; ``metrics.json`` always reports the
surrogate scores, because those are exactly the objective the optimizer minimized (so its total
reproduces ``plan.objective``). If a strategy is infeasible at the budget it writes ``INFEASIBLE.txt``
instead of a plan -- useful precisely because grouping can fit where the ungrouped allocation cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from optisample.artifacts.context import DumpSettings, _DumpContext
from optisample.artifacts.serialize import _write_json, _write_text
from optisample.artifacts.units import _PlanKind, _Unit, make_kind
from optisample.dsp.surrogate import render
from optisample.io.audio import write_wav
from optisample.io.it_writer import write_it
from optisample.io.render import openmpt123_available, render_module
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, Manifest, NoteEvent
from optisample.music import note_name
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping import optimize_instrument_grouped
from optisample.optimize.orchestrate import load_instrument_audio, optimize_instrument, prepare_run
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import AudioMap, EvalContext, Event, PitchTask, score_events


@dataclass(frozen=True)
class PlanArtifacts:
    """What one strategy produced under its subdirectory (or why it could not).

    Its directory is ``DumpResult.directory / name``; only the per-strategy outcome is kept here.
    """

    name: str
    feasible: bool
    reason: str | None
    rendered: bool  # whether an openmpt123 ground-truth render was written
    objective: float | None
    used_bytes: int | None
    elapsed_s: float  # wall-clock for this strategy end to end (optimize + artifact dump)


@dataclass(frozen=True)
class DumpResult:
    """The full dump for one instrument: where it went and how each strategy fared."""

    instrument_id: str
    directory: Path
    plans: tuple[PlanArtifacts, ...]


def _representative_event(task: PitchTask) -> Event:
    """The note worth listening to for a pitch: the most-played dynamic (ties -> the longest)."""
    return max(task.events, key=lambda event: (event.weight, event.duration_s))


def _note_metrics(unit: _Unit, task: PitchTask, ctx: EvalContext) -> tuple[list[dict[str, Any]], float]:
    """Score every event of one pitch from ``unit``; return per-event JSON and the weighted sum.

    It consumes :func:`optisample.optimize.tasks.score_events` -- the exact stream the optimizer sums
    into its objective -- so the returned contribution is precisely this pitch's share of
    ``plan.objective`` (``weight * mean distortion``).
    """
    events_json: list[dict[str, Any]] = []
    contribution = 0.0
    for score in score_events(unit.stored, task, ctx):
        contribution += score.weighted_fidelity
        events_json.append(
            {
                "velocity": score.event.velocity,
                "duration_s": score.event.duration_s,
                "weight": score.event.weight,
                "volume": score.volume,
                "fidelity": score.report.fidelity,
                "breakdown": score.report.breakdown,
                "diagnostics": score.report.diagnostics,
            }
        )
    return events_json, contribution


def _rendered_note(
    dctx: _DumpContext, kind: _PlanKind, unit: _Unit, task: PitchTask, event: Event
) -> tuple[Signal, int, str]:
    """Audio the module produces for one note: the real engine when available, else the surrogate."""
    if dctx.settings.render_ground_truth and openmpt123_available():
        note = NoteEvent(pitch=task.pitch, velocity=event.velocity, duration_s=event.duration_s)
        audio, rate = render_module(kind.make_module([note]), dctx.settings.render)
        return audio, rate, "openmpt123"
    volume = dctx.ctx.velocity_map.volume(event.velocity)
    candidate = render(unit.stored, dctx.sample_rate, pitch=task.pitch, volume=volume, duration_s=event.duration_s)
    return candidate, dctx.sample_rate, "surrogate"


def _dump_one_note(kind: _PlanKind, unit: _Unit, task: PitchTask, out_dir: Path, dctx: _DumpContext) -> dict[str, Any]:
    """Write the reference/rendered A/B pair for one pitch and return its metric record."""
    events_json, contribution = _note_metrics(unit, task, dctx.ctx)
    rep = _representative_event(task)
    stem = f"p{task.pitch:03d}_{note_name(task.pitch)}"
    reference = rep.reference[: max(0, int(round(rep.duration_s * dctx.sample_rate)))]
    write_wav(out_dir / "compare" / f"{stem}_ref.wav", reference, dctx.sample_rate)
    rendered, rate, source = _rendered_note(dctx, kind, unit, task, rep)
    write_wav(out_dir / "compare" / f"{stem}_render.wav", rendered, rate)
    return {
        "pitch": task.pitch,
        "note": note_name(task.pitch),
        "served_by": unit.label,
        "representative": unit.representative,
        "weight": task.weight,
        "mean_distortion": contribution / task.weight if task.weight > 0.0 else 0.0,
        "objective_contribution": contribution,
        "render_source": source,
        "render_rate": rate,
        "representative_event": {"velocity": rep.velocity, "duration_s": rep.duration_s},
        "events": events_json,
    }


def _dump_notes(kind: _PlanKind, out_dir: Path, dctx: _DumpContext) -> dict[str, Any]:
    """Score and A/B-render every covered pitch; ``objective`` here reproduces ``plan.objective``."""
    notes = [_dump_one_note(kind, unit, task, out_dir, dctx) for unit in kind.units for task in unit.tasks]
    return {
        "strategy": kind.name,
        "instrument_id": kind.plan_json["instrument_id"],
        "sample_rate": dctx.sample_rate,
        "objective": sum(note["objective_contribution"] for note in notes),
        "plan_objective": kind.plan_json["objective"],
        "notes": notes,
    }


def _dump_plan(kind: _PlanKind, out_dir: Path, dctx: _DumpContext, started_at: float) -> PlanArtifacts:
    """Write every artifact for one strategy and return a summary of what landed on disk.

    ``started_at`` is the :func:`time.perf_counter` reading taken before the optimize call, so the
    reported ``elapsed_s`` spans the whole strategy (optimize + this dump), not just the I/O here.
    """
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)
    _write_text(out_dir / "report.txt", kind.report_text)
    _write_json(out_dir / "plan.json", kind.plan_json)
    _write_json(out_dir / "velocity_map.json", kind.plan_json["velocity_map"])
    for unit in kind.units:
        write_wav(out_dir / "samples" / f"{unit.label}.wav", unit.stored.pcm, unit.stored.sample_rate)

    module = kind.make_module(dctx.material)
    write_it(out_dir / "module.it", module)
    rendered = False
    if dctx.settings.render_ground_truth and openmpt123_available():
        (out_dir / "render").mkdir(parents=True, exist_ok=True)
        audio, rate = render_module(module, dctx.settings.render)
        write_wav(out_dir / "render" / "module.wav", audio, rate)
        rendered = True

    _write_json(out_dir / "metrics.json", _dump_notes(kind, out_dir, dctx))
    return PlanArtifacts(
        name=kind.name,
        feasible=True,
        reason=None,
        rendered=rendered,
        objective=kind.plan_json["objective"],
        used_bytes=kind.plan_json["budget"]["used_bytes"],
        elapsed_s=perf_counter() - started_at,
    )


def _optimize_and_dump(instrument: InstrumentSpec, out_dir: Path, dctx: _DumpContext, grouped: bool) -> PlanArtifacts:
    """Optimize one strategy and dump it; on an infeasible budget, record why instead of raising."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "grouped" if grouped else "ungrouped"
    started_at = perf_counter()
    try:
        plan: InstrumentPlan | GroupedInstrumentPlan
        if grouped:
            plan = optimize_instrument_grouped(instrument, dctx.audio, dctx.sample_rate, dctx.settings.optimize)
        else:
            plan = optimize_instrument(instrument, dctx.audio, dctx.sample_rate, dctx.settings.optimize)
        kind = make_kind(plan, dctx)
    except BudgetInfeasibleError as exc:
        _write_text(out_dir / "INFEASIBLE.txt", f"{name} allocation is infeasible at this budget:\n{exc}\n")
        return PlanArtifacts(
            name,
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
    dctx = _DumpContext(
        audio=audio,
        sample_rate=sample_rate,
        material=tuple(instrument.material or []),
        ctx=ctx,
        tasks_by_pitch={task.pitch: task for task in tasks},
        settings=settings,
    )
    plans: list[PlanArtifacts] = []
    if settings.ungrouped:
        plans.append(_optimize_and_dump(instrument, out_dir / "ungrouped", dctx, grouped=False))
    if settings.grouped:
        plans.append(_optimize_and_dump(instrument, out_dir / "grouped", dctx, grouped=True))
    return DumpResult(instrument_id=instrument.id, directory=out_dir, plans=tuple(plans))


def dump_project(manifest: Manifest, out_dir: Path | str, settings: DumpSettings) -> list[DumpResult]:
    """Run :func:`dump_instrument` for every instrument in a loaded manifest under ``out_dir``."""
    out_dir = Path(out_dir)
    results: list[DumpResult] = []
    for instrument in manifest.instruments:
        audio, sample_rate = load_instrument_audio(instrument)
        results.append(dump_instrument(instrument, audio, sample_rate, out_dir / instrument.id, settings))
    return results
