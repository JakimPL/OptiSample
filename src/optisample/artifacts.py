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

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.dsp.loop import Loop
from optisample.dsp.surrogate import EncodeContext, StoredSample, default_encode_config, encode, render
from optisample.io.audio import write_wav
from optisample.io.it_writer import ITModule, write_it
from optisample.io.render import default_playback_config, default_render_config, openmpt123_available, render_module
from optisample.metrics.base import Signal
from optisample.metrics.composite import evaluate
from optisample.model import InstrumentSpec, Manifest, NoteEvent
from optisample.optimize.export import ExportContext, build_grouped_it_module, build_it_module
from optisample.optimize.grouping import GroupedInstrumentPlan, format_grouping_report, optimize_instrument_grouped
from optisample.optimize.knapsack import BudgetInfeasibleError
from optisample.optimize.orchestrate import (
    InstrumentPlan,
    OptimizeSettings,
    default_optimize_settings,
    format_report,
    load_instrument_audio,
    optimize_instrument,
    prepare_run,
)
from optisample.optimize.tasks import AudioMap, EvalContext, Event, PitchTask
from optisample.optimize.velocity_map import VelocityVolumeMap

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note_name(pitch: int) -> str:
    """MIDI note number -> scientific pitch name (60 -> ``C4``)."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class DumpSettings:
    """What to dump and how (bundled to keep call sites small)."""

    # transitional: OptimizeSettings is built from a loaded config at the CLI in phase 9.
    optimize: OptimizeSettings = field(default_factory=default_optimize_settings)
    # transitional: RenderConfig is threaded from the CLI in phase 9.
    render: RenderConfig = field(default_factory=default_render_config)
    # transitional: PlaybackConfig is threaded from the CLI in phase 9.
    playback: PlaybackConfig = field(default_factory=default_playback_config)
    render_ground_truth: bool = True  # render module + per-note through openmpt123 if it is installed
    grouped: bool = True
    ungrouped: bool = True


@dataclass(frozen=True)
class PlanArtifacts:
    """What one strategy produced under its subdirectory (or why it could not)."""

    name: str
    directory: Path
    feasible: bool
    reason: str | None
    rendered: bool  # whether an openmpt123 ground-truth render was written
    objective: float | None
    used_bytes: int | None


@dataclass(frozen=True)
class DumpResult:
    """The full dump for one instrument: where it went and how each strategy fared."""

    instrument_id: str
    directory: Path
    plans: tuple[PlanArtifacts, ...]


@dataclass(frozen=True)
class _Unit:
    """One stored sample and the pitch tasks it serves (a zone, or a single key when ungrouped)."""

    label: str
    stored: StoredSample
    tasks: tuple[PitchTask, ...]
    representative: int
    representative_velocity: int


@dataclass(frozen=True)
class _DumpContext:
    """Inputs shared across both strategies for one instrument."""

    audio: AudioMap
    sample_rate: int
    material: tuple[NoteEvent, ...]
    ctx: EvalContext
    tasks_by_pitch: dict[int, PitchTask]
    settings: DumpSettings


@dataclass(frozen=True)
class _PlanKind:
    """A strategy reduced to the pieces the dumper serializes, so it is plan-type agnostic."""

    name: str
    units: tuple[_Unit, ...]
    report_text: str
    plan_json: dict[str, Any]
    make_module: Callable[[Sequence[NoteEvent]], ITModule]


# --- JSON serialization --------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars to Python and non-finite floats (silence -> -inf) to null."""
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _velocity_map_json(velocity_map: VelocityVolumeMap) -> dict[str, Any]:
    reference_volume = max((anchor.volume for anchor in velocity_map.anchors), default=0)
    return {
        "reference_volume": reference_volume,
        "anchors": [
            {"velocity": anchor.velocity, "loudness_lufs": anchor.loudness_lufs, "volume": anchor.volume}
            for anchor in velocity_map.anchors
        ],
        "volumes": list(velocity_map.volumes),
    }


def _loop_json(loop: Loop | None) -> dict[str, int] | None:
    """The loop actually stored (``{start, end}``), or ``None`` when the sample was not looped."""
    return None if loop is None else {"start": loop.start, "end": loop.end}


def _budget_json(plan: InstrumentPlan | GroupedInstrumentPlan) -> dict[str, int]:
    return {
        "module_budget_bytes": plan.module_budget_bytes,
        "sample_budget_bytes": plan.sample_budget_bytes,
        "used_bytes": plan.used_bytes,
        "module_bytes": plan.module_bytes,
    }


def _plan_json_ungrouped(plan: InstrumentPlan, units: tuple[_Unit, ...]) -> dict[str, Any]:
    return {
        "strategy": "ungrouped",
        "instrument_id": plan.instrument_id,
        "method": plan.method,
        "objective": plan.objective,
        "budget": _budget_json(plan),
        "velocity_map": _velocity_map_json(plan.velocity_map),
        "pitches": [
            {
                "pitch": pitch.pitch,
                "note": _note_name(pitch.pitch),
                "weight": pitch.weight,
                "representative_velocity": pitch.representative_velocity,
                "target_rate": pitch.chosen.params.target_rate,
                "depth_bits": pitch.chosen.params.depth_bits,
                "trim_s": pitch.chosen.params.trim_s,
                "loop": _loop_json(unit.stored.loop),  # the loop actually stored, not merely requested
                "frames": pitch.chosen.frames,
                "stored_bytes": pitch.chosen.stored_bytes,
                "distortion": pitch.chosen.distortion,
                "hull_size": len(pitch.hull),
            }
            for pitch, unit in zip(plan.pitches, units)
        ],
    }


def _plan_json_grouped(plan: GroupedInstrumentPlan, units: tuple[_Unit, ...]) -> dict[str, Any]:
    return {
        "strategy": "grouped",
        "instrument_id": plan.instrument_id,
        "objective": plan.objective,
        "budget": _budget_json(plan),
        "velocity_map": _velocity_map_json(plan.velocity_map),
        "zones": [
            {
                "keys": [zone.pitches[0], zone.pitches[-1]],
                "pitches": list(zone.pitches),
                "representative": zone.representative,
                "representative_velocity": zone.representative_velocity,
                "weight": zone.weight,
                "target_rate": zone.chosen.params.target_rate,
                "depth_bits": zone.chosen.params.depth_bits,
                "trim_s": zone.chosen.params.trim_s,
                "loop": _loop_json(unit.stored.loop),  # the loop actually stored, not merely requested
                "frames": zone.chosen.frames,
                "stored_bytes": zone.chosen.stored_bytes,
                "distortion": zone.chosen.distortion,
                "hull_size": len(zone.hull),
            }
            for zone, unit in zip(plan.zones, units)
        ],
    }


# --- stored-sample units (mirror the exporter's re-encode exactly) -------------------------------


def _ungrouped_units(plan: InstrumentPlan, dctx: _DumpContext) -> tuple[_Unit, ...]:
    """One unit per kept pitch, re-encoding in the exporter's order/seed so PCM matches ``module.it``."""
    rng = np.random.default_rng(dctx.settings.optimize.seed)
    units: list[_Unit] = []
    for pitch in plan.pitches:
        task = dctx.tasks_by_pitch[pitch.pitch]
        # transitional: EncodeConfig is threaded through DumpSettings in phase 9.
        encode_ctx = EncodeContext(root_pitch=pitch.pitch, config=default_encode_config(), rng=rng)
        stored = encode(task.representative, dctx.sample_rate, pitch.chosen.params, encode_ctx)
        units.append(
            _Unit(
                label=f"p{pitch.pitch:03d}_{_note_name(pitch.pitch)}",
                stored=stored,
                tasks=(task,),
                representative=pitch.pitch,
                representative_velocity=pitch.representative_velocity,
            )
        )
    return tuple(units)


def _grouped_units(plan: GroupedInstrumentPlan, dctx: _DumpContext) -> tuple[_Unit, ...]:
    """One unit per zone, re-encoding the representative in the exporter's order/seed."""
    rng = np.random.default_rng(dctx.settings.optimize.seed)
    units: list[_Unit] = []
    for index, zone in enumerate(plan.zones):
        signal: Signal = dctx.audio[(zone.representative, zone.representative_velocity)]
        # transitional: EncodeConfig is threaded through DumpSettings in phase 9.
        encode_ctx = EncodeContext(root_pitch=zone.representative, config=default_encode_config(), rng=rng)
        stored = encode(signal, dctx.sample_rate, zone.chosen.params, encode_ctx)
        units.append(
            _Unit(
                label=f"zone{index:02d}_rep{zone.representative:03d}_{_note_name(zone.representative)}",
                stored=stored,
                tasks=tuple(dctx.tasks_by_pitch[pitch] for pitch in zone.pitches),
                representative=zone.representative,
                representative_velocity=zone.representative_velocity,
            )
        )
    return tuple(units)


# --- per-note metrics + A/B audio ----------------------------------------------------------------


def _representative_event(task: PitchTask) -> Event:
    """The note worth listening to for a pitch: the most-played dynamic (ties -> the longest)."""
    return max(task.events, key=lambda event: (event.weight, event.duration_s))


def _note_metrics(unit: _Unit, task: PitchTask, ctx: EvalContext) -> tuple[list[dict[str, Any]], float]:
    """Score every event of one pitch from ``unit``; return per-event JSON and the weighted sum.

    This mirrors :func:`optisample.optimize.tasks.score_reconstruction` exactly, so the returned sum
    is the pitch's contribution to the objective (``weight * mean distortion``).
    """
    events_json: list[dict[str, Any]] = []
    contribution = 0.0
    for event in task.events:
        volume = ctx.velocity_map.volume(event.velocity)
        candidate = render(unit.stored, ctx.sample_rate, pitch=task.pitch, volume=volume, duration_s=event.duration_s)
        reference = event.reference[: max(0, int(round(event.duration_s * ctx.sample_rate)))]
        report = evaluate(reference, candidate, ctx.sample_rate, ctx.composite)
        contribution += event.weight * report.fidelity
        events_json.append(
            {
                "velocity": event.velocity,
                "duration_s": event.duration_s,
                "weight": event.weight,
                "volume": volume,
                "fidelity": report.fidelity,
                "breakdown": report.breakdown,
                "diagnostics": report.diagnostics,
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
    stem = f"p{task.pitch:03d}_{_note_name(task.pitch)}"
    reference = rep.reference[: max(0, int(round(rep.duration_s * dctx.sample_rate)))]
    write_wav(out_dir / "compare" / f"{stem}_ref.wav", reference, dctx.sample_rate)
    rendered, rate, source = _rendered_note(dctx, kind, unit, task, rep)
    write_wav(out_dir / "compare" / f"{stem}_render.wav", rendered, rate)
    return {
        "pitch": task.pitch,
        "note": _note_name(task.pitch),
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


# --- orchestration -------------------------------------------------------------------------------


def _dump_plan(kind: _PlanKind, out_dir: Path, dctx: _DumpContext) -> PlanArtifacts:
    """Write every artifact for one strategy and return a summary of what landed on disk."""
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
        directory=out_dir,
        feasible=True,
        reason=None,
        rendered=rendered,
        objective=kind.plan_json["objective"],
        used_bytes=kind.plan_json["budget"]["used_bytes"],
    )


def _export_ctx(settings: DumpSettings) -> ExportContext:
    """The IT-exporter context (re-encode config + playback + dither seed) built from the dump settings."""
    return ExportContext(encode=settings.optimize.encode, playback=settings.playback, seed=settings.optimize.seed)


def _ungrouped_kind(plan: InstrumentPlan, dctx: _DumpContext) -> _PlanKind:
    units = _ungrouped_units(plan, dctx)

    def make_module(material: Sequence[NoteEvent]) -> ITModule:
        return build_it_module(plan, dctx.audio, dctx.sample_rate, list(material), _export_ctx(dctx.settings))

    return _PlanKind("ungrouped", units, format_report(plan), _plan_json_ungrouped(plan, units), make_module)


def _grouped_kind(plan: GroupedInstrumentPlan, dctx: _DumpContext) -> _PlanKind:
    units = _grouped_units(plan, dctx)

    def make_module(material: Sequence[NoteEvent]) -> ITModule:
        return build_grouped_it_module(plan, dctx.audio, dctx.sample_rate, list(material), _export_ctx(dctx.settings))

    return _PlanKind("grouped", units, format_grouping_report(plan), _plan_json_grouped(plan, units), make_module)


def _optimize_and_dump(instrument: InstrumentSpec, out_dir: Path, dctx: _DumpContext, grouped: bool) -> PlanArtifacts:
    """Optimize one strategy and dump it; on an infeasible budget, record why instead of raising."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "grouped" if grouped else "ungrouped"
    try:
        if grouped:
            grouped_plan = optimize_instrument_grouped(instrument, dctx.audio, dctx.sample_rate, dctx.settings.optimize)
            kind = _grouped_kind(grouped_plan, dctx)
        else:
            plan = optimize_instrument(instrument, dctx.audio, dctx.sample_rate, dctx.settings.optimize)
            kind = _ungrouped_kind(plan, dctx)
    except BudgetInfeasibleError as exc:
        _write_text(out_dir / "INFEASIBLE.txt", f"{name} allocation is infeasible at this budget:\n{exc}\n")
        return PlanArtifacts(
            name, out_dir, feasible=False, reason=str(exc), rendered=False, objective=None, used_bytes=None
        )
    return _dump_plan(kind, out_dir, dctx)


def dump_instrument(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    out_dir: Path | str,
    settings: DumpSettings = DumpSettings(),
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


def dump_project(manifest: Manifest, out_dir: Path | str, settings: DumpSettings = DumpSettings()) -> list[DumpResult]:
    """Run :func:`dump_instrument` for every instrument in a loaded manifest under ``out_dir``."""
    out_dir = Path(out_dir)
    results: list[DumpResult] = []
    for instrument in manifest.instruments:
        audio, sample_rate = load_instrument_audio(instrument)
        results.append(dump_instrument(instrument, audio, sample_rate, out_dir / instrument.id, settings))
    return results
