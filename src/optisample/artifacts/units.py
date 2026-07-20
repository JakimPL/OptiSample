"""Re-encode a plan's stored samples and package each strategy for the dumper.

A :class:`_Unit` is one stored sample plus the pitch tasks it serves; the same seeded re-encode the
exporter runs is replayed here so the decoded WAVs match ``module.it`` byte for byte. The two strategies
differ only in which recording is a unit's representative and which keys it covers -- captured as
``_UnitSpec``s first, then encoded through one shared RNG loop. :func:`make_kind` wraps a plan as a
:class:`_PlanKind` (its units, report text, plan JSON and a module builder) so the dumper serializes
either strategy through one interface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from optisample.artifacts.context import DumpSettings, _DumpContext
from optisample.artifacts.serialize import plan_json
from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.io.it_writer import ITModule
from optisample.metrics.base import Signal
from optisample.model import NoteEvent
from optisample.music import note_name
from optisample.optimize.export import ExportContext, build_module
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.report import format_grouping_report, format_report
from optisample.optimize.tasks import PitchTask


@dataclass(frozen=True)
class _Unit:
    """One stored sample and the pitch tasks it serves (a zone, or a single key when ungrouped)."""

    label: str
    stored: StoredSample
    tasks: tuple[PitchTask, ...]
    representative: int
    representative_velocity: int


@dataclass(frozen=True)
class _PlanKind:
    """A strategy reduced to the pieces the dumper serializes, so it is plan-type agnostic."""

    name: str
    units: tuple[_Unit, ...]
    report_text: str
    plan_json: dict[str, Any]
    make_module: Callable[[Sequence[NoteEvent]], ITModule]


@dataclass(frozen=True)
class _UnitSpec:
    """What to re-encode for one unit, before the shared loop turns it into a :class:`_Unit`."""

    label: str
    signal: Signal
    params: EncodingParams
    tasks: tuple[PitchTask, ...]
    representative: int
    representative_velocity: int


def _ungrouped_specs(plan: InstrumentPlan, dctx: _DumpContext) -> list[_UnitSpec]:
    """One spec per kept pitch, taking the recording from the pitch's own task."""
    specs: list[_UnitSpec] = []
    for pitch in plan.pitches:
        task = dctx.tasks_by_pitch[pitch.pitch]
        specs.append(
            _UnitSpec(
                label=f"p{pitch.pitch:03d}_{note_name(pitch.pitch)}",
                signal=task.representative,
                params=pitch.chosen.params,
                tasks=(task,),
                representative=pitch.pitch,
                representative_velocity=pitch.representative_velocity,
            )
        )
    return specs


def _grouped_specs(plan: GroupedInstrumentPlan, dctx: _DumpContext) -> list[_UnitSpec]:
    """One spec per zone, taking the recording from the zone representative and serving all its keys."""
    specs: list[_UnitSpec] = []
    for index, zone in enumerate(plan.zones):
        specs.append(
            _UnitSpec(
                label=f"zone{index:02d}_rep{zone.representative:03d}_{note_name(zone.representative)}",
                signal=dctx.audio[(zone.representative, zone.representative_velocity)],
                params=zone.chosen.params,
                tasks=tuple(dctx.tasks_by_pitch[pitch] for pitch in zone.pitches),
                representative=zone.representative,
                representative_velocity=zone.representative_velocity,
            )
        )
    return specs


def build_units(plan: InstrumentPlan | GroupedInstrumentPlan, dctx: _DumpContext) -> tuple[_Unit, ...]:
    """Re-encode every stored sample the plan kept, in the exporter's order + seed so the PCM matches.

    The strategy-specific part is choosing each unit's recording and root pitch (the ``_UnitSpec``s);
    encoding them through one seeded RNG is shared, which is what keeps the byte layout reproducing
    ``module.it`` exactly. A unit's representative pitch is always its encode root.
    """
    specs = _grouped_specs(plan, dctx) if isinstance(plan, GroupedInstrumentPlan) else _ungrouped_specs(plan, dctx)
    rng = np.random.default_rng(dctx.settings.optimize.seed)
    units: list[_Unit] = []
    for spec in specs:
        encode_ctx = EncodeContext(root_pitch=spec.representative, config=dctx.settings.optimize.encode, rng=rng)
        stored = encode(spec.signal, dctx.sample_rate, spec.params, encode_ctx)
        units.append(
            _Unit(
                label=spec.label,
                stored=stored,
                tasks=spec.tasks,
                representative=spec.representative,
                representative_velocity=spec.representative_velocity,
            )
        )
    return tuple(units)


def _export_ctx(settings: DumpSettings) -> ExportContext:
    """The IT-exporter context (re-encode config + playback + dither seed) built from the dump settings."""
    return ExportContext(encode=settings.optimize.encode, playback=settings.playback, seed=settings.optimize.seed)


def make_kind(plan: InstrumentPlan | GroupedInstrumentPlan, dctx: _DumpContext) -> _PlanKind:
    """Package a plan (ungrouped or grouped) as the strategy-agnostic pieces the dumper serializes.

    ``build_units`` yields units in plan order, so the stored loops line up with the plan's items when
    :func:`plan_json` zips them together.
    """
    units = build_units(plan, dctx)
    export_ctx = _export_ctx(dctx.settings)
    loops = [unit.stored.loop for unit in units]

    def make_module(material: Sequence[NoteEvent]) -> ITModule:
        return build_module(plan, dctx.audio, dctx.sample_rate, list(material), export_ctx)

    if isinstance(plan, GroupedInstrumentPlan):
        return _PlanKind("grouped", units, format_grouping_report(plan), plan_json(plan, loops), make_module)
    return _PlanKind("ungrouped", units, format_report(plan), plan_json(plan, loops), make_module)
