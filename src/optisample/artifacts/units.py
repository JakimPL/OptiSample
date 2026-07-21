"""Re-encode a plan's stored samples and package each strategy for the dumper.

A :class:`Unit` is one stored sample plus the pitch tasks it serves; the same seeded re-encode the
exporter runs is replayed here so the decoded WAVs match ``module.it`` byte for byte. The plan reports
its stored samples as :class:`~optisample.optimize.plans.SampleUnit`s (which recording is each unit's
representative and which keys it covers), so this module encodes either strategy through one shared RNG
loop without knowing which produced the plan. :func:`make_kind` wraps a plan as a :class:`PlanKind` (its
units, report text, plan document and a module builder) so the dumper serializes either strategy through
one interface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.artifacts.context import DumpContext, DumpSettings
from optisample.artifacts.serialize import PlanDocument, plan_document
from optisample.dsp.surrogate import EncodeContext, StoredSample, encode
from optisample.io.it_writer import ITModule
from optisample.model import NoteEvent
from optisample.optimize.export import ExportContext, build_module
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan, StrategyPlan
from optisample.optimize.report import format_grouping_report, format_report
from optisample.optimize.tasks import PitchTask


@dataclass(frozen=True)
class Unit:
    """One stored sample and the pitch tasks it serves (a zone, or a single key when ungrouped)."""

    label: str
    stored: StoredSample
    tasks: tuple[PitchTask, ...]
    representative: int
    representative_velocity: int


@dataclass(frozen=True)
class PlanKind:
    """A strategy reduced to the pieces the dumper serializes, so it is plan-type agnostic."""

    name: str
    units: tuple[Unit, ...]
    report_text: str
    plan_document: PlanDocument
    make_module: Callable[[Sequence[NoteEvent]], ITModule]


def build_units(plan: StrategyPlan, dctx: DumpContext) -> tuple[Unit, ...]:
    """Re-encode every stored sample the plan kept, in the exporter's order + seed so the PCM matches.

    ``plan.sample_units`` reports the strategy-specific choices -- each unit's representative recording,
    root pitch and the keys it covers -- and encoding them through one seeded RNG is what keeps the byte
    layout reproducing ``module.it`` exactly. A unit's representative pitch is always its encode root and
    its recording is the loudest velocity actually played there.
    """
    rng = np.random.default_rng(dctx.settings.optimize.seed)
    units: list[Unit] = []
    for unit in plan.sample_units():
        signal = dctx.audio[(unit.representative, unit.representative_velocity)]
        encode_ctx = EncodeContext(root_pitch=unit.representative, config=dctx.settings.optimize.encode, rng=rng)
        stored = encode(signal, dctx.sample_rate, unit.params, encode_ctx)
        units.append(
            Unit(
                label=unit.label,
                stored=stored,
                tasks=tuple(dctx.tasks_by_pitch[key] for key in unit.keys),
                representative=unit.representative,
                representative_velocity=unit.representative_velocity,
            )
        )
    return tuple(units)


def _export_ctx(settings: DumpSettings) -> ExportContext:
    """The IT-exporter context (re-encode config + playback + dither seed) built from the dump settings."""
    return ExportContext(encode=settings.optimize.encode, playback=settings.playback, seed=settings.optimize.seed)


def make_kind(plan: InstrumentPlan | GroupedInstrumentPlan, dctx: DumpContext) -> PlanKind:
    """Package a plan (ungrouped or grouped) as the strategy-agnostic pieces the dumper serializes.

    ``build_units`` yields units in plan order, so the stored loops line up with the plan's items when
    :func:`plan_document` zips them together. Only the report formatter is strategy-specific.
    """
    units = build_units(plan, dctx)
    export_ctx = _export_ctx(dctx.settings)
    loops = [unit.stored.loop for unit in units]

    def make_module(material: Sequence[NoteEvent]) -> ITModule:
        return build_module(plan, dctx.audio, dctx.sample_rate, list(material), export_ctx)

    if plan.strategy == "grouped":
        report_text = format_grouping_report(plan)
    else:
        report_text = format_report(plan)
    return PlanKind(plan.strategy, units, report_text, plan_document(plan, loops), make_module)
