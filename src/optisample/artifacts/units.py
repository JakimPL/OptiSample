from collections.abc import Callable, Sequence
from dataclasses import dataclass

from optisample.artifacts.context import DumpContext, DumpSettings
from optisample.artifacts.serialize import PlanDocument, plan_document
from optisample.dsp.surrogate import StoredSample
from optisample.model import NoteEvent
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.samples import encode_plan_units
from optisample.optimize.plans import (
    GroupedInstrumentPlan,
    InstrumentPlan,
    StrategyPlan,
)
from optisample.optimize.report import format_grouping_report, format_report
from optisample.optimize.tasks import PitchTask
from trackmod.module.protocol import TrackerModule


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
    """A strategy reduced to the pieces the dumper serializes, so it is plan-type agnostic.

    ``module`` is the plan playing the instrument's whole material -- the one the dumper writes and
    renders -- while ``make_module`` rebuilds it over any other material, which is how each pitch gets
    its own single-note A/B render.
    """

    name: str
    units: tuple[Unit, ...]
    report_text: str
    plan_document: PlanDocument
    module: TrackerModule
    make_module: Callable[[Sequence[NoteEvent]], TrackerModule]


def build_units(plan: StrategyPlan, dump_context: DumpContext) -> tuple[Unit, ...]:
    """Re-encode every stored sample the plan kept, in the exporter's order + seed so the PCM matches.

    ``plan.sample_units`` reports the strategy-specific choices -- each unit's representative recording,
    root pitch and the keys it covers -- and encoding them through one seeded RNG is what keeps the byte
    layout reproducing ``module.it`` exactly. A unit's representative pitch is always its encode root and
    its recording is the loudest velocity actually played there.
    """
    units: list[Unit] = []
    encoded = encode_plan_units(
        plan.sample_units(),
        dump_context.audio,
        dump_context.sample_rate,
        dump_context.settings.optimize.encode,
        dump_context.settings.optimize.seed,
    )
    for unit, stored in encoded:
        units.append(
            Unit(
                label=unit.label,
                stored=stored,
                tasks=tuple(dump_context.tasks_by_pitch[key] for key in unit.keys),
                representative=unit.representative,
                representative_velocity=unit.representative_velocity,
            )
        )
    return tuple(units)


def _export_context(settings: DumpSettings) -> ExportContext:
    """The exporter context (re-encode config, playback, target format, dither seed) from the settings."""
    return ExportContext(
        encode=settings.optimize.encode,
        playback=settings.playback,
        target=settings.optimize.target,
        seed=settings.optimize.seed,
    )


def make_kind(plan: InstrumentPlan | GroupedInstrumentPlan, dump_context: DumpContext) -> PlanKind:
    """Package a plan (ungrouped or grouped) as the strategy-agnostic pieces the dumper serializes.

    ``build_units`` yields units in plan order, so the stored loops line up with the plan's items when
    :func:`plan_document` zips them together. The module is built here so the report and the plan
    document can state what the written file actually occupies. Only the report formatter is
    strategy-specific.
    """
    units = build_units(plan, dump_context)
    export_context = _export_context(dump_context.settings)
    loops = [unit.stored.loop for unit in units]

    def make_module(material: Sequence[NoteEvent]) -> TrackerModule:
        return build_module(plan, dump_context.audio, dump_context.sample_rate, list(material), export_context)

    module = make_module(dump_context.material)
    size = module.size()
    if plan.strategy == "grouped":
        report_text = format_grouping_report(plan, size)
    else:
        report_text = format_report(plan, size)
    return PlanKind(plan.strategy, units, report_text, plan_document(plan, loops, size), module, make_module)
