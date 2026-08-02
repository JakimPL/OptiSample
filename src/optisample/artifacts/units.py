from collections.abc import Callable, Sequence
from dataclasses import dataclass

from optisample.artifacts.context import DumpContext
from optisample.artifacts.documents.plan import PlanDocument, plan_document
from optisample.dsp.surrogate import StoredSample
from optisample.keys import SampleKey
from optisample.model import NoteEvent
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.coverage import key_coverage, played_keys
from optisample.optimize.export.samples import encode_plan_units
from optisample.optimize.layers.bands import VelocityLayers
from optisample.optimize.layers.slots import SlotLayout, plan_slots
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
    """One stored sample and the pitch tasks it serves (a zone, or a single key when ungrouped).

    ``layer`` is the velocity band the sample answers for, so the tasks are the notes that band covers
    and the artifacts written for them are filed under the same layer the module plays them through.
    """

    label: str
    stored: StoredSample
    layer: int
    tasks: tuple[PitchTask, ...]
    representative_key: SampleKey

    @property
    def representative(self) -> int:
        """The pitch the stored sample is rooted at, which is its recording's own pitch."""
        return self.representative_key.pitch


@dataclass(frozen=True)
class PlanKind:
    """A strategy reduced to the pieces the dumper serializes, so it is plan-type agnostic.

    ``module`` is the plan playing the instrument's whole material -- the one the dumper writes and
    renders -- while ``make_module`` rebuilds it over any other material, which is how each pitch gets
    its own single-note A/B render. ``layout`` names the instruments the module numbers, so the dumper
    writes each one on its own and files a unit's artifacts under the band that plays them.
    """

    name: str
    layout: SlotLayout
    units: tuple[Unit, ...]
    report_text: str
    plan_document: PlanDocument
    module: TrackerModule
    make_module: Callable[[Sequence[NoteEvent]], TrackerModule]

    @property
    def layers(self) -> VelocityLayers:
        """The velocity split the plan stores, which is the bands its written instruments answer for."""
        return self.layout.layers


def build_units(plan: StrategyPlan, dump_context: DumpContext) -> tuple[Unit, ...]:
    """Re-encode every stored sample the plan kept, in the exporter's order + seed so the PCM matches.

    ``plan.sample_units`` reports the strategy-specific choices -- each unit's representative recording,
    root pitch, velocity layer and the keys it covers -- and encoding them through one seeded RNG is what
    keeps the byte layout reproducing the written module exactly. A unit's representative pitch is always
    its encode root and its recording is the loudest velocity played in the band it answers for. The
    encode config comes from the prepared run, so a unit is re-encoded under the gain staging the
    allocation scored it with.
    """
    units: list[Unit] = []
    tasks = dump_context.layer_tasks(plan.layers)
    encoded = encode_plan_units(
        plan.sample_units(),
        dump_context.recordings,
        dump_context.eval_context.encode,
        dump_context.settings.optimize.seed,
    )
    for unit, stored in encoded:
        units.append(
            Unit(
                label=unit.label,
                stored=stored,
                layer=unit.layer,
                tasks=tuple(tasks[(unit.layer, key)] for key in unit.keys),
                representative_key=unit.representative_key,
            )
        )
    return tuple(units)


def _export_context(dump_context: DumpContext) -> ExportContext:
    """The exporter context: the run's own re-encode config, plus playback, target format and dither seed."""
    settings = dump_context.settings
    return ExportContext(
        encode=dump_context.eval_context.encode,
        playback=settings.playback,
        target=settings.optimize.target,
        envelope=settings.envelope,
        seed=settings.optimize.seed,
    )


def make_kind(plan: InstrumentPlan | GroupedInstrumentPlan, dump_context: DumpContext) -> PlanKind:
    """Package a plan (ungrouped or grouped) as the strategy-agnostic pieces the dumper serializes.

    ``build_units`` yields units in plan order, so the re-encoded samples line up with the plan's items
    when :func:`plan_document` zips them together. The module is built here so the report and the plan
    document can state what the written file actually occupies. Only the report formatter is
    strategy-specific.
    """
    units = build_units(plan, dump_context)
    export_context = _export_context(dump_context)
    encoded = [unit.stored for unit in units]

    def make_module(material: Sequence[NoteEvent]) -> TrackerModule:
        return build_module(plan, dump_context.recordings, list(material), export_context)

    module = make_module(dump_context.material)
    size = module.size()
    layout = plan_slots(plan, export_context.target)
    coverage = key_coverage(
        [instrument.keymap for instrument in module.song.instruments],
        export_context.target,
        played=played_keys(plan.sample_units()),
    )
    if plan.strategy == "grouped":
        report_text = format_grouping_report(plan, size, coverage, layout)
    else:
        report_text = format_report(plan, size, coverage, layout)
    return PlanKind(
        plan.strategy,
        layout,
        units,
        report_text,
        plan_document(plan, encoded, size, coverage, layout),
        module,
        make_module,
    )
