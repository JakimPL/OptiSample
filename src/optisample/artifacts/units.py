from collections.abc import Callable, Sequence
from dataclasses import dataclass

from optisample.artifacts.context import DumpContext
from optisample.artifacts.documents.plan import PlanDocument, plan_document
from optisample.dsp.surrogate import StoredSample
from optisample.io.tracker.envelope import shape_nodes
from optisample.keys import SampleKey
from optisample.model import NoteEvent
from optisample.optimize.export import build_module
from optisample.optimize.export.build import written_voices
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.coverage import key_coverage, played_keys
from optisample.optimize.export.samples import encode_order, encode_plan_units, sample_gains
from optisample.optimize.export.voices import PlayedVoices, WrittenInstruments, written_instruments
from optisample.optimize.layers.bands import VelocityLayers
from optisample.optimize.layers.slots import SlotLayout, plan_slots
from optisample.optimize.plans import (
    GroupedInstrumentPlan,
    InstrumentPlan,
    StrategyPlan,
)
from optisample.optimize.report import format_grouping_report, format_report
from optisample.optimize.tasks import PitchTask
from trackmod.core.envelopes.envelope import Envelope
from trackmod.module.protocol import TrackerModule


@dataclass(frozen=True)
class Unit:
    """One stored sample and the pitch tasks it serves (a zone, or a single key when ungrouped).

    ``layer`` is the velocity band the sample answers for, so the tasks are the notes that band covers
    and the artifacts written for them are filed under the same layer the module plays them through.

    ``stored`` is the waveform the module carries, which is the one the plan paid for. ``whole`` is the
    same waveform with the rest of its take stored behind the region it wraps on
    (:func:`~optisample.optimize.export.samples.whole_samples`), which is what the standalone file written
    for this voice holds; the two are frame for frame the same up to the loop end.
    """

    label: str
    stored: StoredSample
    whole: StoredSample
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


def _whole_samples(
    plan: StrategyPlan,
    layout: SlotLayout,
    envelopes: Sequence[Envelope | None],
    dump_context: DumpContext,
) -> tuple[StoredSample, ...]:
    """Every unit re-encoded keeping what its recording goes on making past the region it wraps on.

    The same units, curves, seed and clock the module's own samples were encoded from, so a waveform here
    holds the stored one frame for frame up to the loop end and carries the rest of the take behind it. It
    is the standalone file written for one voice that holds it; the module carries what the plan paid for.
    A run asking for no tail is answered with the stored waveforms themselves, the same encode reaching the
    same bytes under the same seed.
    """
    units = plan.sample_units()
    order = encode_order(
        layout,
        _export_context(dump_context),
        envelopes=envelopes,
        units=len(units),
        post_loop=dump_context.settings.post_loop,
    )
    return tuple(stored for _, stored in encode_plan_units(units, dump_context.recordings, order))


def build_units(plan: StrategyPlan, dump_context: DumpContext) -> tuple[Unit, ...]:
    """Re-encode every stored sample the plan kept, in the exporter's order + seed so the PCM matches.

    ``plan.sample_units`` reports the strategy-specific choices -- each unit's representative recording,
    root pitch, velocity layer and the keys it covers -- and encoding them through one seeded RNG is what
    keeps the byte layout reproducing the written module exactly. A unit's representative pitch is always
    its encode root and its recording is the loudest velocity played in the band it answers for. The
    encode config comes from the prepared run, so a unit is re-encoded under the gain staging the
    allocation scored it with.

    Each unit is encoded a second time under the same curves, seed and clock, keeping what its take goes on
    making past the loop, which is the form the standalone file written for that voice carries. The module
    reads ``stored`` alone, so what the plan is priced at stands.
    """
    units: list[Unit] = []
    tasks = dump_context.layer_tasks(plan.layers)
    export_context = _export_context(dump_context)
    layout = plan_slots(plan, export_context.target)
    written = written_voices(
        plan,
        layout,
        dump_context.recordings,
        list(dump_context.material),
        export_context,
    )
    whole = _whole_samples(plan, layout, written.envelopes, dump_context)
    for unit, stored, kept in zip(plan.sample_units(), written.planned.stored, whole):
        units.append(
            Unit(
                label=unit.label,
                stored=stored,
                whole=kept,
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
        carrier=settings.optimize.sweep.carrier,
        seed=settings.optimize.seed,
    )


def _written_instruments(
    plan: InstrumentPlan | GroupedInstrumentPlan,
    layout: SlotLayout,
    encoded: Sequence[StoredSample],
    dump_context: DumpContext,
    export_context: ExportContext,
) -> WrittenInstruments:
    """The instruments the module numbers, beside what the one envelope each carries leaves its keys.

    The shapes are fitted the way the module fits them
    (:func:`~optisample.optimize.export.voices.written_instruments`), so the reading reported here belongs
    to the very curve the written file carries.
    """
    sources = PlayedVoices(
        recordings=dump_context.recordings,
        material=dump_context.material,
        velocity_map=plan.velocity_map,
        gains=sample_gains(list(zip(plan.sample_units(), encoded)), plan.velocity_map, export_context.target),
    )
    return written_instruments(
        layout,
        encoded,
        sources,
        nodes=shape_nodes(export_context.target.envelope_point_bound),
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
        plan_document(
            plan,
            encoded,
            size,
            coverage,
            _written_instruments(plan, layout, encoded, dump_context, export_context),
        ),
        module,
        make_module,
    )
