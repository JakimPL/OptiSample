from __future__ import annotations

from dataclasses import replace

import numpy as np

from optisample.artifacts.context import DumpContext, DumpSettings
from optisample.artifacts.units import build_units, make_kind
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan


def _kept_tail(dump_context: DumpContext) -> DumpSettings:
    """The run's settings asking each written voice to carry the rest of its take."""
    return replace(dump_context.settings, post_loop=True)


def _dropped_tail(dump_context: DumpContext) -> DumpSettings:
    """The run's settings asking each written voice to end where the module's own sample does."""
    return replace(dump_context.settings, post_loop=False)


def test_build_units_re_encodes_one_unit_per_sample_unit(
    ungrouped_plan: InstrumentPlan, dump_context: DumpContext
) -> None:
    units = build_units(ungrouped_plan, dump_context)
    sample_units = ungrouped_plan.sample_units()
    assert len(units) == len(sample_units)
    for unit, sample_unit in zip(units, sample_units):
        assert unit.label == sample_unit.label
        assert unit.representative == sample_unit.representative
        assert unit.stored.root_pitch == sample_unit.representative  # the representative is the encode root
        assert tuple(task.pitch for task in unit.tasks) == tuple(sample_unit.keys)  # covers the unit's keys


def test_build_units_is_deterministic(ungrouped_plan: InstrumentPlan, dump_context: DumpContext) -> None:
    first = build_units(ungrouped_plan, dump_context)
    second = build_units(ungrouped_plan, dump_context)
    for unit_a, unit_b in zip(first, second):
        assert np.array_equal(unit_a.stored.pcm, unit_b.stored.pcm)  # same seed + no dither → identical PCM


def test_a_unit_carries_the_whole_take_beside_the_span_the_module_stores(
    ungrouped_plan: InstrumentPlan, dump_context: DumpContext
) -> None:
    """The plan pays for the region; the file written for that voice holds the rest of the take behind it."""
    for unit in build_units(ungrouped_plan, replace(dump_context, settings=_kept_tail(dump_context))):
        assert unit.whole.loop == unit.stored.loop
        assert unit.whole.frames >= unit.stored.frames
        if unit.stored.loop is not None:
            assert unit.whole.frames > unit.stored.frames


def test_a_run_keeping_no_post_loop_writes_the_span_the_module_stores(
    ungrouped_plan: InstrumentPlan, dump_context: DumpContext
) -> None:
    """The same encode under the same seed, so the file written for a voice is the module's own waveform."""
    for unit in build_units(ungrouped_plan, replace(dump_context, settings=_dropped_tail(dump_context))):
        assert np.array_equal(unit.whole.pcm, unit.stored.pcm)


def test_make_kind_packages_an_ungrouped_plan(ungrouped_plan: InstrumentPlan, dump_context: DumpContext) -> None:
    kind = make_kind(ungrouped_plan, dump_context)
    assert kind.name == "ungrouped"
    assert kind.plan_document.strategy == "ungrouped"
    assert len(kind.units) == len(ungrouped_plan.pitches)
    assert kind.report_text.strip()  # a non-empty human report
    assert len(kind.module.song.samples) == len(kind.units)  # one stored sample per unit
    assert kind.plan_document.module.total_bytes == kind.module.size().total


def test_make_kind_rebuilds_the_module_over_the_material_it_was_written_for(
    ungrouped_plan: InstrumentPlan, dump_context: DumpContext
) -> None:
    """The plan alone settles the waveforms, so asking for the same material again writes the same module."""
    kind = make_kind(ungrouped_plan, dump_context)

    again = kind.make_module(list(dump_context.material))

    assert again.song.samples == kind.module.song.samples


def test_a_carriers_waveform_follows_the_material_its_instrument_plays(
    ungrouped_plan: InstrumentPlan, dump_context: DumpContext
) -> None:
    """A carrier is its recording divided by the curve its instrument carries, and that curve is fitted to
    what the instrument plays -- so a module rebuilt over other material stores the same plan through a
    curve of its own. The stored format stands where the plan put it; only the waveform under it moves.
    """
    kind = make_kind(ungrouped_plan, dump_context)

    one_note = kind.make_module(list(dump_context.material[:1]))

    stored = [(sample.name, sample.rate, sample.depth) for sample in one_note.song.samples]
    assert stored == [(sample.name, sample.rate, sample.depth) for sample in kind.module.song.samples]
    assert one_note.song.samples != kind.module.song.samples  # the curve followed the material it was given
    assert one_note.size().patterns < kind.module.size().patterns


def test_make_kind_packages_a_grouped_plan(grouped_plan: GroupedInstrumentPlan, dump_context: DumpContext) -> None:
    kind = make_kind(grouped_plan, dump_context)
    assert kind.name == "grouped"
    assert kind.plan_document.strategy == "grouped"
    assert len(kind.units) == len(grouped_plan.zones)  # one stored sample per zone
