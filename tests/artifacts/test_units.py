from __future__ import annotations

import numpy as np

from optisample.artifacts.context import DumpContext
from optisample.artifacts.units import build_units, make_kind
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan


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


def test_make_kind_packages_an_ungrouped_plan(ungrouped_plan: InstrumentPlan, dump_context: DumpContext) -> None:
    kind = make_kind(ungrouped_plan, dump_context)
    assert kind.name == "ungrouped"
    assert kind.plan_document.strategy == "ungrouped"
    assert len(kind.units) == len(ungrouped_plan.pitches)
    assert kind.report_text.strip()  # a non-empty human report
    assert len(kind.module.song.samples) == len(kind.units)  # one stored sample per unit
    assert kind.plan_document.module.total_bytes == kind.module.size().total


def test_make_kind_rebuilds_the_module_over_other_material(
    ungrouped_plan: InstrumentPlan, dump_context: DumpContext
) -> None:
    kind = make_kind(ungrouped_plan, dump_context)
    one_note = kind.make_module(list(dump_context.material[:1]))
    assert one_note.song.samples == kind.module.song.samples  # the same stored samples, a shorter song
    assert one_note.size().patterns < kind.module.size().patterns


def test_make_kind_packages_a_grouped_plan(grouped_plan: GroupedInstrumentPlan, dump_context: DumpContext) -> None:
    kind = make_kind(grouped_plan, dump_context)
    assert kind.name == "grouped"
    assert kind.plan_document.strategy == "grouped"
    assert len(kind.units) == len(grouped_plan.zones)  # one stored sample per zone
