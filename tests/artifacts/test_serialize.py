from __future__ import annotations

import math

import numpy as np

from optisample.artifacts.serialize import _json_safe, metrics_document, plan_document
from optisample.optimize.export.coverage import KeyCoverage
from optisample.optimize.layers.slots import SlotLayout, pack_slots
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan, StrategyPlan
from trackmod.module.size import SizeReport

_SIZE = SizeReport(patterns=120, pcm=9000, headers=800, largest_pattern=90)
_COVERAGE = KeyCoverage(numbered=120, played=3, answered=120)  # a keyboard answered in full from 3 recordings
_WHOLE_TABLE = 255  # samples one instrument reaches in a format numbering them freely
_ONE_SAMPLE_EACH = 1  # the narrowest format imaginable, which writes every stored zone on its own


def _layout(plan: StrategyPlan, per_instrument: int = _WHOLE_TABLE) -> SlotLayout:
    """The instruments the plan is written as, which the document holds one record each of."""
    return pack_slots(plan.sample_units(), plan.layers, per_instrument)


def test_json_safe_coerces_numpy_and_non_finite() -> None:
    out = _json_safe({"a": np.float64(1.5), "b": np.int64(3), "c": math.inf, "d": [-math.inf, 2.0]})
    assert out == {"a": 1.5, "b": 3, "c": None, "d": [None, 2.0]}
    assert isinstance(out["b"], int) and not isinstance(out["b"], np.integer)


def test_plan_document_ungrouped_writes_pitches_and_method(ungrouped_plan: InstrumentPlan) -> None:
    units = ungrouped_plan.sample_units()
    doc = plan_document(ungrouped_plan, [None] * len(units), _SIZE, _COVERAGE, _layout(ungrouped_plan))
    assert doc.strategy == "ungrouped"
    assert doc.method is not None and doc.pitches is not None and doc.zones is None
    assert doc.budget.used_bytes == ungrouped_plan.used_bytes
    assert doc.module.total_bytes == _SIZE.total  # what the written module occupies, material included
    dumped = doc.model_dump()
    assert "zones" not in dumped  # the absent optional head is dropped at serialization
    assert "method" in dumped and "pitches" in dumped


def test_plan_document_grouped_writes_zones_and_drops_method(grouped_plan: GroupedInstrumentPlan) -> None:
    units = grouped_plan.sample_units()
    doc = plan_document(grouped_plan, [None] * len(units), _SIZE, _COVERAGE, _layout(grouped_plan))
    assert doc.strategy == "grouped"
    assert doc.zones is not None and doc.method is None and doc.pitches is None
    dumped = doc.model_dump()
    assert "method" not in dumped and "pitches" not in dumped  # both absent heads dropped
    assert "zones" in dumped


def test_both_documents_state_the_weighting_their_objective_was_measured_under(
    ungrouped_plan: InstrumentPlan, grouped_plan: GroupedInstrumentPlan
) -> None:
    """A bare objective says nothing on its own, so each document records what scaled the notes into it."""
    for plan in (ungrouped_plan, grouped_plan):
        doc = plan_document(plan, [None] * len(plan.sample_units()), _SIZE, _COVERAGE, _layout(plan))
        assert doc.energy_exponent == plan.energy_exponent
        assert "energy_exponent" in doc.model_dump()


def test_metrics_document_objective_sums_note_contributions() -> None:
    doc = metrics_document("ungrouped", "piano", 44_100, plan_objective=1.5, notes=[])
    assert doc.objective == 0.0  # no notes → no contribution
    assert doc.plan_objective == 1.5  # the plan objective is carried through verbatim
    assert doc.notes == []


def test_the_document_holds_one_record_per_written_instrument(grouped_plan: GroupedInstrumentPlan) -> None:
    """A consumer reads the written instruments off this list, so each is named and placed as written."""
    units = grouped_plan.sample_units()
    doc = plan_document(grouped_plan, [None] * len(units), _SIZE, _COVERAGE, _layout(grouped_plan, _ONE_SAMPLE_EACH))
    assert [record.index for record in doc.instruments] == list(range(len(units)))
    assert [record.samples for record in doc.instruments] == [1] * len(units)
    assert all(record.name.startswith(grouped_plan.instrument_id) for record in doc.instruments)
    assert sum(record.stored_bytes for record in doc.instruments) == doc.budget.used_bytes


def test_an_instrument_record_states_the_keys_it_was_stored_for(grouped_plan: GroupedInstrumentPlan) -> None:
    units = grouped_plan.sample_units()
    doc = plan_document(grouped_plan, [None] * len(units), _SIZE, _COVERAGE, _layout(grouped_plan))
    (record,) = [entry for entry in doc.instruments if entry.keys > 0]
    assert record.lowest_pitch == min(key for unit in units for key in unit.keys)
    assert record.highest_pitch == max(key for unit in units for key in unit.keys)


def test_plan_document_carries_the_reduction_both_strategies_share(
    ungrouped_plan: InstrumentPlan, grouped_plan: GroupedInstrumentPlan
) -> None:
    """The pre-optimization stage runs once per instrument, so both documents record the same outcome."""
    ungrouped = plan_document(
        ungrouped_plan, [None] * len(ungrouped_plan.sample_units()), _SIZE, _COVERAGE, _layout(ungrouped_plan)
    )
    grouped = plan_document(
        grouped_plan, [None] * len(grouped_plan.sample_units()), _SIZE, _COVERAGE, _layout(grouped_plan)
    )
    assert ungrouped.reduction == grouped.reduction
    assert ungrouped.reduction.kept_recordings == len(ungrouped.reduction.recordings)
    assert "reduction" in ungrouped.model_dump()


def test_the_reduction_document_records_every_kept_recording_and_narrowed_grid(
    ungrouped_plan: InstrumentPlan,
) -> None:
    reduction = plan_document(
        ungrouped_plan, [None] * len(ungrouped_plan.sample_units()), _SIZE, _COVERAGE, _layout(ungrouped_plan)
    ).reduction
    assert reduction.listed_recordings >= reduction.kept_recordings
    assert reduction.played_notes >= reduction.scored_classes
    recording = reduction.recordings[0]
    assert recording.key and recording.duration_s > 0.0
    coverage = recording.duration_s >= recording.required_duration_s
    assert recording.covers_material is coverage
    grid = reduction.grids[0]
    assert grid.note and 0.0 < grid.useful_rate_hz
    assert 0 < len(grid.shortlist) <= reduction.grid_size
    assert all(encoding.depth_bits in (8, 16) for encoding in grid.shortlist)
