from __future__ import annotations

import math

import numpy as np

from optisample.artifacts.serialize import _json_safe, metrics_document, plan_document
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from trackmod.module.size import SizeReport

_SIZE = SizeReport(patterns=120, pcm=9000, headers=800, largest_pattern=90)


def test_json_safe_coerces_numpy_and_non_finite() -> None:
    out = _json_safe({"a": np.float64(1.5), "b": np.int64(3), "c": math.inf, "d": [-math.inf, 2.0]})
    assert out == {"a": 1.5, "b": 3, "c": None, "d": [None, 2.0]}
    assert isinstance(out["b"], int) and not isinstance(out["b"], np.integer)


def test_plan_document_ungrouped_writes_pitches_and_method(ungrouped_plan: InstrumentPlan) -> None:
    units = ungrouped_plan.sample_units()
    doc = plan_document(ungrouped_plan, [None] * len(units), _SIZE)
    assert doc.strategy == "ungrouped"
    assert doc.method is not None and doc.pitches is not None and doc.zones is None
    assert doc.budget.used_bytes == ungrouped_plan.used_bytes
    assert doc.module.total_bytes == _SIZE.total  # what the written module occupies, material included
    dumped = doc.model_dump()
    assert "zones" not in dumped  # the absent optional head is dropped at serialization
    assert "method" in dumped and "pitches" in dumped


def test_plan_document_grouped_writes_zones_and_drops_method(grouped_plan: GroupedInstrumentPlan) -> None:
    units = grouped_plan.sample_units()
    doc = plan_document(grouped_plan, [None] * len(units), _SIZE)
    assert doc.strategy == "grouped"
    assert doc.zones is not None and doc.method is None and doc.pitches is None
    dumped = doc.model_dump()
    assert "method" not in dumped and "pitches" not in dumped  # both absent heads dropped
    assert "zones" in dumped


def test_metrics_document_objective_sums_note_contributions() -> None:
    doc = metrics_document("ungrouped", "piano", 44_100, plan_objective=1.5, notes=[])
    assert doc.objective == 0.0  # no notes → no contribution
    assert doc.plan_objective == 1.5  # the plan objective is carried through verbatim
    assert doc.notes == []


def test_plan_document_carries_the_reduction_both_strategies_share(
    ungrouped_plan: InstrumentPlan, grouped_plan: GroupedInstrumentPlan
) -> None:
    """The pre-optimization stage runs once per instrument, so both documents record the same outcome."""
    ungrouped = plan_document(ungrouped_plan, [None] * len(ungrouped_plan.sample_units()), _SIZE)
    grouped = plan_document(grouped_plan, [None] * len(grouped_plan.sample_units()), _SIZE)
    assert ungrouped.reduction == grouped.reduction
    assert ungrouped.reduction.kept_recordings == len(ungrouped.reduction.recordings)
    assert "reduction" in ungrouped.model_dump()


def test_the_reduction_document_records_every_kept_recording_and_narrowed_grid(
    ungrouped_plan: InstrumentPlan,
) -> None:
    reduction = plan_document(ungrouped_plan, [None] * len(ungrouped_plan.sample_units()), _SIZE).reduction
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
