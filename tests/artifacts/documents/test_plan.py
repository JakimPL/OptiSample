from __future__ import annotations

import numpy as np

from optisample.artifacts.documents.plan import plan_document
from optisample.dsp.level import Clock, Level, read_level, unit_level
from optisample.dsp.series import Readings
from optisample.dsp.surrogate import StoredSample
from optisample.optimize.export.coverage import KeyCoverage
from optisample.optimize.export.voices import NO_DRIFT, WrittenInstruments
from optisample.optimize.layers.slots import pack_slots
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan, StrategyPlan
from trackmod.module.size import SizeReport

_SIZE = SizeReport(patterns=120, pcm=9000, headers=800, largest_pattern=90)
_COVERAGE = KeyCoverage(numbered=120, played=3, answered=120)  # a keyboard answered in full from 3 recordings
_WHOLE_TABLE = 255  # samples one instrument reaches in a format numbering them freely
_ONE_SAMPLE_EACH = 1  # the narrowest format imaginable, which writes every stored zone on its own


def _written(
    plan: StrategyPlan,
    per_instrument: int = _WHOLE_TABLE,
    drift_db: float = NO_DRIFT,
) -> WrittenInstruments:
    """The instruments the plan is written as, which the document holds one record each of."""
    layout = pack_slots(plan.sample_units(), plan.layers, per_instrument)
    return WrittenInstruments(layout=layout, drifts=(drift_db,) * layout.count)


def _encoded(plan: StrategyPlan, *, level: Level = unit_level(Clock.PLAYED)) -> list[StoredSample]:
    """One re-encoded sample per plan item, which is where the document reads a loop and a level off."""
    return [
        StoredSample(
            pcm=np.zeros(unit.frames, dtype=np.float64),
            sample_rate=unit.params.target_rate,
            depth_bits=unit.params.depth_bits,
            root_pitch=unit.representative,
            level=level,
        )
        for unit in plan.sample_units()
    ]


def test_plan_document_ungrouped_writes_pitches_and_method(ungrouped_plan: InstrumentPlan) -> None:
    doc = plan_document(ungrouped_plan, _encoded(ungrouped_plan), _SIZE, _COVERAGE, _written(ungrouped_plan))
    assert doc.strategy == "ungrouped"
    assert doc.method is not None and doc.pitches is not None and doc.zones is None
    assert doc.budget.used_bytes == ungrouped_plan.used_bytes
    assert doc.module.total_bytes == _SIZE.total  # what the written module occupies, material included
    dumped = doc.model_dump()
    assert "zones" not in dumped  # the absent optional head is dropped at serialization
    assert "method" in dumped and "pitches" in dumped


def test_plan_document_grouped_writes_zones_and_drops_method(grouped_plan: GroupedInstrumentPlan) -> None:
    doc = plan_document(grouped_plan, _encoded(grouped_plan), _SIZE, _COVERAGE, _written(grouped_plan))
    assert doc.strategy == "grouped"
    assert doc.zones is not None and doc.method is None and doc.pitches is None
    dumped = doc.model_dump()
    assert "method" not in dumped and "pitches" not in dumped  # both absent heads dropped
    assert "zones" in dumped


def test_a_grouped_document_records_the_sample_cap_it_was_held_to(
    ungrouped_plan: InstrumentPlan, grouped_plan: GroupedInstrumentPlan
) -> None:
    """The cap and its charge belong to the strategy that meets them, so only that document states them."""
    grouped = plan_document(grouped_plan, _encoded(grouped_plan), _SIZE, _COVERAGE, _written(grouped_plan))
    assert grouped.reserve is not None
    assert grouped.reserve.cap == grouped_plan.reserve.cap
    assert grouped.reserve.bytes_per_sample == grouped_plan.reserve.bytes_per_sample
    assert grouped.reserve.objective_uncapped == grouped_plan.reserve.objective_uncapped
    assert len(grouped.zones or []) <= grouped.reserve.cap

    ungrouped = plan_document(ungrouped_plan, _encoded(ungrouped_plan), _SIZE, _COVERAGE, _written(ungrouped_plan))
    assert "reserve" not in ungrouped.model_dump()


def test_both_documents_state_the_weighting_their_objective_was_measured_under(
    ungrouped_plan: InstrumentPlan, grouped_plan: GroupedInstrumentPlan
) -> None:
    """A bare objective says nothing on its own, so each document records what scaled the notes into it."""
    for plan in (ungrouped_plan, grouped_plan):
        doc = plan_document(plan, _encoded(plan), _SIZE, _COVERAGE, _written(plan))
        assert doc.energy_exponent == plan.energy_exponent
        assert "energy_exponent" in doc.model_dump()


def test_an_item_states_the_curve_its_looped_sample_is_played_down_by(grouped_plan: GroupedInstrumentPlan) -> None:
    """A looped sample holds one level, so the plan records what brings it down beside the loop itself."""
    ramp = read_level(Readings(values=np.asarray([0.0, -16.5]), seconds=np.asarray([0.6, 3.0])), Clock.PLAYED)

    doc = plan_document(grouped_plan, _encoded(grouped_plan, level=ramp), _SIZE, _COVERAGE, _written(grouped_plan))
    plain = plan_document(grouped_plan, _encoded(grouped_plan), _SIZE, _COVERAGE, _written(grouped_plan))

    zone = (doc.zones or [])[0]
    assert (zone.level.seconds, zone.level.values_db) == ([0.6, 3.0], [0.0, -16.5])
    assert (plain.zones or [])[0].level.values_db == [0.0]  # a sample the recording states no decline for


def test_the_document_holds_one_record_per_written_instrument(grouped_plan: GroupedInstrumentPlan) -> None:
    """A consumer reads the written instruments off this list, so each is named and placed as written."""
    units = grouped_plan.sample_units()
    doc = plan_document(
        grouped_plan, _encoded(grouped_plan), _SIZE, _COVERAGE, _written(grouped_plan, _ONE_SAMPLE_EACH)
    )
    assert [record.index for record in doc.instruments] == list(range(len(units)))
    assert [record.samples for record in doc.instruments] == [1] * len(units)
    assert all(record.name.startswith(grouped_plan.instrument_id) for record in doc.instruments)
    assert sum(record.stored_bytes for record in doc.instruments) == doc.budget.used_bytes


def test_an_instrument_record_states_what_its_one_envelope_leaves_the_key_it_suits_worst(
    grouped_plan: GroupedInstrumentPlan,
) -> None:
    """The reading a reader decides a split on, so the document states the one the module was written with."""
    worst_db = 4.5

    doc = plan_document(
        grouped_plan, _encoded(grouped_plan), _SIZE, _COVERAGE, _written(grouped_plan, drift_db=worst_db)
    )

    assert [record.envelope_drift_db for record in doc.instruments] == [worst_db] * len(doc.instruments)


def test_an_instrument_record_states_the_keys_it_was_stored_for(grouped_plan: GroupedInstrumentPlan) -> None:
    units = grouped_plan.sample_units()
    doc = plan_document(grouped_plan, _encoded(grouped_plan), _SIZE, _COVERAGE, _written(grouped_plan))
    (record,) = [entry for entry in doc.instruments if entry.keys > 0]
    assert record.lowest_pitch == min(key for unit in units for key in unit.keys)
    assert record.highest_pitch == max(key for unit in units for key in unit.keys)


def test_plan_document_carries_the_reduction_both_strategies_share(
    ungrouped_plan: InstrumentPlan, grouped_plan: GroupedInstrumentPlan
) -> None:
    """The pre-optimization stage runs once per instrument, so both documents record the same outcome."""
    ungrouped = plan_document(ungrouped_plan, _encoded(ungrouped_plan), _SIZE, _COVERAGE, _written(ungrouped_plan))
    grouped = plan_document(grouped_plan, _encoded(grouped_plan), _SIZE, _COVERAGE, _written(grouped_plan))
    assert ungrouped.reduction == grouped.reduction
    assert ungrouped.reduction.kept_recordings == len(ungrouped.reduction.recordings)
    assert "reduction" in ungrouped.model_dump()


def test_the_reduction_document_records_every_kept_recording_and_narrowed_grid(
    ungrouped_plan: InstrumentPlan,
) -> None:
    reduction = plan_document(
        ungrouped_plan, _encoded(ungrouped_plan), _SIZE, _COVERAGE, _written(ungrouped_plan)
    ).reduction
    assert reduction.listed_recordings >= reduction.kept_recordings
    assert reduction.played_notes >= reduction.scored_classes
    recording = reduction.recordings[0]
    assert recording.key and recording.duration_s > 0.0
    coverage = recording.duration_s >= recording.required_duration_s
    assert recording.covers_material is coverage
    grid = reduction.grids[0]
    assert grid.note and 0.0 < grid.useful_rate_hz
    assert grid.stored.target_rate >= grid.useful_rate_hz  # the rung reaches the band it was settled from
    assert set(grid.stored.depths) <= {8, 16} and grid.swept > 0
