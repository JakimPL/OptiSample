from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.dsp.levels import gain_to_db
from optisample.keys import SampleKey
from optisample.optimize.export.build import written_voices
from optisample.optimize.export.carriers import plan_trajectories, planned_reach_s
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.envelope import NO_ENVELOPE
from optisample.optimize.export.samples import carried_signal, unit_envelopes
from optisample.optimize.export.voices import NO_SHAPE, PlannedVoices
from optisample.optimize.layers.slots import plan_slots
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import InstrumentPlan
from optisample.optimize.tasks import StoredRecordings
from tests.optimize.export.demo import SR, demo_instrument, demo_material

Recordings = Callable[..., StoredRecordings]

_TEMPO = 125
_FLATTER_DB = 1.0  # how much closer to level a divided waveform has to read before the split has done anything


@pytest.fixture
def planned(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    demo_audio: dict[SampleKey, NDArray[np.float64]],
    recordings: Recordings,
) -> tuple[InstrumentPlan, StoredRecordings]:
    """A plan over the demo instrument, beside the recordings it was allocated from."""
    settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depth=16))
    held = recordings(demo_audio, SR)
    return optimize_instrument(demo_instrument(), held, settings), held


def _voices(plan: InstrumentPlan, held: StoredRecordings) -> PlannedVoices:
    return PlannedVoices(recordings=held, material=demo_material(), velocity_map=plan.velocity_map)


# --- how far a waveform reaches, read off the plan rather than off a stored sample ---------------------------


def test_a_unit_storing_a_loop_reaches_every_moment_asked_of_it(
    planned: tuple[InstrumentPlan, StoredRecordings],
) -> None:
    """A loop wraps for as long as a note is held, so a shape fitted before encoding may follow it anywhere."""
    plan, held = planned
    unit = next(iter(plan.sample_units()))
    looped = dataclasses.replace(unit, params=dataclasses.replace(unit.params, loop_index=0))
    settled = dict(held.settled)
    settled.setdefault(unit.representative_key, ())
    if not settled[unit.representative_key]:
        pytest.skip("the demo settled no loop, so this unit has none to name")

    assert planned_reach_s(looped, held, unit.representative) == np.inf


def test_a_unit_storing_the_played_span_reaches_the_end_of_it(
    planned: tuple[InstrumentPlan, StoredRecordings],
) -> None:
    """A sample stored whole stops where its material does, stretched by the transposition it plays at."""
    plan, held = planned
    unit = next(iter(plan.sample_units()))
    trimmed = dataclasses.replace(unit, params=dataclasses.replace(unit.params, loop_index=None, trim_s=0.5))

    assert planned_reach_s(trimmed, held, unit.representative) == pytest.approx(0.5)
    assert planned_reach_s(trimmed, held, unit.representative + 12) == pytest.approx(0.25)


# --- the shape a slot carries, fitted before anything is stored ----------------------------------------------


def test_every_instrument_the_material_plays_is_fitted_a_shape(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
) -> None:
    """The whole inversion: a curve exists before a single sample does, so a waveform can be divided by it."""
    plan, held = planned
    layout = plan_slots(plan, export_context.target)

    shapes = plan_trajectories(layout, _voices(plan, held), nodes=8)

    assert len(shapes) == layout.count
    assert any(shape is not NO_SHAPE for shape in shapes)


def test_a_shape_states_a_level_for_every_sample_its_instrument_holds(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
) -> None:
    """Each stored sample stands in a group of its own, so its own loudness stays out of the shared curve."""
    plan, held = planned
    layout = plan_slots(plan, export_context.target)

    shapes = plan_trajectories(layout, _voices(plan, held), nodes=8)

    for slot, shape in zip(layout.slots, shapes):
        if shape is NO_SHAPE:
            continue

        assert len(shape.offsets_db) == 1
        assert len(shape.offsets_db[0]) <= len(slot.samples)


# --- what a stored carrier holds -----------------------------------------------------------------------------


def test_a_slot_carrying_no_curve_stores_its_recording_as_it_stands(
    demo_audio: dict[SampleKey, NDArray[np.float64]],
) -> None:
    """There is no level to hand over, so the waveform reaching the encoder is the recording."""
    signal = next(iter(demo_audio.values()))

    assert carried_signal(signal, NO_ENVELOPE, tempo=_TEMPO, sample_rate=SR) is signal


def test_dividing_by_the_written_curve_leaves_a_flatter_waveform(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
    demo_audio: dict[SampleKey, NDArray[np.float64]],
) -> None:
    """The level moves to the envelope, so what the encoder receives travels less than the recording did."""
    plan, held = planned
    layout = plan_slots(plan, export_context.target)
    written = written_voices(plan, layout, held, demo_material(), dataclasses.replace(export_context, carrier=True))
    envelope = next(curve for curve in written.envelopes if curve is not NO_ENVELOPE)
    unit = next(iter(plan.sample_units()))
    signal = held.audio[unit.representative_key]

    flattened = carried_signal(signal, envelope, tempo=export_context.envelope_grid.tempo, sample_rate=SR)

    def travel(values: NDArray[np.float64]) -> float:
        blocks = values[: values.size // 8 * 8].reshape(8, -1)
        levels = gain_to_db(np.sqrt(np.mean(blocks**2, axis=1)))
        return float(np.max(levels) - np.min(levels))

    assert travel(flattened) < travel(signal) - _FLATTER_DB


def test_each_sample_is_handed_the_curve_of_the_instrument_holding_it(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
) -> None:
    """A keymap routes a key to one sample and one instrument, so a waveform is divided by that one curve."""
    plan, held = planned
    layout = plan_slots(plan, export_context.target)
    curves = tuple(range(layout.count))  # stand-ins, so the mapping is what the test reads

    handed = unit_envelopes(layout, curves, len(plan.sample_units()))  # type: ignore[arg-type]

    for index, slot in enumerate(layout.slots):
        for sample in slot.samples:
            assert handed[sample] == index


# --- which way round the export runs -------------------------------------------------------------------------


def test_storing_recordings_and_storing_carriers_write_different_waveforms(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
) -> None:
    """The switch is what it says it is: the same plan lands as two different sets of stored bytes."""
    plan, held = planned
    layout = plan_slots(plan, export_context.target)
    material = demo_material()

    levelled = written_voices(plan, layout, held, material, dataclasses.replace(export_context, carrier=False))
    carried = written_voices(plan, layout, held, material, dataclasses.replace(export_context, carrier=True))

    assert [sample.pcm.size for sample in levelled.planned.samples] == [
        sample.pcm.size for sample in carried.planned.samples
    ]
    assert any(
        not np.allclose(left.pcm, right.pcm) for left, right in zip(levelled.planned.samples, carried.planned.samples)
    )
