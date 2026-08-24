from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.dsp.level import gain_to_db
from optisample.dsp.surrogate import render
from optisample.dsp.surrogate.sample import CARRIES_ITS_LEVEL
from optisample.io.tracker.envelope import NO_ENVELOPE, carried_signal
from optisample.keys import SampleKey
from optisample.optimize.export.build import layout_units, module_carries, written_voices
from optisample.optimize.export.carriers import plan_trajectories, planned_reach_s
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.samples import unit_envelopes
from optisample.optimize.export.voices import NO_SHAPE, PlannedVoices
from optisample.optimize.layers.slots import SlotLayout, plan_slots
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import InstrumentPlan
from optisample.optimize.tasks import StoredRecordings
from tests.optimize.export.demo import SR, demo_instrument, demo_material

Recordings = Callable[..., StoredRecordings]

_TEMPO = 125
_FLATTER_DB = 1.0  # how much closer to level a divided waveform has to read before the split has done anything
_BLOCKS = 8  # equal stretches a span's level is read over, enough to follow a note's decline across one
_HELD_S = 0.5  # the longest note the demo material holds, which is the stretch a stored sample is heard over
# what the two roundings of a written curve and the codec together leave between a carrier played back
# and the recording it was taken from, the whole of which the measured reading sits an order under
_WRITTEN_DB = 0.5
_EVERY_ATTACK = 0.0  # the gate admitting every recording, whatever its attack asks of a written curve
_NO_ATTACK = 1_000.0  # a gate no recording clears, which is how a module is held to storing recordings
_ONE_TICK = 1.0  # the gate as it ships: a curve needs a tick of run to state an attack across
_STRUCK_S = 0.001  # an attack no written curve has room to state
_SLOW_ATTACK_S = 0.1  # an attack running several ticks, which a curve states with room to spare


@pytest.fixture
def planned(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    demo_audio: dict[SampleKey, NDArray[np.float64]],
    recordings: Recordings,
) -> tuple[InstrumentPlan, StoredRecordings]:
    """A plan over the demo instrument, beside the recordings it was allocated from."""
    settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depths=(16,), carriers=(True,)))
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


def _block_levels(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """The level a span holds over each of :data:`_BLOCKS` equal stretches of it, in decibels."""
    blocks = values[: values.size // _BLOCKS * _BLOCKS].reshape(_BLOCKS, -1)
    return gain_to_db(np.sqrt(np.mean(blocks**2, axis=1)))


def _travel_db(values: NDArray[np.float64]) -> float:
    """How far a span's level moves between the loudest stretch of it and the quietest."""
    levels = _block_levels(values)
    return float(np.max(levels) - np.min(levels))


def _apart_db(played: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """How far two spans stand apart in level, at the stretch they stand furthest apart on."""
    return float(np.max(np.abs(_block_levels(played) - _block_levels(reference))))


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
    context = dataclasses.replace(export_context, min_carried_attack_ticks=_EVERY_ATTACK)
    written = written_voices(plan, layout, held, demo_material(), context)
    envelope = next(curve for curve in written.envelopes if curve is not NO_ENVELOPE)
    unit = next(iter(plan.sample_units()))
    signal = held.audio[unit.representative_key]

    flattened = carried_signal(signal, envelope, tempo=export_context.envelope_grid.tempo, sample_rate=SR)

    assert _travel_db(flattened) < _travel_db(signal) - _FLATTER_DB


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


def test_a_stored_carrier_is_rendered_where_the_recording_it_stands_for_sounded(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
) -> None:
    """The whole of the wiring: a waveform holding no level is scored through the curve that carries it.

    The stored PCM travels level-flat, so playing it as it stands puts a note out nowhere near the level
    its recording held. Playing it through the very curve it was divided by
    (:func:`~optisample.optimize.export.samples._played_through`) is what puts the decline back, and the
    surrogate renderer reads that curve off the sample -- which is what lets the objective price a carrier
    at all.
    """
    plan, held = planned
    layout = plan_slots(plan, export_context.target)
    context = dataclasses.replace(export_context, min_carried_attack_ticks=_EVERY_ATTACK)
    carried = written_voices(plan, layout, held, demo_material(), context)
    unit = next(iter(plan.sample_units()))
    stored = carried.planned.stored[0]
    reference = held.audio[unit.representative_key][: round(_HELD_S * SR)]

    played = render(stored, SR, duration_s=_HELD_S)
    flat = render(dataclasses.replace(stored, level=CARRIES_ITS_LEVEL), SR, duration_s=_HELD_S)

    assert _apart_db(played, reference) < _WRITTEN_DB
    assert _apart_db(flat, reference) > _apart_db(played, reference)


# --- which way round the export runs -------------------------------------------------------------------------


def test_a_module_whose_recordings_rise_too_fast_for_a_curve_stores_them_as_they_were_played(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
) -> None:
    """The gate is what it says it is: one plan lands as two different sets of stored bytes.

    Every sample here was priced as a carrier, so what separates the two is the room a written curve has
    to state the attack each recording makes. Given the room, the level moves onto the curve and the
    waveform is what it leaves; given none, every waveform keeps the level it was played at.
    """
    plan, held = planned
    layout = plan_slots(plan, export_context.target)
    material = demo_material()

    levelled = written_voices(
        plan, layout, held, material, dataclasses.replace(export_context, min_carried_attack_ticks=_NO_ATTACK)
    )
    carried = written_voices(
        plan, layout, held, material, dataclasses.replace(export_context, min_carried_attack_ticks=_EVERY_ATTACK)
    )

    assert [sample.pcm.size for sample in levelled.planned.samples] == [
        sample.pcm.size for sample in carried.planned.samples
    ]
    assert any(
        not np.allclose(left.pcm, right.pcm) for left, right in zip(levelled.planned.samples, carried.planned.samples)
    )


def _slow(frames: int) -> NDArray[np.float64]:
    """A take rising over a stretch several ticks long, which is a level a written curve states with room."""
    rise = round(_SLOW_ATTACK_S * SR)
    envelope = np.concatenate([np.linspace(0.0, 1.0, rise), np.linspace(1.0, 0.2, max(1, frames - rise))])
    tone = np.sin(2.0 * np.pi * 200.0 * np.arange(frames, dtype=np.float64) / SR)
    return tone * envelope[:frames]


def _struck(frames: int) -> NDArray[np.float64]:
    """A take at full level a millisecond in and falling from there, which no written curve states."""
    silent = round(_STRUCK_S * SR)
    falling = np.concatenate([np.zeros(silent), np.linspace(1.0, 0.2, max(1, frames - silent))])
    tone = np.sin(2.0 * np.pi * 200.0 * np.arange(frames, dtype=np.float64) / SR)
    return np.asarray(tone * falling[:frames], dtype=np.float64)


def _takes(held: StoredRecordings, layout: SlotLayout, *, struck: int) -> StoredRecordings:
    """``held`` rebuilt so ``struck`` of the samples the module stores rise faster than a tick.

    The recordings are replaced by the key each stored sample repitches from, since those are the ones the
    module's storage is decided on; anything the plan passed over is left as it stands.
    """
    stored = [unit.representative_key for unit in layout_units(layout)]
    replaced = {key: (_struck if key in stored[:struck] else _slow)(signal.size) for key, signal in held.audio.items()}
    return dataclasses.replace(held, audio=replaced)


def test_a_module_is_written_the_way_most_of_its_samples_asked_to_be(
    planned: tuple[InstrumentPlan, StoredRecordings],
    export_context: ExportContext,
) -> None:
    """The format gives an instrument one envelope for every sample it holds, so the module states one storage.

    Stating the one most of its samples asked for is what keeps a briskly struck take from deciding for a
    set of slow ones, and a slow one from deciding for a set of struck ones. Half is not most, so a module
    split evenly stores its recordings, which is the reading that is right whatever the split.
    """
    plan, held = planned
    layout = plan_slots(plan, export_context.target)
    gated = dataclasses.replace(export_context, min_carried_attack_ticks=_ONE_TICK)
    units = len(layout_units(layout))

    assert module_carries(layout, _takes(held, layout, struck=0), gated)
    assert not module_carries(layout, _takes(held, layout, struck=units // 2), gated)
    assert not module_carries(layout, _takes(held, layout, struck=units), gated)
