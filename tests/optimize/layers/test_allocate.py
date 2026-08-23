from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.layers import LayersConfig
from optisample.config.optimize import SweepConfig
from optisample.config.tracker import TrackerFormat
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping.optimize import optimize_instrument_grouped
from optisample.optimize.layers.allocate import (
    LayeredAllocation,
    _Layering,
    _universe,
    allocate_layers,
    fits_format,
    preference,
)
from optisample.optimize.layers.bands import (
    VelocityBand,
    VelocityLayers,
    partitions,
    velocity_cells,
)
from optisample.optimize.orchestrate import prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import (
    FIRST_LAYER,
    NO_RESERVE,
    SINGLE_LAYER,
    SampleReserve,
    split_budget,
)
from optisample.optimize.tasks import AudioMap, StoredRecordings

Recordings = Callable[..., StoredRecordings]

SR = 44_100  # the rate the shared ``piano_note`` factory renders at
PITCHES = (60, 64)
VELOCITIES = (20, 70, 120)
_NOTE_S = 0.3
_GENEROUS_KB = 96.0
_TIGHT_KB = 3.0  # room for one layer's cheapest samples, and not for a second layer's
_ONE_LAYER = 1
_THREE_LAYERS = 3
_WHOLE_AXIS = VelocityBand(0, MIDI_MAX_VELOCITY)
_ONE_SAMPLE = 1  # the tightest cap there is, which only a plan of one zone per layer meets
_UNCHARGED = SampleReserve(cap=_THREE_LAYERS, bytes_per_sample=NO_RESERVE, objective_uncapped=0.0)


@pytest.fixture
def audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    """One recording per ``(pitch, velocity)`` -- the grid a layer picks its own representative from."""
    return {
        SampleKey(pitch, velocity): piano_note(pitch, velocity, dur=_NOTE_S, seed=pitch * 191 + velocity)
        for pitch in PITCHES
        for velocity in VELOCITIES
    }


@pytest.fixture
def instrument() -> InstrumentSpec:
    """Every key played at all three dynamics, so each velocity band covers the whole keyboard."""
    return InstrumentSpec(
        id="piano",
        budget_kb=_GENEROUS_KB,
        samples=[
            SourceSample(file=Path(f"{pitch}_{velocity}.wav"), pitch=pitch, velocity=velocity)
            for pitch in PITCHES
            for velocity in VELOCITIES
        ],
        material=[
            NoteEvent(pitch=pitch, velocity=velocity, duration_s=_NOTE_S, count=2)
            for pitch in PITCHES
            for velocity in VELOCITIES
        ],
    )


@pytest.fixture
def allocate(
    audio: AudioMap,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    layers: Callable[..., LayersConfig],
    recordings: Recordings,
) -> Callable[..., LayeredAllocation]:
    """Factory: run the layered search over ``instrument`` with the layering knobs a test varies.

    ``max_samples`` caps what the split may store between its layers; the rest of the overrides are the
    layering config's own.
    """

    def _allocate(
        instrument: InstrumentSpec, *, max_samples: int | None = None, **overrides: object
    ) -> LayeredAllocation:
        settings = optimize_settings(
            sweep=sweep(rates=(11_025,), depths=(8,), dither=False),
            layers=layers(nodes=len(VELOCITIES), **overrides),
            max_samples=max_samples,
        )
        return allocate_layers(instrument, prepare_run(instrument, recordings(audio, SR), settings), settings)

    return _allocate


def test_one_layer_stores_a_single_band_over_the_whole_axis(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """The behaviour freeze: capping the layers at one is the plan pitch grouping produced before."""
    allocation = allocate(instrument, max_layers=_ONE_LAYER)
    assert allocation.layers == VelocityLayers((_WHOLE_AXIS,))
    assert allocation.budget.instruments == SINGLE_LAYER
    assert {zone.layer for zone in allocation.zones} == {FIRST_LAYER}
    assert [pitch for zone in allocation.zones for pitch in zone.pitches] == list(PITCHES)


def test_the_layers_tile_the_axis_and_each_covers_the_keys_it_plays(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """A note reaches exactly one layer, and that layer holds a zone for every key it is played at."""
    allocation = allocate(instrument, max_layers=_THREE_LAYERS, min_gain=0.0)
    bands = allocation.layers.bands
    assert bands[0].lowest == 0 and bands[-1].highest == MIDI_MAX_VELOCITY
    for below, above in zip(bands, bands[1:]):
        assert above.lowest == below.highest + 1

    for layer in range(allocation.layers.count):
        covered = [pitch for zone in allocation.zones if zone.layer == layer for pitch in zone.pitches]
        assert covered == list(PITCHES)


def test_layers_buy_what_they_exist_to_buy_on_material_played_at_three_dynamics(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """The move layering adds: a quiet note reconstructs from a quiet recording rather than a scaled ff one."""
    one = allocate(instrument, max_layers=_ONE_LAYER)
    many = allocate(instrument, max_layers=_THREE_LAYERS)
    assert many.layers.count > one.layers.count
    assert many.objective < one.objective  # the single-layer split is in the search space and was beaten
    stored = {(zone.layer, zone.representative_key.velocity) for zone in many.zones}
    assert len({velocity for _, velocity in stored}) == many.layers.count  # each layer keeps its own dynamic


def test_a_margin_no_gain_can_meet_keeps_the_single_layer_plan(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """``min_gain`` is the taste dial: ask an impossible improvement of a layer and none is stored."""
    demanding = allocate(instrument, max_layers=_THREE_LAYERS, min_gain=100.0)
    assert demanding.layers.count == _ONE_LAYER
    assert demanding.objective == pytest.approx(allocate(instrument, max_layers=_ONE_LAYER).objective)


def test_a_band_several_splits_share_is_scored_once_between_them(
    instrument: InstrumentSpec,
    audio: AudioMap,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    layers: Callable[..., LayersConfig],
    recordings: Recordings,
) -> None:
    """A band's keys are the same wherever it appears, so the whole search scores each band once."""
    settings = optimize_settings(
        sweep=sweep(rates=(11_025,), depths=(8,), dither=False),
        layers=layers(max_layers=_THREE_LAYERS, nodes=len(VELOCITIES)),
    )
    inputs = prepare_run(instrument, recordings(audio, SR), settings)
    layering = _Layering(instrument, inputs, settings)
    splits = tuple(partitions(velocity_cells(instrument.material, len(VELOCITIES)), _THREE_LAYERS))

    universe = _universe(layering, splits)

    assert len(universe.segments) == len({band for split in splits for band in split.bands})
    for split, place in zip(splits, universe.placement):
        placed = [[task.pitch for task in universe.segments[segment]] for segment in place]
        assert placed == [[task.pitch for task in layering.keys(band)] for band in split.bands]


def test_the_budget_a_split_is_solved_against_reserves_its_own_instruments(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    allocation = allocate(instrument, max_layers=_THREE_LAYERS, min_gain=0.0)
    assert allocation.budget.instruments == allocation.layers.count
    assert allocation.total_bytes <= allocation.budget.sample_bytes


def test_the_material_bounds_how_many_layers_are_offered(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """A single velocity cuts into one cell, so a cap of three layers still stores one."""
    one_dynamic = instrument.model_copy(
        update={"material": [NoteEvent(pitch=pitch, velocity=70, duration_s=_NOTE_S, count=2) for pitch in PITCHES]}
    )
    allocation = allocate(one_dynamic, max_layers=_THREE_LAYERS, min_gain=0.0)
    assert allocation.layers == VelocityLayers((_WHOLE_AXIS,))


def test_a_budget_too_small_for_one_layer_fails_as_it_always_did(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """The single-layer split is the cheapest floor, so its infeasibility is the whole run's."""
    with pytest.raises(BudgetInfeasibleError):
        allocate(instrument.model_copy(update={"budget_kb": 1.0}), max_layers=_THREE_LAYERS)


def test_a_split_the_budget_cannot_carry_is_passed_over(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """More layers means more mandatory samples, so a budget holding one layer may hold only one."""
    tight = allocate(instrument.model_copy(update={"budget_kb": _TIGHT_KB}), max_layers=_THREE_LAYERS, min_gain=0.0)
    assert tight.layers.count == _ONE_LAYER
    assert tight.total_bytes <= tight.budget.sample_bytes


def test_a_split_reserving_more_instruments_than_the_format_numbers_is_passed_over(
    target: ExportTarget,
) -> None:
    """Layers each cut into the samples one instrument owns is how a split reaches the instrument count."""
    crowded = LayeredAllocation(
        layers=VelocityLayers((_WHOLE_AXIS,)),
        budget=split_budget(_GENEROUS_KB, target.storage, target.max_instruments + 1),
        zones=(),
        total_bytes=100,
        objective=0.0,
        reserve=_UNCHARGED,
    )
    assert not fits_format(crowded, target)
    listed = split_budget(_GENEROUS_KB, target.storage, target.max_instruments)
    assert fits_format(replace(crowded, budget=listed), target)


def test_the_sample_cap_counts_every_layer_together(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """One stored sample between them leaves one layer holding one zone, and prices the richer splits out."""
    capped = allocate(instrument, max_layers=_THREE_LAYERS, min_gain=0.0, max_samples=_ONE_SAMPLE)
    assert capped.reserve.cap == _ONE_SAMPLE
    assert len(capped.zones) == _ONE_SAMPLE
    assert capped.layers.count == _ONE_LAYER
    assert [pitch for zone in capped.zones for pitch in zone.pitches] == list(PITCHES)  # one sample, both keys
    assert capped.total_bytes <= capped.budget.sample_bytes


def test_a_split_the_layers_alone_would_earn_is_given_up_to_the_cap(
    instrument: InstrumentSpec,
    allocate: Callable[..., LayeredAllocation],
) -> None:
    """A layer costs a sample of its own, so the cap decides how much vocabulary the plan can afford."""
    free = allocate(instrument, max_layers=_THREE_LAYERS, min_gain=0.0)
    capped = allocate(instrument, max_layers=_THREE_LAYERS, min_gain=0.0, max_samples=_ONE_SAMPLE)
    assert free.layers.count > capped.layers.count
    assert capped.objective > free.objective  # storing fewer samples costs fidelity


def test_asking_for_more_layers_than_the_format_numbers_is_refused(
    instrument: InstrumentSpec,
    audio: AudioMap,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    layers: Callable[..., LayersConfig],
    target: ExportTarget,
    recordings: Recordings,
) -> None:
    settings = optimize_settings(
        sweep=sweep(rates=(11_025,), depths=(8,), dither=False),
        layers=layers(max_layers=target.max_instruments + 1),
    )
    inputs = prepare_run(instrument, recordings(audio, SR), settings)
    with pytest.raises(ValueError, match="velocity layers"):
        allocate_layers(instrument, inputs, settings)


def test_the_plan_carries_the_layers_it_settled_on(
    instrument: InstrumentSpec,
    audio: AudioMap,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    recordings: Recordings,
) -> None:
    plan = optimize_instrument_grouped(
        instrument,
        recordings(audio, SR),
        optimize_settings(sweep=sweep(rates=(11_025,), depths=(8,), dither=False)),
    )
    assert plan.layers.count == plan.budget.instruments
    assert {unit.layer for unit in plan.sample_units()} == {zone.layer for zone in plan.zones}


def test_the_format_answers_how_many_layers_and_samples_it_numbers(
    retarget: Callable[[TrackerFormat], ExportTarget],
) -> None:
    """The caps a layered plan is held to come off the same table every other bound is read from."""
    for tracker in (TrackerFormat.IT, TrackerFormat.XM):
        target = retarget(tracker)
        assert target.max_instruments >= _THREE_LAYERS
        assert target.max_samples >= target.max_instruments


@pytest.mark.parametrize(("count", "expected"), [(1, 10.0), (2, 11.0), (3, 12.1)])
def test_each_extra_layer_raises_what_a_split_has_to_beat(count: int, expected: float) -> None:
    """A layer earns its place by ``min_gain``, so the mark-up compounds with how many are stored."""
    allocation = LayeredAllocation(
        layers=VelocityLayers(tuple(VelocityBand(index, index) for index in range(count))),
        budget=None,  # type: ignore[arg-type]
        zones=(),
        total_bytes=0,
        objective=10.0,
        reserve=_UNCHARGED,
    )
    assert preference(allocation, 0.1) == pytest.approx(expected)
