"""How a plan's velocity layers reach the written module: one instrument each, named per note."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.config.render import RenderConfig
from optisample.io.render import openmpt123_available, render_module
from optisample.optimize.export.build import instrument_name
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.plans import GroupedInstrumentPlan
from tests.optimize.export.demo import demo_material
from tests.optimize.export.test_material import song_cells
from trackmod.core.notes.command import NoteCommand
from trackmod.module.protocol import TrackerModule

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")


def test_the_demo_material_earns_a_velocity_split(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    """Pitch 60 is played softly and loudly, so storing a recording for each beats one for both."""
    plan, _ = layered_build()
    assert plan.layers.count > 1
    assert [band.label for band in plan.layers.bands] == ["v000-v050", "v051-v127"]


def test_a_layered_plan_writes_one_instrument_per_velocity_band(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, module = layered_build()
    instruments = module.song.instruments
    assert len(instruments) == plan.layers.count
    assert [instrument.name for instrument in instruments] == [f"piano {band.label}" for band in plan.layers.bands]


def test_every_note_plays_through_the_layer_the_plan_assigned_its_dynamic(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, module = layered_build()
    notes = [cell for cell in song_cells(module) if cell.note != NoteCommand.CUT]
    played = [event.velocity for event in demo_material()]
    assert [cell.instrument for cell in notes] == [plan.layers.band_index(velocity) for velocity in played]
    assert {cell.instrument for cell in notes} == set(range(plan.layers.count))  # every layer is reached


def test_each_layer_routes_only_to_the_samples_its_own_band_stores(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, module = layered_build()
    stored: dict[int, set[int]] = {layer: set() for layer in range(plan.layers.count)}
    for index, unit in enumerate(plan.sample_units()):
        stored[unit.layer].add(index)

    for layer, instrument in enumerate(module.song.instruments):
        assert set(instrument.samples) == stored[layer]


def test_each_layer_stores_the_recording_nearest_its_own_loudest_dynamic(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, _ = layered_build()
    for unit in plan.sample_units():
        band = plan.layers.bands[unit.layer]
        assert band.covers(unit.representative_key.velocity)


def test_a_note_of_an_unplayed_dynamic_still_resolves_to_a_layer(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    """The bands tile the whole velocity axis, so a key struck harder than anything recorded still sounds."""
    plan, _ = layered_build()
    assert plan.layers.band_index(127) == plan.layers.count - 1
    assert plan.layers.band_index(0) == 0


@pytest.mark.parametrize(
    ("layers", "layer", "expected"),
    [
        (UNSPLIT, 0, "piano"),
        (VelocityLayers((VelocityBand(0, 50), VelocityBand(51, 127))), 0, "piano v000-v050"),
        (VelocityLayers((VelocityBand(0, 50), VelocityBand(51, 127))), 1, "piano v051-v127"),
    ],
    ids=["one-layer-keeps-the-instrument-name", "quiet-band", "loud-band"],
)
def test_instrument_name_states_the_band_a_split_tells_apart(layers: VelocityLayers, layer: int, expected: str) -> None:
    assert instrument_name("piano", layers, layer) == expected


@requires_openmpt
def test_the_written_layered_module_sounds_through_the_real_engine(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]], render_config: RenderConfig
) -> None:
    _, module = layered_build()
    assert module.violations() == ()
    audio, rate = render_module(module, render_config)
    assert rate == render_config.sample_rate
    assert float(np.max(np.abs(audio))) > 0.0  # every layer's instrument resolves to a sample that plays
