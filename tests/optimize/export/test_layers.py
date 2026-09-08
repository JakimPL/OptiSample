from collections.abc import Callable

import numpy as np
import pytest

from optisample.config.render import RenderConfig
from optisample.dsp.surrogate.params import EncodingParams
from optisample.io.render import openmpt123_available, render_module
from optisample.io.tracker.voices import routed_voices
from optisample.keys import SampleKey
from optisample.optimize.export.build import _NAME_CHARS, instrument_name
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import SlotLayout, pack_slots
from optisample.optimize.plans import GroupedInstrumentPlan, SampleUnit
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
    instruments = routed_voices(module.song).instruments
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

    for layer, instrument in enumerate(routed_voices(module.song).instruments):
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


_SPLIT = VelocityLayers((VelocityBand(0, 50), VelocityBand(51, 127)))
_WHOLE_TABLE = 255  # samples an instrument reaches in a format numbering them freely
_CUT_AT = 2  # samples one instrument owns in a format numbering few, so a wider band is written as several
_LONG_ID = "grand piano, close mic, soft pedal"


def _unit(layer: int, pitch: int) -> SampleUnit:
    return SampleUnit(
        label=f"p{pitch:03d}",
        representative_key=SampleKey(pitch, 100),
        layer=layer,
        keys=(pitch,),
        params=EncodingParams(target_rate=22_050, depth_bits=16),
        frames=2400,
        stored_bytes=1000,
        distortion=1.0,
        objective_share=1.0,
        hull_size=1,
        weight=1.0,
    )


def _layout(layers: VelocityLayers, per_instrument: int) -> SlotLayout:
    """One stored key per band per octave, packed as the given format would write them."""
    units = tuple(_unit(layer, pitch) for layer in range(layers.count) for pitch in (60, 72, 84))
    return pack_slots(units, layers, per_instrument)


@pytest.mark.parametrize(
    ("layers", "per_instrument", "index", "expected"),
    [
        (UNSPLIT, _WHOLE_TABLE, 0, "piano"),
        (_SPLIT, _WHOLE_TABLE, 0, "piano v000-v050"),
        (_SPLIT, _WHOLE_TABLE, 1, "piano v051-v127"),
        (UNSPLIT, _CUT_AT, 0, "piano C4-C5"),
        (UNSPLIT, _CUT_AT, 1, "piano C6-C6"),
        (_SPLIT, _CUT_AT, 1, "piano v000-v050 C6-C6"),
    ],
    ids=[
        "one-instrument-keeps-the-instrument-name",
        "quiet-band",
        "loud-band",
        "the-low-keys-of-a-cut-band",
        "the-high-keys-of-a-cut-band",
        "both-axes-split",
    ],
)
def test_instrument_name_states_each_axis_the_plan_split(
    layers: VelocityLayers, per_instrument: int, index: int, expected: str
) -> None:
    assert instrument_name("piano", _layout(layers, per_instrument), index) == expected


def test_a_long_instrument_id_is_shortened_to_leave_the_axes_it_states_room() -> None:
    """Every format writes the name into a fixed field, so the part telling instruments apart survives."""
    name = instrument_name(_LONG_ID, _layout(_SPLIT, _CUT_AT), 1)
    assert name.endswith(" v000-v050 C6-C6")
    assert len(name) <= _NAME_CHARS


def test_a_long_instrument_id_fits_the_field_even_with_nothing_to_state_beside_it() -> None:
    """A document records the name the module holds, so the fit is settled here rather than at the writer."""
    assert instrument_name(_LONG_ID, _layout(UNSPLIT, _WHOLE_TABLE), 0) == _LONG_ID[:_NAME_CHARS]


@requires_openmpt
def test_the_written_layered_module_sounds_through_the_real_engine(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]], render_config: RenderConfig
) -> None:
    _, module = layered_build()
    assert module.violations() == ()
    audio, rate = render_module(module, render_config)
    assert rate == render_config.sample_rate
    assert float(np.max(np.abs(audio))) > 0.0  # every layer's instrument resolves to a sample that plays
