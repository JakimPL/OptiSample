import json
from pathlib import Path

import pytest

from optisample.artifacts.bank import MANIFEST_VERSION, bank_document, write_bank
from optisample.artifacts.paths import PlanPaths
from optisample.dsp.surrogate.params import EncodingParams
from optisample.music import MIDI_HIGHEST_PITCH, MIDI_LOWEST_PITCH, MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout, pack_slots
from optisample.optimize.plans import SampleUnit
from optisample.optimize.reduce.keys import SampleKey

_SPLIT = VelocityLayers((VelocityBand(0, 50), VelocityBand(51, MIDI_MAX_VELOCITY)))
_EXTENSION = ".iti"
_NAME = "piano"
_PER_INSTRUMENT = 12  # samples one instrument owns in a format numbering few of them
_CUT_KEYS = range(60, 84)


def _slot(layer: int, band: VelocityBand) -> InstrumentSlot:
    """One written instrument, which the bank reads for the band it answers."""
    return InstrumentSlot(layer=layer, band=band, samples=(), units=())


def _layout(layers: VelocityLayers) -> SlotLayout:
    """One instrument per band, which is what a format reaching the whole sample table writes."""
    return SlotLayout(layers=layers, slots=tuple(_slot(layer, band) for layer, band in enumerate(layers.bands)))


def _written(paths: PlanPaths, layers: VelocityLayers) -> tuple[Path, ...]:
    return tuple(paths.instrument_file(band.label, _EXTENSION) for band in layers.bands)


def _unit(pitch: int) -> SampleUnit:
    """One stored sample answering a single key, which is what cuts a band into the most instruments."""
    return SampleUnit(
        label=f"p{pitch:03d}",
        representative_key=SampleKey(pitch, MIDI_MAX_VELOCITY),
        layer=0,
        keys=(pitch,),
        params=EncodingParams(target_rate=22_050, depth_bits=16),
        frames=2400,
        stored_bytes=1000,
        distortion=1.0,
        objective_share=1.0,
        hull_size=1,
        weight=1.0,
    )


def _cut_layout() -> SlotLayout:
    """One velocity band written as two instruments, which is what a format numbering few samples gives."""
    return pack_slots(tuple(_unit(pitch) for pitch in _CUT_KEYS), UNSPLIT, _PER_INSTRUMENT)


def _cut_written(paths: PlanPaths, layout: SlotLayout) -> tuple[Path, ...]:
    return tuple(paths.instrument_file(slot.file_label, _EXTENSION) for slot in layout.slots)


@pytest.fixture
def paths(tmp_path: Path) -> PlanPaths:
    return PlanPaths(directory=tmp_path / "grouped")


def test_a_bank_states_one_layer_for_every_band_the_plan_split(paths: PlanPaths) -> None:
    """A note picks its dynamics by picking a layer, so each band the allocation chose is a layer of its own."""
    bank = bank_document(_NAME, _layout(_SPLIT), _written(paths, _SPLIT), paths)
    assert bank.name == _NAME
    assert [layer.select["velocity"].low for layer in bank.layers] == [0, 51]
    assert [layer.select["velocity"].high for layer in bank.layers] == [50, MIDI_MAX_VELOCITY]


def test_a_bank_states_the_version_a_consumer_reads(paths: PlanPaths) -> None:
    bank = bank_document(_NAME, _layout(UNSPLIT), _written(paths, UNSPLIT), paths)
    assert bank.version == MANIFEST_VERSION


def test_a_layer_names_its_files_against_the_directory_the_manifest_sits_in(paths: PlanPaths) -> None:
    """Both are read from beside the manifest, so a tree copied somewhere else still plays."""
    (layer,) = bank_document(_NAME, _layout(UNSPLIT), _written(paths, UNSPLIT), paths).layers
    assert layer.source.file == "instruments/v000-v127.iti"
    assert layer.velocity_map == "velocity_map.json"


def test_every_dynamic_reaches_exactly_one_layer(paths: PlanPaths) -> None:
    """The plan's bands tile the velocity axis, and stating each one keeps that true of the bank."""
    bank = bank_document(_NAME, _layout(_SPLIT), _written(paths, _SPLIT), paths)
    bands = [layer.select["velocity"] for layer in bank.layers]
    for velocity in range(MIDI_MAX_VELOCITY + 1):
        assert len([band for band in bands if band.low <= velocity <= band.high]) == 1


def test_a_bank_of_one_band_answers_every_dynamic(paths: PlanPaths) -> None:
    (layer,) = bank_document(_NAME, _layout(UNSPLIT), _written(paths, UNSPLIT), paths).layers
    assert (layer.select["velocity"].low, layer.select["velocity"].high) == (0, MIDI_MAX_VELOCITY)


def test_a_band_written_as_one_instrument_leaves_its_dynamics_naming_it(paths: PlanPaths) -> None:
    """A file answering every key its band plays needs no pitch stated, which is what omitting it says."""
    bank = bank_document(_NAME, _layout(_SPLIT), _written(paths, _SPLIT), paths)
    assert all(set(layer.select) == {"velocity"} for layer in bank.layers)


def test_a_band_the_format_cut_states_the_keys_each_of_its_instruments_owns(paths: PlanPaths) -> None:
    """Both files answer every key they were filled over, so the manifest is what tells them apart."""
    layout = _cut_layout()
    bank = bank_document(_NAME, layout, _cut_written(paths, layout), paths)
    assert [(layer.select["pitch"].low, layer.select["pitch"].high) for layer in bank.layers] == [(0, 71), (72, 127)]
    assert all(layer.select["velocity"].low == 0 for layer in bank.layers)


def test_the_keys_a_cut_band_states_tile_the_whole_keyboard(paths: PlanPaths) -> None:
    """A note outside the stored runs reaches the instrument filled towards it rather than falling silent."""
    layout = _cut_layout()
    bank = bank_document(_NAME, layout, _cut_written(paths, layout), paths)
    bands = [layer.select["pitch"] for layer in bank.layers]
    for pitch in range(MIDI_LOWEST_PITCH, MIDI_HIGHEST_PITCH + 1):
        assert len([band for band in bands if band.low <= pitch <= band.high]) == 1


def test_a_plan_of_one_instrument_per_band_is_written_as_a_bank(paths: PlanPaths) -> None:
    paths.directory.mkdir(parents=True)
    write_bank(_NAME, _layout(_SPLIT), _written(paths, _SPLIT), paths)
    assert json.loads(paths.bank_json.read_text())["name"] == _NAME


def test_a_plan_whose_keys_tell_its_instruments_apart_is_written_as_a_bank_too(paths: PlanPaths) -> None:
    """Every plan is reachable through one manifest, whichever axes its instruments were split along."""
    paths.directory.mkdir(parents=True)
    layout = _cut_layout()
    write_bank(_NAME, layout, _cut_written(paths, layout), paths)
    written = json.loads(paths.bank_json.read_text())
    assert [layer["select"]["pitch"] for layer in written["layers"]] == [
        {"low": 0, "high": 71},
        {"low": 72, "high": 127},
    ]
