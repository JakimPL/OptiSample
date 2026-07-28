import json
from pathlib import Path

import pytest

from optisample.artifacts.bank import MANIFEST_VERSION, bank_document, write_bank
from optisample.artifacts.paths import PlanPaths
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout

_SPLIT = VelocityLayers((VelocityBand(0, 50), VelocityBand(51, MIDI_MAX_VELOCITY)))
_EXTENSION = ".iti"
_NAME = "piano"


def _slot(layer: int, band: VelocityBand) -> InstrumentSlot:
    """One written instrument, which the bank reads for the band it answers."""
    return InstrumentSlot(layer=layer, band=band, samples=(), units=())


def _layout(layers: VelocityLayers) -> SlotLayout:
    """One instrument per band, which is what a format reaching the whole sample table writes."""
    return SlotLayout(layers=layers, slots=tuple(_slot(layer, band) for layer, band in enumerate(layers.bands)))


def _written(paths: PlanPaths, layers: VelocityLayers) -> tuple[Path, ...]:
    return tuple(paths.instrument_file(band.label, _EXTENSION) for band in layers.bands)


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


def test_a_plan_of_one_instrument_per_band_is_written_as_a_bank(paths: PlanPaths) -> None:
    paths.directory.mkdir(parents=True)
    write_bank(_NAME, _layout(_SPLIT), _written(paths, _SPLIT), paths)
    assert json.loads(paths.bank_json.read_text())["name"] == _NAME
    assert not paths.unbanked.exists()


def test_a_plan_whose_keys_tell_its_instruments_apart_says_so_beside_them(paths: PlanPaths) -> None:
    """A reader of that tree loads the files one at a time, so the note says how many there are."""
    paths.directory.mkdir(parents=True)
    quiet, loud = _SPLIT.bands
    layout = SlotLayout(layers=_SPLIT, slots=(_slot(0, quiet), _slot(1, loud), _slot(1, loud)))
    write_bank(_NAME, layout, _written(paths, _SPLIT), paths)
    note = paths.unbanked.read_text()
    assert "3 instruments written where the velocity split states 2" in note
    assert "instruments/" in note
    assert not paths.bank_json.exists()
