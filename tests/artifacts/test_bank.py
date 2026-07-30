from collections.abc import Sequence

from optisample.artifacts.bank import MANIFEST_VERSION, BankDocument, bank_document
from optisample.artifacts.serialize import VelocityMapDocument
from optisample.dsp.surrogate.params import EncodingParams
from optisample.keys import SampleKey
from optisample.music import MIDI_HIGHEST_PITCH, MIDI_LOWEST_PITCH, MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout, pack_slots
from optisample.optimize.plans import SampleUnit

_SPLIT = VelocityLayers((VelocityBand(0, 50), VelocityBand(51, MIDI_MAX_VELOCITY)))
_EXTENSION = ".iti"
_NAME = "piano"
_PER_INSTRUMENT = 12  # samples one instrument owns in a format numbering few of them
_CUT_KEYS = range(60, 84)
_VOLUMES = [velocity // 2 for velocity in range(MIDI_MAX_VELOCITY + 1)]


def _velocity_map() -> VelocityMapDocument:
    """The table the whole plan measured, which every layer carries beside the samples it was measured on."""
    return VelocityMapDocument(reference_volume=max(_VOLUMES), anchors=[], volumes=_VOLUMES)


def _slot(layer: int, band: VelocityBand) -> InstrumentSlot:
    """One written instrument, which the bank reads for the band it answers."""
    return InstrumentSlot(layer=layer, band=band, samples=(), units=())


def _layout(layers: VelocityLayers) -> SlotLayout:
    """One instrument per band, which is what a format reaching the whole sample table writes."""
    return SlotLayout(layers=layers, slots=tuple(_slot(layer, band) for layer, band in enumerate(layers.bands)))


def _entries(layout: SlotLayout) -> tuple[str, ...]:
    """The names the bank stores its instruments under, in the order the module numbers them."""
    return tuple(f"instruments/{slot.file_label}{_EXTENSION}" for slot in layout.slots)


def _bank(layout: SlotLayout, entries: Sequence[str]) -> BankDocument:
    return bank_document(_NAME, layout, entries, _velocity_map())


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


def test_a_bank_states_one_layer_for_every_band_the_plan_split() -> None:
    """A note picks its dynamics by picking a layer, so each band the allocation chose is a layer of its own."""
    layout = _layout(_SPLIT)
    bank = _bank(layout, _entries(layout))
    assert bank.name == _NAME
    assert [layer.select["velocity"].low for layer in bank.layers] == [0, 51]
    assert [layer.select["velocity"].high for layer in bank.layers] == [50, MIDI_MAX_VELOCITY]


def test_a_bank_states_the_version_a_consumer_reads() -> None:
    layout = _layout(UNSPLIT)
    assert _bank(layout, _entries(layout)).version == MANIFEST_VERSION


def test_a_layer_names_the_entry_of_the_bank_its_instrument_is_stored_as() -> None:
    """An entry is named against the bank, so the same manifest describes it archived and spread out."""
    layout = _layout(UNSPLIT)
    (layer,) = _bank(layout, _entries(layout)).layers
    assert layer.source.file == "instruments/v000-v127.iti"


def test_a_layer_carries_the_map_its_dynamics_were_measured_with() -> None:
    """The map travels inside the bank, so a layer plays the dynamics its own waveforms were stored for."""
    layout = _layout(UNSPLIT)
    (layer,) = _bank(layout, _entries(layout)).layers
    assert layer.velocity_map.volumes == _VOLUMES
    assert layer.velocity_map.reference_volume == max(_VOLUMES)


def test_every_dynamic_reaches_exactly_one_layer() -> None:
    """The plan's bands tile the velocity axis, and stating each one keeps that true of the bank."""
    layout = _layout(_SPLIT)
    bands = [layer.select["velocity"] for layer in _bank(layout, _entries(layout)).layers]
    for velocity in range(MIDI_MAX_VELOCITY + 1):
        assert len([band for band in bands if band.low <= velocity <= band.high]) == 1


def test_a_bank_of_one_band_answers_every_dynamic() -> None:
    layout = _layout(UNSPLIT)
    (layer,) = _bank(layout, _entries(layout)).layers
    assert (layer.select["velocity"].low, layer.select["velocity"].high) == (0, MIDI_MAX_VELOCITY)


def test_a_band_written_as_one_instrument_leaves_its_dynamics_naming_it() -> None:
    """A file answering every key its band plays needs no pitch stated, which is what omitting it says."""
    layout = _layout(_SPLIT)
    assert all(set(layer.select) == {"velocity"} for layer in _bank(layout, _entries(layout)).layers)


def test_a_band_the_format_cut_states_the_keys_each_of_its_instruments_owns() -> None:
    """Both files answer every key they were filled over, so the manifest is what tells them apart."""
    layout = _cut_layout()
    bank = _bank(layout, _entries(layout))
    assert [(layer.select["pitch"].low, layer.select["pitch"].high) for layer in bank.layers] == [(0, 71), (72, 127)]
    assert all(layer.select["velocity"].low == 0 for layer in bank.layers)


def test_the_keys_a_cut_band_states_tile_the_whole_keyboard() -> None:
    """A note outside the stored runs reaches the instrument filled towards it rather than falling silent."""
    layout = _cut_layout()
    bands = [layer.select["pitch"] for layer in _bank(layout, _entries(layout)).layers]
    for pitch in range(MIDI_LOWEST_PITCH, MIDI_HIGHEST_PITCH + 1):
        assert len([band for band in bands if band.low <= pitch <= band.high]) == 1


def test_a_layer_stands_for_every_instrument_the_plan_was_written_as() -> None:
    """Every plan is reachable through one manifest, whichever axes its instruments were split along."""
    layout = _cut_layout()
    entries = _entries(layout)
    assert [layer.source.file for layer in _bank(layout, entries).layers] == list(entries)
