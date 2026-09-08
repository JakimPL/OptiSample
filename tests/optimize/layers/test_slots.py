from trackmod import BitDepth

from optisample.dsp.surrogate.params import EncodingParams
from optisample.keys import SampleKey
from optisample.music import MIDI_HIGHEST_PITCH, MIDI_LOWEST_PITCH, MIDI_MAX_VELOCITY
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import pack_slots, reserved_slots
from optisample.optimize.plans import SampleUnit

_SPLIT = VelocityLayers((VelocityBand(0, 50), VelocityBand(51, MIDI_MAX_VELOCITY)))
_VELOCITY = 100
_WHOLE_TABLE = 255  # samples an instrument reaches in a format numbering them freely (Impulse Tracker)
_PER_INSTRUMENT = 16  # samples one FastTracker 2 instrument owns
_FIRST_LAYER = 0
_SECOND_LAYER = 1


def _unit(layer: int, keys: tuple[int, ...], *, stored_bytes: int, share: float, weight: float) -> SampleUnit:
    return SampleUnit(
        label=f"layer{layer}_rep{keys[0]:03d}",
        representative_key=SampleKey(keys[0], _VELOCITY),
        layer=layer,
        keys=keys,
        params=EncodingParams(target_rate=22_050, depth=BitDepth.SIXTEEN),
        frames=2400,
        stored_bytes=stored_bytes,
        distortion=share,
        objective_share=share,
        hull_size=1,
        weight=weight,
    )


def _key(pitch: int, layer: int = _FIRST_LAYER) -> SampleUnit:
    """One stored sample answering a single key, which is the widest a band can come out."""
    return _unit(layer, (pitch,), stored_bytes=1000, share=1.0, weight=1.0)


_QUIET = _unit(_FIRST_LAYER, (60, 61), stored_bytes=4000, share=1.5, weight=3.0)
_LOUD = _unit(_SECOND_LAYER, (60,), stored_bytes=6000, share=0.5, weight=2.0)
_ALSO_LOUD = _unit(_SECOND_LAYER, (61,), stored_bytes=5000, share=0.25, weight=1.0)
_LAYERED = (_QUIET, _LOUD, _ALSO_LOUD)


def test_a_format_reaching_the_whole_table_writes_one_instrument_per_band() -> None:
    layout = pack_slots(_LAYERED, _SPLIT, _WHOLE_TABLE)
    assert [slot.layer for slot in layout.slots] == [_FIRST_LAYER, _SECOND_LAYER]
    assert [slot.band.label for slot in layout.slots] == ["v000-v050", "v051-v127"]


def test_an_instrument_sums_only_the_samples_written_into_it() -> None:
    quiet, loud = pack_slots(_LAYERED, _SPLIT, _WHOLE_TABLE).slots
    assert (len(quiet.samples), quiet.keys, quiet.stored_bytes) == (1, 2, 4000)
    assert (len(loud.samples), loud.keys, loud.stored_bytes) == (2, 2, 11_000)


def test_the_written_instruments_add_up_to_what_the_whole_plan_stores() -> None:
    """A reader checks a split against the plan it came out of, so the instruments account for all of it."""
    slots = pack_slots(_LAYERED, _SPLIT, _WHOLE_TABLE).slots
    assert sum(slot.stored_bytes for slot in slots) == sum(unit.stored_bytes for unit in _LAYERED)
    assert sum(slot.weight for slot in slots) == sum(unit.weight for unit in _LAYERED)
    assert sum(slot.objective_share for slot in slots) == sum(unit.objective_share for unit in _LAYERED)
    assert sum(len(slot.samples) for slot in slots) == len(_LAYERED)


def test_a_band_storing_nothing_still_keeps_the_instrument_its_dynamics_resolve_to() -> None:
    quiet, loud = pack_slots((_LOUD, _ALSO_LOUD), _SPLIT, _WHOLE_TABLE).slots
    assert (len(quiet.samples), quiet.keys, quiet.stored_bytes) == (0, 0, 0)
    assert (quiet.weight, quiet.objective_share, quiet.span) == (0.0, 0.0, "")
    assert quiet.band.label == "v000-v050"
    assert len(loud.samples) == 2


def test_every_written_instrument_is_named_by_the_keys_and_the_dynamics_it_answers() -> None:
    """A file per instrument needs a name per instrument, so both axes a plan splits are in the label."""
    quiet, loud = pack_slots(_LAYERED, _SPLIT, _WHOLE_TABLE).slots
    assert quiet.file_label == "p060-p061_v000-v050"
    assert loud.file_label == "p060-p061_v051-v127"


def test_the_instruments_a_cut_band_is_written_as_are_named_apart() -> None:
    slots = pack_slots(tuple(_key(pitch) for pitch in range(60, 100)), UNSPLIT, _PER_INSTRUMENT).slots
    labels = [slot.file_label for slot in slots]
    assert labels == ["p060-p075_v000-v127", "p076-p091_v000-v127", "p092-p099_v000-v127"]
    assert len(set(labels)) == len(labels)  # each instrument lands in a file of its own


def test_a_band_storing_nothing_is_named_by_the_band_alone() -> None:
    """That band is written as one instrument, so its own label is all the name the file needs."""
    quiet, _ = pack_slots((_LOUD, _ALSO_LOUD), _SPLIT, _WHOLE_TABLE).slots
    assert quiet.file_label == "v000-v050"


def test_an_unlayered_plan_is_written_as_one_instrument_holding_everything() -> None:
    (whole,) = pack_slots((_unit(0, (60, 61, 62), stored_bytes=9000, share=2.0, weight=5.0),), UNSPLIT, 255).slots
    assert whole.band.label == "v000-v127"
    assert (len(whole.samples), whole.keys, whole.stored_bytes) == (1, 3, 9000)
    assert whole.span == "C4-D4"


def test_a_slot_names_the_position_each_sample_takes_in_the_song_table() -> None:
    """A keymap routes keys by position in the plan's single sample list, which the slot carries."""
    quiet, loud = pack_slots(_LAYERED, _SPLIT, _WHOLE_TABLE).slots
    assert quiet.samples == (0,)
    assert loud.samples == (1, 2)
    assert loud.units == (_LOUD, _ALSO_LOUD)


def test_a_band_wider_than_the_format_allows_is_cut_into_runs_it_holds() -> None:
    units = tuple(_key(pitch) for pitch in range(60, 100))
    slots = pack_slots(units, UNSPLIT, _PER_INSTRUMENT).slots
    assert [len(slot.samples) for slot in slots] == [16, 16, 8]
    assert sum(len(slot.samples) for slot in slots) == len(units)


def test_every_instrument_of_a_cut_band_owns_its_own_ascending_run_of_keys() -> None:
    """Cutting in key order is what leaves each instrument a contiguous stretch to answer."""
    units = tuple(_key(pitch) for pitch in range(60, 100))
    slots = pack_slots(units, UNSPLIT, _PER_INSTRUMENT).slots
    assert [slot.pitches[0] for slot in slots] == [60, 76, 92]
    assert [slot.pitches[-1] for slot in slots] == [75, 91, 99]
    assert [slot.span for slot in slots] == ["C4-D#5", "E5-G6", "G#6-D#7"]


def test_the_keys_are_cut_in_pitch_order_however_the_plan_lists_them() -> None:
    units = (_key(80), _key(60), _key(70))
    (whole,) = pack_slots(units, UNSPLIT, _WHOLE_TABLE).slots
    assert whole.samples == (1, 2, 0)
    assert whole.pitches == (60, 70, 80)


def test_a_note_plays_through_the_instrument_owning_its_key() -> None:
    layout = pack_slots(tuple(_key(pitch) for pitch in range(60, 100)), UNSPLIT, _PER_INSTRUMENT)
    assert layout.instrument(_FIRST_LAYER, 60) == 0
    assert layout.instrument(_FIRST_LAYER, 75) == 0
    assert layout.instrument(_FIRST_LAYER, 76) == 1
    assert layout.instrument(_FIRST_LAYER, 99) == 2


def test_a_key_outside_every_stored_run_plays_through_the_instrument_filled_to_it() -> None:
    """The keymaps are filled outward, so a key below the lowest run and one above the highest both sound."""
    layout = pack_slots(tuple(_key(pitch) for pitch in range(60, 100)), UNSPLIT, _PER_INSTRUMENT)
    assert layout.instrument(_FIRST_LAYER, 12) == 0
    assert layout.instrument(_FIRST_LAYER, 120) == 2


def test_a_note_plays_through_its_own_band_before_its_own_keys() -> None:
    layout = pack_slots(_LAYERED, _SPLIT, _WHOLE_TABLE)
    assert layout.instrument(_FIRST_LAYER, 60) == 0
    assert layout.instrument(_SECOND_LAYER, 60) == 1
    assert layout.count == 2
    assert layout.layer_slots(_SECOND_LAYER) == (1,)


def test_the_instruments_a_cut_band_is_written_as_own_key_bands_that_tile_the_keyboard() -> None:
    """A bank picks a file by these bands, so every key the format numbers reaches exactly one of them."""
    layout = pack_slots(tuple(_key(pitch) for pitch in range(60, 100)), UNSPLIT, _PER_INSTRUMENT)
    bands = [layout.key_band(index) for index in range(layout.count)]
    assert [(band.lowest, band.highest) for band in bands] == [(0, 75), (76, 91), (92, 127)]
    for pitch in range(MIDI_LOWEST_PITCH, MIDI_HIGHEST_PITCH + 1):
        assert len([band for band in bands if band.covers(pitch)]) == 1


def test_a_band_the_format_writes_whole_owns_the_keyboard_its_dynamics_answer_for() -> None:
    """One instrument per band means the dynamics name it on their own, which the whole-axis band states."""
    layout = pack_slots(_LAYERED, _SPLIT, _WHOLE_TABLE)
    assert [(layout.key_band(index).lowest, layout.key_band(index).highest) for index in range(2)] == [
        (MIDI_LOWEST_PITCH, MIDI_HIGHEST_PITCH),
        (MIDI_LOWEST_PITCH, MIDI_HIGHEST_PITCH),
    ]


def test_a_band_storing_nothing_owns_the_keyboard_its_dynamics_answer_for() -> None:
    """That band is written as its one instrument, so no key of it is left for another to claim."""
    layout = pack_slots((_LOUD, _ALSO_LOUD), _SPLIT, _WHOLE_TABLE)
    assert (layout.key_band(0).lowest, layout.key_band(0).highest) == (MIDI_LOWEST_PITCH, MIDI_HIGHEST_PITCH)


def test_each_band_cuts_its_own_keys_apart_from_the_others() -> None:
    """Two bands split by the format each tile the keyboard, since a note picks its dynamics first."""
    units = tuple(_key(pitch, layer) for layer in (_FIRST_LAYER, _SECOND_LAYER) for pitch in range(60, 92))
    layout = pack_slots(units, _SPLIT, _PER_INSTRUMENT)
    quiet = [layout.key_band(index) for index in layout.layer_slots(_FIRST_LAYER)]
    loud = [layout.key_band(index) for index in layout.layer_slots(_SECOND_LAYER)]
    assert [(band.lowest, band.highest) for band in quiet] == [(0, 75), (76, 127)]
    assert [(band.lowest, band.highest) for band in loud] == [(0, 75), (76, 127)]


def test_a_reserve_cuts_each_band_into_the_runs_the_format_writes() -> None:
    assert reserved_slots((40, 8), _PER_INSTRUMENT) == 3 + 1


def test_a_band_reserves_its_instrument_however_little_it_stores() -> None:
    assert reserved_slots((0, 1), _PER_INSTRUMENT) == 2


def test_a_format_reaching_the_whole_table_reserves_one_instrument_per_band() -> None:
    assert reserved_slots((61, 61, 61), _WHOLE_TABLE) == 3
