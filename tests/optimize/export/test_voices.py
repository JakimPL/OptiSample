from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from optisample.dsp.loop import Loop
from optisample.dsp.series import Series
from optisample.dsp.surrogate import EncodingParams, StoredSample
from optisample.keys import SampleKey
from optisample.model import NoteEvent
from optisample.optimize.export.voices import (
    NO_DRIFT,
    NO_SHAPE,
    PlayedVoices,
    instrument_shape,
    slot_members,
    written_instruments,
)
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout
from optisample.optimize.plans import SampleUnit
from optisample.optimize.tasks import StoredRecordings
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from trackmod.spec.levels import MAX_VOLUME

_SR = 22_050
_ROOT = 60
_SERVED = (60, 62)
_VELOCITY = 100
_SPAN_S = 2.0
_HELD_S = 1.5  # how long the material holds a note, which is the stretch its envelope has to reach across
_LOOP_START_S = 0.4
_LOOP_END_S = 0.5
_FALL_DB_PER_S = -12.0  # the rate every reference recording declines at, which one shape has to follow
_TONE_HZ = 220.0
_NODES = 24
_MIDI_VELOCITIES = 128
_ANCHORS = (VelocityAnchor(velocity=_VELOCITY, loudness_lufs=-20.0, volume=MAX_VOLUME),)
_SILENT = 0
_LOUD_GAIN = MAX_VOLUME
_QUIET_GAIN = 32  # half the amplitude of the loudest step, which is 6 dB of the module's own staging
_A_SLIVER_S = 0.001  # a note shorter than the finest window a trajectory is ever read in


def _falling(span_s: float) -> Series:
    """The level a struck note holds moment by moment: a steady number of decibels a second."""
    moments = np.arange(round(span_s * _SR), dtype=np.float64) / _SR
    return np.asarray(10.0 ** (_FALL_DB_PER_S * moments / 20.0))


def _tone(levels: Series) -> Series:
    """A tone carrying ``levels``, which is the material a level reading is taken off."""
    moments = np.arange(levels.size, dtype=np.float64) / _SR
    return np.asarray(np.sin(2.0 * np.pi * _TONE_HZ * moments) * levels)


def _stored(*, looped: bool = True) -> StoredSample:
    """The waveform a zone stores: the opening of the recording, held at one level by a loop past its start."""
    levels = _falling(_LOOP_END_S)
    loop = Loop(start=round(_LOOP_START_S * _SR), end=round(_LOOP_END_S * _SR)) if looped else None
    if loop is not None:
        levels[loop.start :] = levels[loop.start]  # a settled loop region is leveled flat, which is what it holds

    return StoredSample(pcm=_tone(levels), sample_rate=_SR, depth_bits=16, root_pitch=_ROOT, loop=loop)


def _unit(keys: tuple[int, ...]) -> SampleUnit:
    return SampleUnit(
        label="zone",
        representative_key=SampleKey(_ROOT, _VELOCITY),
        layer=0,
        keys=keys,
        params=EncodingParams(target_rate=_SR, depth_bits=16),
        frames=round(_LOOP_END_S * _SR),
        stored_bytes=1000,
        distortion=1.0,
        objective_share=1.0,
        hull_size=1,
        weight=1.0,
    )


def _slot(keys: tuple[int, ...] = _SERVED, band: VelocityBand = UNSPLIT.bands[0]) -> InstrumentSlot:
    return InstrumentSlot(layer=0, band=band, samples=(0,), units=(_unit(keys),))


def _velocity_map(*, volume: int = MAX_VOLUME) -> VelocityVolumeMap:
    return VelocityVolumeMap(tuple(volume for _ in range(_MIDI_VELOCITIES)), _ANCHORS)


def _sources(
    *,
    material: Sequence[NoteEvent] | None = None,
    gains: tuple[int, ...] = (_LOUD_GAIN,),
    volume: int = MAX_VOLUME,
    keys: tuple[int, ...] = _SERVED,
) -> PlayedVoices:
    audio = {SampleKey(pitch, _VELOCITY): _tone(_falling(_SPAN_S)) for pitch in keys}
    played = (
        material
        if material is not None
        else [NoteEvent(pitch=pitch, velocity=_VELOCITY, duration_s=_HELD_S) for pitch in keys]
    )
    return PlayedVoices(
        recordings=StoredRecordings(audio=audio, settled={}, sample_rate=_SR),
        material=list(played),
        velocity_map=_velocity_map(volume=volume),
        gains=gains,
    )


def _shape(sources: PlayedVoices) -> tuple[float, float]:
    """The level the fitted shape reads at the loop start and at the end of the note, in decibels."""
    fitted = instrument_shape(_slot(), (_stored(),), sources, nodes=_NODES)
    assert fitted is not NO_SHAPE
    read = fitted.curve.at(np.asarray([_LOOP_START_S, _HELD_S], dtype=np.float64))
    return float(read[0]), float(read[1])


# --- which keys an instrument answers -----------------------------------------------------------------------


def test_one_voice_stands_for_every_key_the_material_plays_through_the_instrument() -> None:
    members = slot_members(_slot(), (_stored(),), _sources())

    assert len(members) == len(_SERVED)


def test_a_key_the_material_never_plays_asks_nothing_of_the_envelope() -> None:
    """A stored zone reaches keys the song leaves alone, and a shape answers only what is heard."""
    played = [NoteEvent(pitch=_SERVED[0], velocity=_VELOCITY, duration_s=_HELD_S)]

    members = slot_members(_slot(), (_stored(),), _sources(material=played))

    assert len(members) == 1


def test_a_key_the_pattern_writes_at_silence_asks_nothing_of_the_envelope() -> None:
    """A note volume of zero is never heard, so the trajectory it would state is left out of the fit."""
    assert slot_members(_slot(), (_stored(),), _sources(volume=_SILENT)) == ()


def test_an_instrument_the_material_plays_no_recorded_key_of_carries_no_shape() -> None:
    assert instrument_shape(_slot(), (_stored(),), _sources(material=[]), nodes=_NODES) is NO_SHAPE


def test_a_key_held_too_briefly_to_read_one_window_of_asks_nothing_of_the_envelope() -> None:
    """Every key of a set is read at the window its longest note settles, which a short one falls inside."""
    uneven = [
        NoteEvent(pitch=_SERVED[0], velocity=_VELOCITY, duration_s=_HELD_S),
        NoteEvent(pitch=_SERVED[1], velocity=_VELOCITY, duration_s=_A_SLIVER_S),
    ]

    members = slot_members(_slot(), (_stored(),), _sources(material=uneven))

    assert len(members) == 1


def test_a_voice_is_worth_the_playing_time_the_material_spends_on_its_key() -> None:
    """One shape answers loudest to the keys the song leans on, which is what the weight carries."""
    leaned_on = [
        NoteEvent(pitch=_SERVED[0], velocity=_VELOCITY, duration_s=_HELD_S, count=4),
        NoteEvent(pitch=_SERVED[1], velocity=_VELOCITY, duration_s=_HELD_S),
    ]

    members = slot_members(_slot(), (_stored(),), _sources(material=leaned_on))

    assert members[0].weight == pytest.approx(4.0 * members[1].weight)


# --- what the shape carries -----------------------------------------------------------------------------------


def test_the_shape_supplies_the_decline_the_loop_stopped_following() -> None:
    """A looped sample holds one level, so everything the recording did afterwards is the envelope's."""
    at_loop, at_end = _shape(_sources())

    assert at_end - at_loop == pytest.approx(_FALL_DB_PER_S * (_HELD_S - _LOOP_START_S), abs=1.0)


def test_the_shape_leaves_the_attack_to_the_waveform() -> None:
    """Up to the loop the stored waveform is the recording, so the envelope has nothing to add there."""
    fitted = instrument_shape(_slot(), (_stored(),), _sources(), nodes=_NODES)
    assert fitted is not NO_SHAPE

    opening = fitted.curve.at(np.asarray([0.05, _LOOP_START_S], dtype=np.float64))

    assert float(opening[1] - opening[0]) == pytest.approx(0.0, abs=1.0)


def test_the_shape_stands_at_the_level_the_module_already_writes_for_the_sample() -> None:
    """The 0-64 gain beside the envelope is spent restoring balance, so the shape carries what is left."""
    loud = _shape(_sources(gains=(_LOUD_GAIN,)))
    quiet = _shape(_sources(gains=(_QUIET_GAIN,)))

    assert quiet[0] - loud[0] == pytest.approx(6.02, abs=0.1)


def test_a_sample_stored_whole_is_measured_over_the_stretch_it_keeps_sounding() -> None:
    """A note stops where an unlooped waveform runs out, so no envelope is fitted past that moment."""
    members = slot_members(_slot(), (_stored(looped=False),), _sources())

    assert max(float(member.readings.seconds[-1]) for member in members) < _LOOP_END_S


# --- what the whole layout reports -------------------------------------------------------------------------


def test_every_written_instrument_reports_what_its_one_envelope_leaves_its_keys() -> None:
    layout = SlotLayout(layers=VelocityLayers(bands=UNSPLIT.bands), slots=(_slot(),))

    written = written_instruments(layout, (_stored(),), _sources(), nodes=_NODES)

    assert written.layout is layout
    assert len(written.drifts) == 1
    assert written.drifts[0] > NO_DRIFT


def test_an_instrument_carrying_no_shape_gives_up_nothing_for_sharing_one() -> None:
    layout = SlotLayout(layers=VelocityLayers(bands=UNSPLIT.bands), slots=(_slot(),))

    written = written_instruments(layout, (_stored(),), _sources(material=[]), nodes=_NODES)

    assert written.drifts == (NO_DRIFT,)
