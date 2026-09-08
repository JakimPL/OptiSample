import numpy as np
import pytest
from trackmod import Instrument, InstrumentVoices, Sample, Song, flattened
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback

from optisample.io.tracker.voices import instrument_voices, routed_voices

_KEY = Note(60)
_ROWS = 32
_CHANNELS = 1


def _sample() -> Sample:
    return Sample(name="s", pcm=np.zeros(64, dtype=np.float64), rate=22_050)


def _song(voices: InstrumentVoices | object) -> Song:
    builder = PatternBuilder(rows=_ROWS, channels=_CHANNELS)
    return Song(
        name="probe",
        channels=_CHANNELS,
        patterns=(builder.build(),),
        order=OrderList.sequential(1),
        voices=voices,
        playback=Playback(speed=6, tempo=125),
    )


@pytest.fixture
def voices() -> InstrumentVoices:
    keymap = routed_keymap({_KEY: KeyAssignment(sample=0, note=_KEY)})
    return instrument_voices((Instrument(name="probe", keymap=keymap),), (_sample(),))


def test_the_table_holds_the_instruments_and_the_samples_their_keys_reach(voices: InstrumentVoices) -> None:
    assert len(voices.instruments) == 1
    assert voices.samples == (_sample(),)
    assert voices.slots == 1  # the instrument column names instruments, so a slot is an instrument


def test_a_song_built_here_is_read_back_as_the_routed_kind(voices: InstrumentVoices) -> None:
    assert routed_voices(_song(voices)) is voices


def test_a_song_naming_its_samples_directly_holds_no_instruments_to_read(voices: InstrumentVoices) -> None:
    """A sample table is what the Amiga formats store, and reading instruments off one says so outright."""
    with pytest.raises(TypeError, match="routes no keys through instruments"):
        routed_voices(_song(flattened(voices)))
