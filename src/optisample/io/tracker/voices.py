from collections.abc import Sequence

from trackmod import Instrument, InstrumentVoices, Sample, SampleVoices, Song


def instrument_voices(instruments: Sequence[Instrument], samples: Sequence[Sample]) -> InstrumentVoices:
    """The voice table a song states ``instruments`` and the ``samples`` their keymaps reach through.

    A tracker's instrument column names either a waveform outright or an instrument routing keys onto
    waveforms, and a format stores one kind or the other. Every song written here carries a volume
    envelope and a per-key routing, so it is the second kind that is built, and the pair is stated
    together because a keymap names positions in the very sample table beside it.
    """
    return InstrumentVoices(instruments=tuple(instruments), samples=tuple(samples))


def routed_voices(song: Song) -> InstrumentVoices:
    """``song``'s voice table, read as the routed kind every song written here carries.

    The table is what a cell's instrument column names, and it comes in two kinds, so reading the
    instruments off a song asks which kind it holds. A song built by this package always answers with
    the routed kind (:func:`instrument_voices`), which is also what FastTracker 2 requires and what
    Impulse Tracker is written as, so this states that invariant once for every reader.

    Raises:
        TypeError: when the song names its waveforms directly, holding no instruments to read.
    """
    match song.voices:
        case InstrumentVoices() as voices:
            return voices
        case SampleVoices():
            raise TypeError(f"song {song.name!r} names its samples directly and routes no keys through instruments")
