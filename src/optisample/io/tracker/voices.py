from collections.abc import Sequence

from trackmod import Instrument, InstrumentVoices, Sample, SampleVoices, Song, Voices


def instrument_voices(instruments: Sequence[Instrument], samples: Sequence[Sample]) -> InstrumentVoices:
    """The voice table a song states ``instruments`` and the ``samples`` their keymaps reach through.

    A tracker's instrument column names either a waveform outright or an instrument routing keys onto
    waveforms, and a format stores one kind or the other. Every song written here carries a volume
    envelope and a per-key routing, so it is the second kind that is built, and the pair is stated
    together because a keymap names positions in the very sample table beside it.
    """
    return InstrumentVoices(instruments=tuple(instruments), samples=tuple(samples))


def routed(voices: Voices, *, held_by: str) -> InstrumentVoices:
    """``voices`` read as the routed kind every song written here carries.

    A voice table comes in two kinds, so reading instruments off one asks which kind it holds. Both the
    songs this package builds and the files it reads back carry the routed kind, so this states that
    invariant once for every reader. ``held_by`` names what the table was read from, which is what a
    refusal reports.

    Raises:
        TypeError: when the table names its waveforms directly, holding no instruments to read.
    """
    match voices:
        case InstrumentVoices() as routed_kind:
            return routed_kind
        case SampleVoices():
            raise TypeError(f"{held_by} names its samples directly and routes no keys through instruments")


def routed_voices(song: Song) -> InstrumentVoices:
    """``song``'s voice table, read as the routed kind every song written here carries.

    A song built by this package always answers with the routed kind (:func:`instrument_voices`), which
    is also what FastTracker 2 requires and what Impulse Tracker is written as.

    Raises:
        TypeError: when the song names its waveforms directly, holding no instruments to read.
    """
    return routed(song.voices, held_by=f"song {song.name!r}")
