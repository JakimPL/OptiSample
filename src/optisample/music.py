from typing import Final

from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import RATE_NOTE

SEMITONES_PER_OCTAVE: Final = 12
MIDI_A4: Final = 69
A4_FREQ_HZ: Final = 440.0
MIDI_MAX_VELOCITY: Final = 127

NOTE_NAMES: Final = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(pitch: int) -> str:
    """MIDI note number -> scientific pitch name (60 -> ``C4``)."""
    return f"{NOTE_NAMES[pitch % SEMITONES_PER_OCTAVE]}{pitch // SEMITONES_PER_OCTAVE - 1}"


def semitone_ratio(semitones: float) -> float:
    """Playback speed / frequency ratio for a pitch shift of ``semitones`` (12 semitones = 2x)."""
    return float(2.0 ** (semitones / SEMITONES_PER_OCTAVE))


def midi_to_freq(pitch: int) -> float:
    """MIDI note number -> fundamental frequency in Hz (A4/69 = 440 Hz)."""
    return A4_FREQ_HZ * semitone_ratio(pitch - MIDI_A4)


def sounded_note(key: Note, root_key: Note) -> Note:
    """The note ``key`` sounds so a sample recorded at ``root_key`` plays transposed to ``key``'s pitch.

    A tracker plays a stored sample at its recorded rate when the reference key sounds, so routing a key
    to that reference shifted by its distance from the recording's own pitch transposes the sample by
    exactly that interval, which leaves the sample's stored rate free to state the rate it truly holds.

    Raises:
        ValueError: when the transposition lands outside the tracker key range.
    """
    return Note(RATE_NOTE + key.value - root_key.value)
