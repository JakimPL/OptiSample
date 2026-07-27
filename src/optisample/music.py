from typing import Final

from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import RATE_NOTE

SEMITONES_PER_OCTAVE: Final = 12
MIDI_A4: Final = 69
A4_FREQ_HZ: Final = 440.0
MIDI_MAX_VELOCITY: Final = 127

NOTE_NAMES: Final = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_LABEL_PREFIX: Final = "p"
_LABEL_SEPARATOR: Final = "_"


def note_name(pitch: int) -> str:
    """MIDI note number -> scientific pitch name (60 -> ``C4``)."""
    return f"{NOTE_NAMES[pitch % SEMITONES_PER_OCTAVE]}{pitch // SEMITONES_PER_OCTAVE - 1}"


def pitch_label(pitch: int) -> str:
    """MIDI note number -> the stable name every per-pitch artifact is filed under (60 -> ``p060_C4``).

    The zero-padded number leads so a directory listing reads in pitch order, and the note name follows
    so the same listing is readable as music. One spelling, so a key's label, its A/B pair and its
    audition folder all name the same pitch the same way.
    """
    return f"{_LABEL_PREFIX}{pitch:03d}{_LABEL_SEPARATOR}{note_name(pitch)}"


def labelled_pitch(label: str) -> int:
    """The MIDI note number :func:`pitch_label` filed an artifact under (``p060_C4`` -> 60).

    Reads the number back off the one spelling that wrote it, so a reader walking a directory recovers
    the key each artifact belongs to.

    Raises:
        ValueError: when ``label`` opens with something other than a zero-padded pitch number.
    """
    return int(label.split(_LABEL_SEPARATOR)[0].removeprefix(_LABEL_PREFIX))


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
