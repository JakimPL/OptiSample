import re
from typing import Final

from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import NOTES_PER_OCTAVE, RATE_NOTE

MIDI_A4: Final = 69
A4_FREQ_HZ: Final = 440.0
MIDI_MAX_VELOCITY: Final = 127
MIDI_LOWEST_PITCH: Final = 0
MIDI_HIGHEST_PITCH: Final = 127

NOTE_NAMES: Final = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_LABEL_PREFIX: Final = "p"
_LABEL_SEPARATOR: Final = "_"

_LOWEST_OCTAVE: Final = -1  # MIDI 0 is C-1, so an octave number counts from one below the lowest C
_ACCIDENTAL_STEPS: Final = {"": 0, "#": 1, "b": -1}
_NOTE_NAME_PATTERN: Final = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def note_name(pitch: int) -> str:
    """MIDI note number -> scientific pitch name (60 -> ``C4``)."""
    return f"{NOTE_NAMES[pitch % NOTES_PER_OCTAVE]}{pitch // NOTES_PER_OCTAVE - 1}"


def named_pitch(name: str) -> int:
    """The MIDI note number a scientific pitch name spells (``C4`` -> 60, ``Bb2`` -> 46).

    Both spellings of a black key are read, so a set of recordings named with flats resolves the same
    keys a set named with sharps does, and the letter is read in either case so a name is recognized
    however its writer capitalized it.

    Raises:
        ValueError: when ``name`` spells no pitch name, or spells one outside the MIDI range.
    """
    match = _NOTE_NAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"{name!r} spells no scientific pitch name")

    letter, accidental, octave = match.groups()
    semitone = NOTE_NAMES.index(letter.upper()) + _ACCIDENTAL_STEPS[accidental]
    pitch = semitone + (int(octave) - _LOWEST_OCTAVE) * NOTES_PER_OCTAVE
    if not MIDI_LOWEST_PITCH <= pitch <= MIDI_HIGHEST_PITCH:
        raise ValueError(f"pitch name {name!r} lands outside the MIDI range")

    return pitch


def pitch_label(pitch: int) -> str:
    """MIDI note number -> the stable name every per-pitch artifact is filed under (60 -> ``p060_C4``).

    The zero-padded number leads so a directory listing reads in pitch order, and the note name follows
    so the same listing is readable as music. One spelling, so a key's label, its A/B pair and its
    audition folder all name the same pitch the same way.
    """
    return f"{_LABEL_PREFIX}{pitch:03d}{_LABEL_SEPARATOR}{note_name(pitch)}"


def labeled_pitch(label: str) -> int:
    """The MIDI note number :func:`pitch_label` filed an artifact under (``p060_C4`` -> 60).

    Reads the number back off the one spelling that wrote it, so a reader walking a directory recovers
    the key each artifact belongs to.

    Raises:
        ValueError: when ``label`` opens with something other than a zero-padded pitch number.
    """
    return int(label.split(_LABEL_SEPARATOR)[0].removeprefix(_LABEL_PREFIX))


def semitone_ratio(semitones: float) -> float:
    """Playback speed / frequency ratio for a pitch shift of ``semitones`` (12 semitones = 2x)."""
    return float(2.0 ** (semitones / NOTES_PER_OCTAVE))


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
