"""Pitch and tuning primitives shared across the package.

Note naming, the equal-tempered semitone ratio, and the MIDI tuning reference live here so every
module that reasons about pitch draws on one definition. This is a leaf module -- it depends on
nothing else in the package, so anything may import from it without risking a cycle.
"""

from __future__ import annotations

SEMITONES_PER_OCTAVE = 12  # equal temperament: an octave is 12 semitones and doubles the frequency.
MIDI_A4 = 69  # MIDI note number of A4, the tuning reference.
A4_FREQ_HZ = 440.0  # frequency of A4 in hertz.

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(pitch: int) -> str:
    """MIDI note number -> scientific pitch name (60 -> ``C4``)."""
    return f"{NOTE_NAMES[pitch % SEMITONES_PER_OCTAVE]}{pitch // SEMITONES_PER_OCTAVE - 1}"


def semitone_ratio(semitones: float) -> float:
    """Playback speed / frequency ratio for a pitch shift of ``semitones`` (12 semitones = 2x)."""
    return float(2.0 ** (semitones / SEMITONES_PER_OCTAVE))
