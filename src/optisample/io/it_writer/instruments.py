"""The instrument record: its data shape, the keyboard note map, and the 554-byte IMPI header."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from optisample.io.it_format import INSTRUMENT_HEADER, KEYBOARD_NOTES
from optisample.io.it_writer.constants import (
    _IMPI,
    _NAME_BYTES,
    _PPC_C5,
    MAX_GLOBAL_VOLUME,
    _ascii,
    require_it_note,
)


@dataclass(frozen=True)
class ITInstrument:
    """One instrument: a 120-entry ``(play_note, sample_number)`` keyboard map plus playback defaults."""

    name: str
    note_map: tuple[tuple[int, int], ...]
    global_volume: int = 128
    default_pan: int = 32
    new_note_action: int = 0  # 0=cut, 1=continue, 2=off, 3=fade.


def identity_note_map(assignments: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """Build a 120-entry note map where key ``n`` plays note ``n`` on ``assignments.get(n, 0)``.

    Sample numbers are 1-based (``0`` = no sample). With one dedicated sample per key this is an
    identity mapping; the per-sample ``C5Speed`` carries the real pitch, so each key plays naturally.
    """
    for note, sample_number in assignments.items():
        require_it_note(note)
        if sample_number < 0:
            raise ValueError(f"sample number {sample_number} must be non-negative")
    return tuple((note, assignments.get(note, 0)) for note in range(KEYBOARD_NOTES))


def _instrument_header(instrument: ITInstrument) -> bytes:
    """Serialize a 554-byte IMPI instrument header with disabled envelopes."""
    if len(instrument.note_map) != KEYBOARD_NOTES:
        raise ValueError(f"note map must have {KEYBOARD_NOTES} entries, got {len(instrument.note_map)}")
    # Envelopes (offsets 304..550) and the 4 trailing reserved bytes stay zero = disabled.
    return INSTRUMENT_HEADER.pack(
        {
            "magic": _IMPI,
            "new_note_action": instrument.new_note_action & 0xFF,
            "pitch_pan_center": _PPC_C5,
            "global_volume": min(instrument.global_volume, MAX_GLOBAL_VOLUME),
            "default_pan": instrument.default_pan & 0xFF,
            "name": _ascii(instrument.name, _NAME_BYTES),
            "note_map": tuple(
                (play_note & 0xFF, sample_number & 0xFF) for play_note, sample_number in instrument.note_map
            ),
        }
    )
