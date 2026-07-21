"""Impulse Tracker format primitives: the fixed byte values of the ``.IT`` layout plus the small
guards and text encoder shared by every record serializer.

These are the spec-defined constants (magic tags, header flags, pattern-cell masks, volume ceilings)
from ITTECH.TXT. They carry no optimizer knowledge -- the record serializers in
:mod:`~optisample.io.it_writer.samples`, :mod:`~optisample.io.it_writer.instruments`,
:mod:`~optisample.io.it_writer.patterns` and :mod:`~optisample.io.it_writer.module` supply values
against them.
"""

from __future__ import annotations

from typing import Final

from optisample.dsp.surrogate import MAX_VOLUME
from optisample.io.it_format import MAX_IT_NOTE

_IMPM: Final = b"IMPM"
_IMPS: Final = b"IMPS"
_IMPI: Final = b"IMPI"

NAME_BYTES: Final = 26  # every IT name field (song, instrument, sample) is 26 ASCII bytes, null-padded.
IT_C5_NOTE: Final = 60  # MIDI note 60 = IT's C-5 reference key (natural playback rate / pitch-pan centre).

MAX_ROWS: Final = 200  # IT patterns hold 1..200 rows.
TICKS_PER_ROW_BASE: Final = 2.5  # one tick lasts 2.5 / tempo seconds; a row lasts `speed` ticks.

NOTE_CUT: Final = 254  # pattern note value that cuts the playing note instantly.

_CWT: Final = 0x0214  # "created with" IT 2.14 -> selects the 554-byte instrument + envelope format.
_CMWT: Final = 0x0214  # "compatible with"; >= 0x0200 is required for the instrument format.
_FLAG_USE_INSTRUMENTS: Final = 0x04
_FLAG_LINEAR_SLIDES: Final = 0x08
_PPC_C5: Final = IT_C5_NOTE  # pitch-pan centre at C-5.

_SMP_FLAG_DATA: Final = 0x01  # sample data present.
_SMP_FLAG_16BIT: Final = 0x02  # 16-bit (else 8-bit).
_SMP_FLAG_LOOP: Final = 0x10  # forward loop enabled (loop begin/end fields are read).
_CVT_SIGNED: Final = 0x01  # signed PCM (the standard IT storage).

MAX_GLOBAL_VOLUME: Final = 128  # file-header and instrument global-volume ceiling.
MAX_MIX_VOLUME: Final = 128  # file-header mix-volume ceiling.
PANNING_SEPARATION: Final = 128  # full stereo separation in the file header.
PAN_CENTER: Final = 32  # centred channel pan (IT pan spans 0..64).
CHANNEL_VOLUME_FULL: Final = MAX_VOLUME  # every stored channel plays at full volume (0..64).
NOTE_ACTION_CUT: Final = 0  # new-note action: cut the previous note (IT: 0=cut, 1=continue, 2=off, 3=fade).

_MASK_NOTE: Final = 0x01
_MASK_INSTRUMENT: Final = 0x02
_MASK_VOLUME: Final = 0x04
_MASK_EFFECT: Final = 0x08

CHANNEL_MARKER: Final = 0x80  # high bit set on a packed cell's channel byte (always followed by a mask).
END_OF_ROW: Final = 0x00  # a zero byte terminates a packed pattern row.
ORDER_TERMINATOR: Final = 0xFF  # ends the order list.
OFFSET_TABLE_ENTRY_BYTES: Final = 4  # each instrument/sample/pattern offset is a little-endian u32.

_INT16_SCALE: Final = 32768.0
_INT8_SCALE: Final = 128.0


def require_it_note(note: int) -> int:
    """Return ``note`` if it is a playable IT key (0..119); raise ``ValueError`` otherwise."""
    if not 0 <= note <= MAX_IT_NOTE:
        raise ValueError(f"note {note} is outside the IT key range 0..{MAX_IT_NOTE}")
    return note


def _ascii(text: str, length: int) -> bytes:
    """Encode ``text`` to exactly ``length`` bytes, ASCII, null-padded (over-long names truncated)."""
    raw = text.encode("ascii", errors="replace")[:length]
    return raw + bytes(length - len(raw))
