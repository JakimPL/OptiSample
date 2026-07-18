"""Hand-rolled uncompressed Impulse Tracker (``.IT``) writer.

No mature Python IT writer exists, so this module serializes the uncompressed IT format directly
from the ITTECH.TXT specification. It is deliberately *domain-agnostic*: it knows about samples,
instruments, patterns and byte layout, not about the optimizer -- the plan->module bridge lives in
:mod:`optisample.optimize.export`.

Layout produced (all multi-byte fields little-endian)::

    [192-B file header][orders + 0xFF][instr offsets][sample offsets][pattern offsets]
    [554-B instrument headers][80-B sample headers][packed patterns][sample PCM]

Only what the POC needs is emitted: mono signed 8/16-bit PCM, no loops, disabled envelopes, one
instrument with an identity note map (each key plays its own dedicated sample at natural rate via a
per-sample ``C5Speed``). IT214 compression is deferred (GPL-licensing-gated); the optimizer's byte
model is the uncompressed size regardless.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, SAMPLE_HEADER_BYTES

_IMPM = b"IMPM"
_IMPS = b"IMPS"
_IMPI = b"IMPI"

_KEYBOARD_NOTES = 120  # IT keys C-0..B-9 (0..119); the note map holds one (note, sample) pair each.
_CHANNELS_STORED = 64  # the file header always carries 64 channel pan + 64 channel volume bytes.
_MAX_ROWS = 200  # IT patterns hold 1..200 rows.
_MAX_IT_NOTE = _KEYBOARD_NOTES - 1

NOTE_OFF = 255  # pattern note value that releases the playing note.
NOTE_CUT = 254  # pattern note value that cuts it instantly.

_CWT = 0x0214  # "created with" IT 2.14 -> selects the 554-byte instrument + envelope format.
_CMWT = 0x0214  # "compatible with"; >= 0x0200 is required for the instrument format.
_FLAG_USE_INSTRUMENTS = 0x04
_FLAG_LINEAR_SLIDES = 0x08
_PPC_C5 = 60  # pitch-pan centre at C-5.

_SMP_FLAG_DATA = 0x01  # sample data present.
_SMP_FLAG_16BIT = 0x02  # 16-bit (else 8-bit).
_CVT_SIGNED = 0x01  # signed PCM (the standard IT storage).

_MASK_NOTE = 0x01
_MASK_INSTRUMENT = 0x02
_MASK_VOLUME = 0x04
_MASK_EFFECT = 0x08

_INT16_SCALE = 32768.0
_INT8_SCALE = 128.0


@dataclass(frozen=True)
class ITSample:
    """One stored sample: its PCM (float in ``[-1, 1]``), storage depth and playback rate."""

    name: str
    pcm: NDArray[np.floating]
    depth_bits: int = 16
    c5speed: int = 44_100
    global_volume: int = 64
    default_volume: int = 64

    @property
    def frames(self) -> int:
        return int(np.asarray(self.pcm).size)


@dataclass(frozen=True)
class ITInstrument:
    """One instrument: a 120-entry ``(play_note, sample_number)`` keyboard map plus playback defaults."""

    name: str
    note_map: tuple[tuple[int, int], ...]
    global_volume: int = 128
    default_pan: int = 32
    new_note_action: int = 0  # 0=cut, 1=continue, 2=off, 3=fade.


@dataclass(frozen=True)
class ITCell:
    """One pattern cell; ``None`` fields are left empty (not written) in the packed stream."""

    note: int | None = None
    instrument: int | None = None
    volume: int | None = None
    effect: tuple[int, int] | None = None


@dataclass(frozen=True)
class ITPattern:
    """A pattern: a row count and the sparse cells placed on it as ``(row, channel, cell)``."""

    rows: int
    cells: tuple[tuple[int, int, ITCell], ...] = ()


@dataclass(frozen=True)
class ITPlayback:
    """Global playback settings (bundled to keep :class:`ITModule` small)."""

    speed: int = 6
    tempo: int = 125
    global_volume: int = 128
    mix_volume: int = 48


@dataclass(frozen=True)
class ITModule:
    """A complete module ready to serialize."""

    name: str
    samples: tuple[ITSample, ...]
    instruments: tuple[ITInstrument, ...]
    patterns: tuple[ITPattern, ...]
    orders: tuple[int, ...]
    playback: ITPlayback = field(default_factory=ITPlayback)


def identity_note_map(assignments: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """Build a 120-entry note map where key ``n`` plays note ``n`` on ``assignments.get(n, 0)``.

    Sample numbers are 1-based (``0`` = no sample). With one dedicated sample per key this is an
    identity mapping; the per-sample ``C5Speed`` carries the real pitch, so each key plays naturally.
    """
    for note, sample_number in assignments.items():
        if not 0 <= note <= _MAX_IT_NOTE:
            raise ValueError(f"note {note} out of IT range 0..{_MAX_IT_NOTE}")
        if sample_number < 0:
            raise ValueError(f"sample number {sample_number} must be non-negative")
    return tuple((note, assignments.get(note, 0)) for note in range(_KEYBOARD_NOTES))


def _ascii(text: str, length: int) -> bytes:
    """Encode ``text`` to exactly ``length`` bytes, ASCII, null-padded (over-long names truncated)."""
    raw = text.encode("ascii", errors="replace")[:length]
    return raw + bytes(length - len(raw))


def _pcm_bytes(sample: ITSample) -> bytes:
    """Convert float PCM in ``[-1, 1]`` to signed little-endian bytes at the sample's depth."""
    pcm = np.asarray(sample.pcm, dtype=np.float64)
    if sample.depth_bits == 16:
        quantized = np.clip(np.round(pcm * _INT16_SCALE), -_INT16_SCALE, _INT16_SCALE - 1).astype("<i2")
    elif sample.depth_bits == 8:
        quantized = np.clip(np.round(pcm * _INT8_SCALE), -_INT8_SCALE, _INT8_SCALE - 1).astype("<i1")
    else:
        raise ValueError(f"unsupported depth {sample.depth_bits} (expected 8 or 16)")
    return quantized.tobytes()


def _sample_header(sample: ITSample, data_offset: int) -> bytes:
    """Serialize an 80-byte IMPS sample header pointing at ``data_offset``."""
    if sample.depth_bits not in (8, 16):
        raise ValueError(f"unsupported depth {sample.depth_bits} (expected 8 or 16)")
    buf = bytearray(SAMPLE_HEADER_BYTES)
    buf[0:4] = _IMPS
    buf[17] = min(sample.global_volume, 64)
    buf[18] = _SMP_FLAG_DATA | (_SMP_FLAG_16BIT if sample.depth_bits == 16 else 0)
    buf[19] = min(sample.default_volume, 64)
    buf[20:46] = _ascii(sample.name, 26)
    buf[46] = _CVT_SIGNED
    struct.pack_into("<I", buf, 48, sample.frames)  # Length (in frames)
    struct.pack_into("<I", buf, 60, int(sample.c5speed))  # C5Speed
    struct.pack_into("<I", buf, 72, data_offset)  # SamplePointer
    return bytes(buf)


def _instrument_header(instrument: ITInstrument) -> bytes:
    """Serialize a 554-byte IMPI instrument header with disabled envelopes."""
    if len(instrument.note_map) != _KEYBOARD_NOTES:
        raise ValueError(f"note map must have {_KEYBOARD_NOTES} entries, got {len(instrument.note_map)}")
    buf = bytearray(INSTRUMENT_HEADER_BYTES)
    buf[0:4] = _IMPI
    buf[17] = instrument.new_note_action & 0xFF
    buf[23] = _PPC_C5  # pitch-pan centre
    buf[24] = min(instrument.global_volume, 128)
    buf[25] = instrument.default_pan & 0xFF
    buf[32:58] = _ascii(instrument.name, 26)
    for note, (play_note, sample_number) in enumerate(instrument.note_map):
        buf[64 + 2 * note] = play_note & 0xFF
        buf[64 + 2 * note + 1] = sample_number & 0xFF
    # Envelopes (offsets 304..550) and the 4 trailing reserved bytes stay zero = disabled.
    return bytes(buf)


def _pack_cell(stream: bytearray, channel: int, cell: ITCell) -> None:
    """Append one channel marker + its present fields to a row's packed stream."""
    mask = 0
    if cell.note is not None:
        mask |= _MASK_NOTE
    if cell.instrument is not None:
        mask |= _MASK_INSTRUMENT
    if cell.volume is not None:
        mask |= _MASK_VOLUME
    if cell.effect is not None:
        mask |= _MASK_EFFECT
    if mask == 0:
        return
    stream.append(((channel + 1) | 0x80) & 0xFF)  # channel marker, always followed by an explicit mask
    stream.append(mask)
    if cell.note is not None:
        stream.append(cell.note & 0xFF)
    if cell.instrument is not None:
        stream.append(cell.instrument & 0xFF)
    if cell.volume is not None:
        stream.append(cell.volume & 0xFF)
    if cell.effect is not None:
        command, value = cell.effect
        stream.append(command & 0xFF)
        stream.append(value & 0xFF)


def _pack_pattern(pattern: ITPattern) -> bytes:
    """Serialize a pattern: 8-byte header (packed length, rows, reserved) + the packed row stream."""
    if not 1 <= pattern.rows <= _MAX_ROWS:
        raise ValueError(f"pattern rows {pattern.rows} out of range 1..{_MAX_ROWS}")
    by_row: dict[int, list[tuple[int, ITCell]]] = {}
    for row, channel, cell in pattern.cells:
        if not 0 <= row < pattern.rows:
            raise ValueError(f"cell row {row} out of range 0..{pattern.rows - 1}")
        by_row.setdefault(row, []).append((channel, cell))
    stream = bytearray()
    for row in range(pattern.rows):
        for channel, cell in sorted(by_row.get(row, []), key=lambda item: item[0]):
            _pack_cell(stream, channel, cell)
        stream.append(0)  # end-of-row marker
    return struct.pack("<HHI", len(stream), pattern.rows, 0) + bytes(stream)


def _file_header(module: ITModule, counts: tuple[int, int, int, int]) -> bytes:
    """Serialize the 192-byte IMPM header (up to and including the channel pan/volume arrays)."""
    ord_num, ins_num, smp_num, pat_num = counts
    buf = bytearray(FILE_HEADER_BYTES)
    buf[0:4] = _IMPM
    buf[4:30] = _ascii(module.name, 26)
    flags = _FLAG_USE_INSTRUMENTS | _FLAG_LINEAR_SLIDES
    struct.pack_into("<HHHHHHHH", buf, 30, 0, ord_num, ins_num, smp_num, pat_num, _CWT, _CMWT, flags)
    playback = module.playback
    buf[48] = min(playback.global_volume, 128)
    buf[49] = min(playback.mix_volume, 128)
    buf[50] = playback.speed
    buf[51] = playback.tempo
    buf[52] = 128  # panning separation
    for channel in range(_CHANNELS_STORED):
        buf[64 + channel] = 32  # centre pan
        buf[128 + channel] = 64  # full channel volume
    return bytes(buf)


def _offsets(blobs: list[bytes], start: int) -> list[int]:
    """The file offset each blob occupies when laid end to end from ``start``."""
    result = []
    position = start
    for blob in blobs:
        result.append(position)
        position += len(blob)
    return result


def _serialize_body(module: ITModule, start: int) -> tuple[list[int], bytes]:
    """Serialize everything after the offset tables; return the tables (instr+sample+pattern) and body.

    Sample headers carry a pointer to their PCM, which lands after the patterns, so PCM offsets are
    resolved first and the sample headers built against them.
    """
    instrument_blobs = [_instrument_header(instrument) for instrument in module.instruments]
    pattern_blobs = [_pack_pattern(pattern) for pattern in module.patterns]
    pcm_blobs = [_pcm_bytes(sample) for sample in module.samples]

    samples_at = start + sum(len(blob) for blob in instrument_blobs)
    patterns_at = samples_at + SAMPLE_HEADER_BYTES * len(module.samples)
    data_at = patterns_at + sum(len(blob) for blob in pattern_blobs)

    data_offsets = _offsets(pcm_blobs, data_at)
    sample_blobs = [_sample_header(sample, offset) for sample, offset in zip(module.samples, data_offsets)]

    tables = (
        _offsets(instrument_blobs, start) + _offsets(sample_blobs, samples_at) + _offsets(pattern_blobs, patterns_at)
    )
    body = b"".join(instrument_blobs + sample_blobs + pattern_blobs + pcm_blobs)
    return tables, body


def write_it_module(module: ITModule) -> bytes:
    """Serialize ``module`` to the complete bytes of an uncompressed ``.IT`` file."""
    orders = tuple(module.orders) + (0xFF,)  # 0xFF terminates the order list.
    counts = (len(orders), len(module.instruments), len(module.samples), len(module.patterns))
    table_end = FILE_HEADER_BYTES + len(orders) + 4 * (counts[1] + counts[2] + counts[3])
    tables, body = _serialize_body(module, table_end)

    out = bytearray(_file_header(module, counts))
    out += bytes(orders)
    for offset in tables:
        out += struct.pack("<I", offset)
    out += body
    return bytes(out)


def write_it(path: Path | str, module: ITModule) -> None:
    """Write ``module`` to ``path`` as an uncompressed ``.IT`` file."""
    Path(path).write_bytes(write_it_module(module))
