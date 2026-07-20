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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from optisample.config.render import PlaybackConfig
from optisample.dsp.surrogate import MAX_VOLUME
from optisample.io.it_format import (
    CHANNELS_STORED,
    FILE_HEADER,
    INSTRUMENT_HEADER,
    KEYBOARD_NOTES,
    MAX_IT_NOTE,
    SAMPLE_HEADER,
)
from optisample.metrics.size import FILE_HEADER_BYTES, SAMPLE_HEADER_BYTES

_IMPM = b"IMPM"
_IMPS = b"IMPS"
_IMPI = b"IMPI"

_NAME_BYTES = 26  # every IT name field (song, instrument, sample) is 26 ASCII bytes, null-padded.

MAX_ROWS = 200  # IT patterns hold 1..200 rows.
TICKS_PER_ROW_BASE = 2.5  # one tick lasts 2.5 / tempo seconds; a row lasts `speed` ticks.

NOTE_OFF = 255  # pattern note value that releases the playing note.
NOTE_CUT = 254  # pattern note value that cuts it instantly.

_CWT = 0x0214  # "created with" IT 2.14 -> selects the 554-byte instrument + envelope format.
_CMWT = 0x0214  # "compatible with"; >= 0x0200 is required for the instrument format.
_FLAG_USE_INSTRUMENTS = 0x04
_FLAG_LINEAR_SLIDES = 0x08
_PPC_C5 = 60  # pitch-pan centre at C-5.

_SMP_FLAG_DATA = 0x01  # sample data present.
_SMP_FLAG_16BIT = 0x02  # 16-bit (else 8-bit).
_SMP_FLAG_LOOP = 0x10  # forward loop enabled (loop begin/end fields are read).
_CVT_SIGNED = 0x01  # signed PCM (the standard IT storage).

MAX_GLOBAL_VOLUME = 128  # file-header and instrument global-volume ceiling.
MAX_MIX_VOLUME = 128  # file-header mix-volume ceiling.
PANNING_SEPARATION = 128  # full stereo separation in the file header.
PAN_CENTER = 32  # centred channel pan (IT pan spans 0..64).
CHANNEL_VOLUME_FULL = MAX_VOLUME  # every stored channel plays at full volume (0..64).

_MASK_NOTE = 0x01
_MASK_INSTRUMENT = 0x02
_MASK_VOLUME = 0x04
_MASK_EFFECT = 0x08

CHANNEL_MARKER = 0x80  # high bit set on a packed cell's channel byte (always followed by a mask).
END_OF_ROW = 0x00  # a zero byte terminates a packed pattern row.
ORDER_TERMINATOR = 0xFF  # ends the order list.
OFFSET_TABLE_ENTRY_BYTES = 4  # each instrument/sample/pattern offset is a little-endian u32.

_INT16_SCALE = 32768.0
_INT8_SCALE = 128.0


def require_it_note(note: int) -> int:
    """Return ``note`` if it is a playable IT key (0..119); raise ``ValueError`` otherwise."""
    if not 0 <= note <= MAX_IT_NOTE:
        raise ValueError(f"note {note} is outside the IT key range 0..{MAX_IT_NOTE}")
    return note


@dataclass(frozen=True)
class ITSample:
    """One stored sample: its PCM (float in ``[-1, 1]``), storage depth and playback rate."""

    name: str
    pcm: NDArray[np.floating]
    depth_bits: int = 16
    c5speed: int = 44_100
    global_volume: int = 64
    default_volume: int = 64
    loop: tuple[int, int] | None = None  # forward loop over half-open frame range [begin, end)

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
    """Global playback settings (bundled to keep :class:`ITModule` small).

    Built from :class:`~optisample.config.render.PlaybackConfig` at the export boundary via
    :func:`it_playback`; ``speed``/``tempo`` set the row duration, so they affect rendered timing.
    """

    speed: int
    tempo: int
    global_volume: int
    mix_volume: int


def it_playback(config: PlaybackConfig) -> ITPlayback:
    """Build the writer's :class:`ITPlayback` value-object from a :class:`PlaybackConfig`."""
    return ITPlayback(
        speed=config.speed,
        tempo=config.tempo,
        global_volume=config.global_volume,
        mix_volume=config.mix_volume,
    )


@dataclass(frozen=True)
class ITModule:
    """A complete module ready to serialize."""

    name: str
    samples: tuple[ITSample, ...]
    instruments: tuple[ITInstrument, ...]
    patterns: tuple[ITPattern, ...]
    orders: tuple[int, ...]
    playback: ITPlayback


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
    flags = _SMP_FLAG_DATA | (_SMP_FLAG_16BIT if sample.depth_bits == 16 else 0)
    loop_begin, loop_end = 0, 0
    if sample.loop is not None:
        flags |= _SMP_FLAG_LOOP
        loop_begin, loop_end = sample.loop  # Loop End is the frame after the loop; playback wraps here.
    return SAMPLE_HEADER.pack(
        {
            "magic": _IMPS,
            "global_volume": min(sample.global_volume, MAX_VOLUME),
            "flags": flags,
            "default_volume": min(sample.default_volume, MAX_VOLUME),
            "name": _ascii(sample.name, _NAME_BYTES),
            "convert": _CVT_SIGNED,
            "length": sample.frames,
            "loop_begin": loop_begin,
            "loop_end": loop_end,
            "c5speed": int(sample.c5speed),
            "sample_pointer": data_offset,
        }
    )


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
    stream.append(((channel + 1) | CHANNEL_MARKER) & 0xFF)
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
    if not 1 <= pattern.rows <= MAX_ROWS:
        raise ValueError(f"pattern rows {pattern.rows} out of range 1..{MAX_ROWS}")
    by_row: dict[int, list[tuple[int, ITCell]]] = {}
    for row, channel, cell in pattern.cells:
        if not 0 <= row < pattern.rows:
            raise ValueError(f"cell row {row} out of range 0..{pattern.rows - 1}")
        by_row.setdefault(row, []).append((channel, cell))
    stream = bytearray()
    for row in range(pattern.rows):
        for channel, cell in sorted(by_row.get(row, []), key=lambda item: item[0]):
            _pack_cell(stream, channel, cell)
        stream.append(END_OF_ROW)
    return struct.pack("<HHI", len(stream), pattern.rows, 0) + bytes(stream)


def _file_header(module: ITModule, counts: tuple[int, int, int, int]) -> bytes:
    """Serialize the 192-byte IMPM header (up to and including the channel pan/volume arrays)."""
    ord_num, ins_num, smp_num, pat_num = counts
    playback = module.playback
    return FILE_HEADER.pack(
        {
            "magic": _IMPM,
            "name": _ascii(module.name, _NAME_BYTES),
            "highlight": 0,
            "order_count": ord_num,
            "instrument_count": ins_num,
            "sample_count": smp_num,
            "pattern_count": pat_num,
            "created_with": _CWT,
            "compatible_with": _CMWT,
            "flags": _FLAG_USE_INSTRUMENTS | _FLAG_LINEAR_SLIDES,
            "global_volume": min(playback.global_volume, MAX_GLOBAL_VOLUME),
            "mix_volume": min(playback.mix_volume, MAX_MIX_VOLUME),
            "speed": playback.speed,
            "tempo": playback.tempo,
            "panning_separation": PANNING_SEPARATION,
            "channel_pan": bytes([PAN_CENTER]) * CHANNELS_STORED,
            "channel_volume": bytes([CHANNEL_VOLUME_FULL]) * CHANNELS_STORED,
        }
    )


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
    orders = tuple(module.orders) + (ORDER_TERMINATOR,)
    counts = (len(orders), len(module.instruments), len(module.samples), len(module.patterns))
    table_end = FILE_HEADER_BYTES + len(orders) + OFFSET_TABLE_ENTRY_BYTES * (counts[1] + counts[2] + counts[3])
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
