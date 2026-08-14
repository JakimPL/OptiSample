from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import model_validator

from optisample.artifacts.documents.loops import (
    SettledLoopRecord,
    settled_loop_record,
    settled_offers,
)
from optisample.artifacts.serialize import Frozen, read_msgpack
from optisample.dsp.envelope import Decomposition, Signal
from optisample.dsp.loop import Loop
from optisample.dsp.surrogate import SettledLoops
from optisample.keys import SampleKey
from optisample.loop.settle import StoredLoop

PAYLOAD_DTYPE: Final = np.dtype("<f4")  # little-endian float32, the width the float WAVs beside it carry


class ProvenanceRecord(Frozen):
    """Where a calibrated sample came from: the stage that wrote it, and the dataset entry it was read off.

    A container travels on its own, so it states the run behind it and a reader learns that from the file
    rather than from where the file happens to sit. ``index`` is the render index the dataset joins a note
    to its recording on and ``source`` names the WAV of the same stem holding that recording whole, so a
    reader pairs the two and checks the split against the audio it was taken from.
    """

    stage: str
    instrument_id: str
    index: int
    source: str


class SampleDocument(Frozen):
    """One recording as the calibrated unit a later stage stores it from: its carrier, level and loops.

    ``level * carrier`` is the recording as the loop stage analysed it
    (:class:`~optisample.dsp.envelope.Decomposition`), so the pair between them carries everything the WAV
    beside it holds, while a copy stored from the carrier alone spends the whole depth of its grid on the
    waveform -- worth +5.4...+26.1 dB segmental SNR at 8 bits, since the attack transient stops setting the
    code range and the level travels as a curve instead.

    ``root_pitch`` is the pitch the recording was played at, which its level was read over two periods of
    and every candidate loop searched around, and ``sample_rate`` the rate the stage analysed it at, which
    the frames of both payloads and of every loop are counted in. ``loops`` runs from the cheapest stored
    span upward, the order an encoding indexes them by, so the container names the stretches of its own
    carrier a player may wrap on. ``carrier`` and ``level`` hold one :data:`PAYLOAD_DTYPE` value per frame
    and ``frames`` states how many, which is what a reader checks a payload against.
    """

    key: str
    root_pitch: int
    sample_rate: int
    frames: int
    carrier: bytes
    level: bytes
    loops: list[SettledLoopRecord]
    provenance: ProvenanceRecord

    @model_validator(mode="after")
    def payloads_hold_the_frames_stated(self) -> SampleDocument:
        """Hold both payloads to the frame count the document states, so a curve reads back whole.

        Raises:
            ValueError: when either payload holds a number of frames other than ``frames``.
        """
        stated = self.frames * PAYLOAD_DTYPE.itemsize
        held = {"carrier": len(self.carrier), "level": len(self.level)}
        short = {name: length for name, length in held.items() if length != stated}
        if short:
            raise ValueError(f"{short} bytes against the {stated} the {self.frames} frames of {self.key} ask for")

        return self


def _packed(curve: Signal) -> bytes:
    """A curve as the payload a container stores: one little-endian float32 per frame."""
    return np.asarray(curve, dtype=PAYLOAD_DTYPE).tobytes()


def _unpacked(payload: bytes) -> Signal:
    """The curve one payload holds, read back at the width every stage does its arithmetic in."""
    return np.asarray(np.frombuffer(payload, dtype=PAYLOAD_DTYPE), dtype=np.float64)


def sample_document(
    decomposition: Decomposition,
    offered: Sequence[StoredLoop],
    *,
    key: SampleKey,
    sample_rate: int,
    provenance: ProvenanceRecord,
) -> SampleDocument:
    """One recording's split and its settled loops, as the container the stage writes beside its WAV."""
    return SampleDocument(
        key=key.label,
        root_pitch=key.pitch,
        sample_rate=sample_rate,
        frames=int(decomposition.carrier.size),
        carrier=_packed(decomposition.carrier),
        level=_packed(decomposition.level),
        loops=[settled_loop_record(stored, sample_rate) for stored in offered],
        provenance=provenance,
    )


def read_sample(path: Path) -> SampleDocument:
    """The calibrated sample at ``path``, validated against its own shape."""
    return read_msgpack(path, SampleDocument)


def sample_decomposition(document: SampleDocument) -> Decomposition:
    """The recording a container holds, as the pair it was split into: ``level`` times ``carrier``."""
    return Decomposition(level=_unpacked(document.level), carrier=_unpacked(document.carrier))


def sample_loops(document: SampleDocument) -> SettledLoops:
    """The loops a container states, in the shape every encode reads them through."""
    return settled_offers(document.loops)


def faithful_offer(document: SampleDocument) -> int | None:
    """Which of a recording's settled loops stands closest to the material it stands in for.

    The offers a container holds are ordered by the span each stores, cheapest first, so the one a listener
    would pick is named by its timbre distance rather than by its position. A recording the loop stage
    settled nothing over answers with nothing, which stores the span it plays.
    """
    if not document.loops:
        return None

    return min(range(len(document.loops)), key=lambda index: document.loops[index].quality.spectral_distance)


def faithful_loop(document: SampleDocument) -> Loop | None:
    """The region a player wraps on to hold a note the container's recording sounds past its own end.

    This is :func:`faithful_offer` as the frames it names, which is what a reading of the recording whole
    -- one waveform, at the rate the stage analysed it -- repeats to sustain the note.
    """
    offer = faithful_offer(document)
    if offer is None:
        return None

    return sample_loops(document)[offer].loop
