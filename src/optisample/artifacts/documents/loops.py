from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from optisample.artifacts.serialize import Frozen
from optisample.dsp.decay import NO_DECAY, LinearDecay
from optisample.dsp.loop import Loop, LoopQuality, Material
from optisample.dsp.surrogate import SettledLoop, SettledLoops
from optisample.keys import SampleKey
from optisample.loop.settle import RejectedLoop, Settlement, StoredLoop
from optisample.music import note_name


class LoopRecord(Frozen):
    """The loop actually stored after re-encoding: a half-open ``[start, end)`` frame range."""

    start: int
    end: int


class DecayRecord(Frozen):
    """The ramp a looped sample is played down by: when it falls, and how far.

    Seconds run from the note's onset, ``start_s`` sitting where the stored material ends, and
    ``final_gain`` is the share of the loop's own level a note still sounds at once the ramp is through.
    """

    start_s: float
    end_s: float
    final_gain: float


class LoopQualityRecord(Frozen):
    """What measuring a loop said about it: the wrap it makes, the level it holds, the timbre it holds on to.

    ``seam_step`` reads the wrap in units of the loop region's own frame-to-frame motion,
    ``level_drift_db`` the fall across the region that holding it at one level had to flatten, and
    ``spectral_distance`` the decibel distance between the loop's timbre and the material past it.
    """

    seam_step: float
    level_drift_db: float
    spectral_distance: float


class SettledLoopRecord(Frozen):
    """The loop one recording is stored around, in frames of the recording the stage wrote beside this.

    Frames are what a player wraps between and seconds are where a listener hears it, so both are stated.
    ``decay`` is the ramp a note held past the stored span falls on, where the recording states one to make.
    """

    start: int
    end: int
    start_s: float
    end_s: float
    quality: LoopQualityRecord
    decay: DecayRecord | None


class RejectedLoopRecord(Frozen):
    """A candidate the settlement passed over, and the gate it fell outside of.

    Reading these says why a recording ended up stored around a later loop, or around none: each entry is
    a cheaper loop that was measured and found wanting on ``gate``.
    """

    start_s: float
    end_s: float
    quality: LoopQualityRecord
    gate: str


class RecordingLoopsRecord(Frozen):
    """What the loop stage found for one recording: the loops it offers, and the ones it turned down.

    ``offered`` runs from the cheapest stored span to the dearest, the order an encoding indexes them in,
    and is empty for a recording stored over the span it plays. ``lacking`` states the reading that came up
    short for a recording whose material offered no candidate to measure at all
    (:class:`~optisample.dsp.loop.Material`), so the two ways a recording ends up unlooped read apart:
    ``rejected`` for one whose candidates were measured and passed over, ``lacking`` for one that had none.
    ``cc`` carries the controller buckets its identity was keyed under, so a reader rebuilds the same
    :class:`~optisample.keys.SampleKey` the audio is held under. ``search_s`` is the stretch candidates were
    measured over, which is the longest note the material plays at this pitch.
    """

    key: str
    pitch: int
    note: str
    velocity: int
    cc: list[tuple[int, int]]
    search_s: float
    offered: list[SettledLoopRecord]
    rejected: list[RejectedLoopRecord]
    lacking: str | None


class LoopsDocument(Frozen):
    """Every loop one instrument's recordings offer, beside the dataset they were read from.

    ``sample_rate`` is the rate the recordings were analysed at, which the frames in every
    :class:`SettledLoopRecord` are counted in. Reading this back is what lets a later stage store the loops
    a run already settled, hand-tuned or as they came.
    """

    instrument_id: str
    sample_rate: int
    recordings: list[RecordingLoopsRecord]


def loop_record(loop: Loop | None) -> LoopRecord | None:
    """The loop a stored sample wraps on (``{start, end}``), where it holds one."""
    return None if loop is None else LoopRecord(start=loop.start, end=loop.end)


def decay_record(decay: LinearDecay | None) -> DecayRecord | None:
    """The ramp a stored sample is played down by, where the recording states one to make."""
    if decay is None:
        return None

    return DecayRecord(start_s=decay.start_s, end_s=decay.end_s, final_gain=decay.final_gain)


def _loop_quality_record(quality: LoopQuality) -> LoopQualityRecord:
    """What measuring one candidate said about it, read out into the document's own fields."""
    return LoopQualityRecord(
        seam_step=quality.seam_step,
        level_drift_db=quality.level_drift_db,
        spectral_distance=quality.spectral_distance,
    )


def settled_loop_record(stored: StoredLoop, sample_rate: int) -> SettledLoopRecord:
    """The loop a recording is stored around, stated in frames and in seconds alike."""
    return SettledLoopRecord(
        start=stored.loop.start,
        end=stored.loop.end,
        start_s=stored.loop.start / sample_rate,
        end_s=stored.loop.end / sample_rate,
        quality=_loop_quality_record(stored.quality),
        decay=decay_record(stored.decay),
    )


def _rejected_loop_record(rejected: RejectedLoop, sample_rate: int) -> RejectedLoopRecord:
    """One candidate the settlement passed over, placed in the note by the second."""
    return RejectedLoopRecord(
        start_s=rejected.loop.start / sample_rate,
        end_s=rejected.loop.end / sample_rate,
        quality=_loop_quality_record(rejected.quality),
        gate=rejected.gate.value,
    )


def _lacking_value(lacking: Material | None) -> str | None:
    """The reading a recording came up short on, as the document states it."""
    return None if lacking is None else lacking.value


def loops_document(
    instrument_id: str,
    sample_rate: int,
    settlements: Mapping[SampleKey, Settlement],
    searched: Mapping[SampleKey, float],
) -> LoopsDocument:
    """What the loop stage decided for one instrument, as the document written beside its dataset.

    Recordings come out in key order, so the document a run writes reads the same way twice over the same
    dataset. ``searched`` states the stretch each recording's candidates were measured over.
    """
    return LoopsDocument(
        instrument_id=instrument_id,
        sample_rate=sample_rate,
        recordings=[
            RecordingLoopsRecord(
                key=key.label,
                pitch=key.pitch,
                note=note_name(key.pitch),
                velocity=key.velocity,
                cc=list(key.cc),
                search_s=searched[key],
                offered=[settled_loop_record(stored, sample_rate) for stored in settlements[key].offered],
                rejected=[_rejected_loop_record(rejected, sample_rate) for rejected in settlements[key].rejected],
                lacking=_lacking_value(settlements[key].lacking),
            )
            for key in sorted(settlements)
        ],
    )


def _settled_key(record: RecordingLoopsRecord) -> SampleKey:
    """The identity one document entry names, rebuilt as the audio map holds it."""
    return SampleKey(pitch=record.pitch, velocity=record.velocity, cc=tuple(record.cc))


def read_loops(path: Path) -> LoopsDocument:
    """The loops document at ``path``, validated against its own shape."""
    return LoopsDocument.model_validate_json(path.read_text(encoding="utf-8"))


def settled_offers(offered: Sequence[SettledLoopRecord]) -> SettledLoops:
    """One recording's offers in the shape every encode reads them through, in the order params index them.

    Frames come back as they were written, so records read beside the dataset they were measured over name
    the same stretches of the same recording under the same indices. Every artifact carrying settled loops
    -- the loops document and the calibrated ``.sample`` alike -- puts them back through here.
    """
    return tuple(
        SettledLoop(loop=Loop(start=stored.start, end=stored.end), decay=_read_decay(stored.decay))
        for stored in offered
    )


def settled_loops(document: LoopsDocument) -> dict[SampleKey, SettledLoops]:
    """The loops a document states, keyed the way the audio they were measured over is."""
    return {_settled_key(record): settled_offers(record.offered) for record in document.recordings}


def _read_decay(record: DecayRecord | None) -> LinearDecay | None:
    """The ramp one document entry states, where it states one."""
    if record is None:
        return NO_DECAY

    return LinearDecay(start_s=record.start_s, end_s=record.end_s, final_gain=record.final_gain)
