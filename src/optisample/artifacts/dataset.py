from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from optisample.artifacts.documents.reduction import WrittenSampleRecord
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NoteRecord
from optisample.keys import SampleKey, nearest_key
from optisample.metrics.base import Signal
from optisample.model import NoteEvent
from optisample.optimize.tasks import AudioMap

RenderIndices = Mapping[SampleKey, int]


def keys_by_pitch(audio: AudioMap) -> dict[int, list[SampleKey]]:
    """The recordings available at each pitch, in key order, which is what a played note is routed among."""
    available: dict[int, list[SampleKey]] = {}
    for key in sorted(audio):
        available.setdefault(key.pitch, []).append(key)

    return available


def note_records(
    material: Sequence[NoteEvent],
    audio: AudioMap,
    indices: RenderIndices,
) -> list[NoteRecord]:
    """Every played note as a dataset entry, pointing at the recording it is scored against.

    A note is routed by :func:`~optisample.keys.nearest_key`, the same lookup
    :func:`~optisample.optimize.reduce.events.merge_events` scores it through, so a written dataset pairs
    each note with exactly the recording the objective already measured it against. An event standing for
    several played notes is written once per note, so the material keeps its weight.

    Every stage writing a dataset routes its notes this way, so a stage reading one back stands where the
    stage before it left off.
    """
    available = keys_by_pitch(audio)
    return [
        NoteRecord(
            index=indices[nearest_key(available[event.pitch], event.velocity)],
            pitch=event.pitch,
            velocity=event.velocity,
            duration_s=event.duration_s,
            cc_averages=event.cc_averages,
        )
        for event in material
        for _ in range(event.count)
    ]


def tracked_ccs(material: Sequence[NoteEvent]) -> list[int]:
    """Every controller the material reports an average for, which the dataset declares up front."""
    return sorted({controller for event in material for controller in event.cc_averages})


def write_recording(
    signal: Signal,
    key: SampleKey,
    index: int,
    sample_rate: int,
    samples_dir: Path,
) -> WrittenSampleRecord:
    """Write one recording as ``{index:04d}_{key label}.wav`` and state what a reader finds there.

    The leading index is the render index a later ingest reads back and the rest of the name says which
    recording the file holds, so a dataset written twice over the same recordings names them alike.
    """
    name = f"{index:04d}_{key.label}.wav"
    write_wav(samples_dir / name, signal, sample_rate)
    return WrittenSampleRecord(
        index=index,
        key=key.label,
        file=name,
        frames=int(signal.size),
        duration_s=int(signal.size) / sample_rate,
    )
