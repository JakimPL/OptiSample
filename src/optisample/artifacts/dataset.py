from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from optisample.artifacts.documents.reduction import WrittenSampleRecord
from optisample.artifacts.instruments.dump import InstrumentSettings, WrittenInstruments, write_dataset_instruments
from optisample.io.audio import write_wav
from optisample.io.dataset import SourceDataset, SubsetDataset
from optisample.io.note_extractor import NoteRecord
from optisample.io.source import write_source_subset
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


def recording_stem(key: SampleKey, index: int) -> str:
    """The name one recording is written under: the render index it joins on, then the identity it holds.

    The leading index is what a later ingest reads back to pair a note with its file, and the rest says
    which recording that file holds, so a dataset written twice over the same recordings names them alike.
    Everything a stage writes about one recording shares this stem, which is what pairs a calibrated
    ``.sample`` with the WAV standing beside it.
    """
    return f"{index:04d}_{key.label}"


def write_recording(
    signal: Signal,
    key: SampleKey,
    index: int,
    sample_rate: int,
    samples_dir: Path,
) -> WrittenSampleRecord:
    """Write one recording as ``{stem}.wav`` and state what a reader finds there (see :func:`recording_stem`)."""
    name = f"{recording_stem(key, index)}.wav"
    write_wav(samples_dir / name, signal, sample_rate)
    return WrittenSampleRecord(
        index=index,
        key=key.label,
        file=name,
        frames=int(signal.size),
        duration_s=int(signal.size) / sample_rate,
    )


@dataclass(frozen=True)
class SlicedDataset:
    """The slice one run took of its source, beside the instruments written from what it kept."""

    dataset: SubsetDataset
    instruments: WrittenInstruments


def write_slice(
    source: SourceDataset,
    out_dir: Path,
    *,
    instrument_id: str,
    fraction: float,
    instruments: InstrumentSettings,
) -> SlicedDataset:
    """Write the share of ``source`` a slice keeps, and carry each recording it kept as an instrument.

    A slice is written in the shape its source came in, so it reads back the way the whole dataset would,
    and the instruments beside it make the very first stage of a run playable in a tracker.
    """
    dataset = write_source_subset(source, out_dir, instrument_id=instrument_id, fraction=fraction)
    return SlicedDataset(
        dataset=dataset,
        instruments=write_dataset_instruments(dataset.source, settings=instruments),
    )
