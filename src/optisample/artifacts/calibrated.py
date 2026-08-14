from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.artifacts.dataset import recording_stem
from optisample.artifacts.documents.sample import ProvenanceRecord, SampleDocument, sample_document
from optisample.artifacts.paths import calibrated_path
from optisample.artifacts.serialize import write_msgpack
from optisample.config.loop import LoopConfig
from optisample.dsp.envelope import decompose, level_reading
from optisample.keys import SampleKey
from optisample.loop.settle import StoredLoop
from optisample.metrics.base import Signal
from optisample.music import midi_to_freq

_RECORDING_SUFFIX: Final = ".wav"  # the file a container stands beside, which its provenance names


@dataclass(frozen=True)
class CalibratedRecording:
    """One recording a container is written for: the audio as its stage holds it, and what it offers.

    ``signal`` is the very audio landing in the WAV of the same stem, so every loop in ``offered`` names
    frames of the file standing beside the container. ``index`` is the render index the dataset joins a
    note to this recording on, which names both files.
    """

    key: SampleKey
    index: int
    signal: Signal
    offered: tuple[StoredLoop, ...]


@dataclass(frozen=True)
class CalibratedRun:
    """What every container one stage writes shares: where they land, the run behind them, and the rate.

    ``stage`` is what each container states as its own provenance, so a container carried off on its own
    says whose decisions it holds, and ``config`` supplies the weighting the level is read through, which
    is the one every measurement of these recordings was taken under.
    """

    samples_dir: Path
    instrument_id: str
    stage: str
    sample_rate: int
    config: LoopConfig


def calibrated_sample(recording: CalibratedRecording, run: CalibratedRun) -> SampleDocument:
    """``recording`` as the calibrated unit it is carried on: its split, its loops and where it came from.

    The level is read at the pitch the key sounds, which is the reading its loops were settled under, so
    the split a container carries is the one every measurement of that recording was taken through. A
    recording offering no loop states none, and the pair it holds still puts the recording back together.
    """
    reading = level_reading(run.sample_rate, run.config.envelope, midi_to_freq(recording.key.pitch))
    return sample_document(
        decompose(recording.signal, reading),
        recording.offered,
        key=recording.key,
        sample_rate=run.sample_rate,
        provenance=ProvenanceRecord(
            stage=run.stage,
            instrument_id=run.instrument_id,
            index=recording.index,
            source=f"{recording_stem(recording.key, recording.index)}{_RECORDING_SUFFIX}",
        ),
    )


def write_calibrated(recordings: Iterable[CalibratedRecording], run: CalibratedRun) -> None:
    """Write every recording as a ``.sample`` beside the WAV of the same stem.

    A container states the recording split into the level it moves through and the carrier that level
    scales, so a stage reading one back stores the carrier over the whole depth of its grid and plays the
    level as a curve. Its loops travel with it, which is what carries the stage's decision forward in the
    company of the very audio it was measured over, and what the standalone instruments written from the
    dataset wrap on (:func:`~optisample.artifacts.instruments.dump.write_dataset_instruments`).
    """
    for recording in recordings:
        stem = recording_stem(recording.key, recording.index)
        write_msgpack(calibrated_path(run.samples_dir, stem), calibrated_sample(recording, run))
