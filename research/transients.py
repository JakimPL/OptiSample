from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from optisample.config.root import OptiConfig
from optisample.dsp.onset import ATTACK_READING, attack_seconds, crest_db, onset_frame, onset_window
from optisample.dsp.timebase import seconds_to_frames
from optisample.io.tracker.envelope import envelope_grid
from optisample.io.tracker.target import export_target
from optisample.metrics.composite import CompositeFidelity, build_composite, evaluate
from optisample.optimize.carrier import CurveSettings
from research.bench import Recording, encoder

Signal = NDArray[np.float64]

_TEMPO: Final = 125
_QUIET: Final = 1e-12
_DEPTHS: Final = (16, 8)
_EVERY_ATTACK: Final = 0.0  # the gate the bench reads under, so both storages are measured on every take
_MS_PER_S: Final = 1000.0


@dataclass(frozen=True)
class Storage:
    """One way a sample may be held: the grid it is quantized on, and whether its level travels apart."""

    depth: int
    carrier: bool

    @property
    def label(self) -> str:
        """How this storage is named in a record."""
        return f"{self.depth}{'c' if self.carrier else 'p'}"


class TransientRecord(BaseModel):
    """What one storage of one take did to its own onset, read beside what the objective made of it.

    ``fidelity`` is the composite over the whole span -- the number an allocation ranks by -- and
    ``onset_fidelity`` the same composite over the onset alone. The three readings beside them are shape
    rather than spectrum: how much of the attack's crest survived, how much longer the rise takes, and how
    far the noise ahead of the onset came up. Each is read from the signal's own onset, so a delay the
    codec introduces leaves them alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str
    material_class: str
    key: str
    storage: str
    depth: int
    carrier: bool
    fidelity: float
    onset_fidelity: float
    crest_loss_db: float
    attack_growth_ms: float
    pre_onset_rise_db: float
    onset_segmental_snr_db: float


def floor_db(signal: Signal, sample_rate: int) -> float:
    """The power sitting ahead of a take's own onset, in decibels under the whole take's peak.

    A curve that has already begun to rise where the material has not divides a stored waveform's opening
    silence up rather than down, so this is where a level handed to a coarse grid shows itself.
    """
    at = onset_frame(signal, sample_rate, ATTACK_READING)
    ahead = signal[max(0, at - seconds_to_frames(ATTACK_READING.pre_s, sample_rate)) : at]
    power = float(np.sqrt(np.mean(np.square(ahead)))) if ahead.size else 0.0
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return 20.0 * float(np.log10(max(power, _QUIET) / max(peak, _QUIET)))


def storages() -> tuple[Storage, ...]:
    """Every storage the bench compares: each grid held as it was played, and each handed to a curve."""
    return tuple(Storage(depth=depth, carrier=carrier) for depth in _DEPTHS for carrier in (False, True))


def _scored(
    recording: Recording,
    storage: Storage,
    written: Signal,
    composite: CompositeFidelity,
) -> TransientRecord:
    """One record: what the objective made of the whole span, and what the onset itself measures."""
    reference, sample_rate = recording.signal, recording.sample_rate
    whole = evaluate(reference, written, sample_rate, composite)
    onset = evaluate(
        onset_window(reference, sample_rate, ATTACK_READING),
        onset_window(written, sample_rate, ATTACK_READING),
        sample_rate,
        composite,
    )
    return TransientRecord(
        instrument_id=recording.instrument_id,
        material_class=recording.material_class,
        key=str(recording.key),
        storage=storage.label,
        depth=storage.depth,
        carrier=storage.carrier,
        fidelity=whole.fidelity,
        onset_fidelity=onset.fidelity,
        crest_loss_db=crest_db(reference, sample_rate, ATTACK_READING) - crest_db(written, sample_rate, ATTACK_READING),
        attack_growth_ms=_MS_PER_S
        * (
            attack_seconds(written, sample_rate, ATTACK_READING)
            - attack_seconds(reference, sample_rate, ATTACK_READING)
        ),
        pre_onset_rise_db=floor_db(written, sample_rate) - floor_db(reference, sample_rate),
        onset_segmental_snr_db=onset.diagnostics["segmental_snr_db"],
    )


def measure(
    recordings: Sequence[Recording],
    kept: Sequence[Storage],
    config: OptiConfig,
    rate: int,
) -> list[TransientRecord]:
    """Score every storage of every recording, on the whole span and on the onset alone.

    Both readings come off the very encoder a run stores with, so what they measure is what a module would
    carry, and the gate is opened for all of them so a storage the shipped config would refuse is still
    measured -- which is what says whether refusing it was right.

    Scoring the onset apart is the whole point: every term of the composite is a mean over frames, so a
    transient lasting a thousandth of a note reaches the objective as a thousandth of it, however much of
    what a listener recognises lives there.
    """
    target = export_target(config.export.tracker)
    grid = envelope_grid(target, tempo=_TEMPO, release_s=config.export.envelope.release_s)
    settings = CurveSettings(config=config.encode, target=target, grid=grid, min_attack_ticks=_EVERY_ATTACK)
    encoded = encoder(config, settings)
    composite = build_composite(config.analysis.metrics)
    return [
        _scored(
            recording,
            storage,
            encoded(rate, storage.depth, compress=False, carrier=storage.carrier)(recording),
            composite,
        )
        for recording in recordings
        for storage in kept
    ]


def write(records: Sequence[TransientRecord], path: Path) -> None:
    """Write the bench's readings as one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")


def read(path: Path) -> list[TransientRecord]:
    """Every reading the bench left at ``path``."""
    return [TransientRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
