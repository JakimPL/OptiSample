from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from optisample.config.root import OptiConfig
from optisample.io.tracker.envelope import envelope_grid
from optisample.io.tracker.target import export_target
from optisample.metrics.composite import CompositeFidelity, build_composite, evaluate
from optisample.optimize.carrier import CurveSettings
from research.bench import Recording, encoder

Signal = NDArray[np.float64]

_TEMPO: Final = 125
_ONSET_SHARE: Final = 0.1  # share of a take's peak the onset is placed at
_RISE_LOW: Final = 0.1  # share of the attack's peak a rise is read from
_RISE_HIGH: Final = 0.9  # share of it the rise is read to
_PRE_MS: Final = 10.0  # stretch before an onset a stored sample's own noise floor is read over
_ONSET_MS: Final = 60.0  # stretch after it a transient is scored over, which holds the widest analysis window
_SMOOTH_MS: Final = 0.5  # stretch the amplitude is smoothed over before a rise is read off it
_QUIET: Final = 1e-12
_DEPTHS: Final = (16, 8)


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


def _frames(milliseconds: float, sample_rate: int) -> int:
    """``milliseconds`` as a whole number of frames, at least one."""
    return max(1, round(milliseconds * sample_rate / 1000.0))


def _smoothed(signal: Signal, sample_rate: int) -> Signal:
    """The amplitude ``signal`` holds, averaged over a stretch short enough to leave a transient standing."""
    window = _frames(_SMOOTH_MS, sample_rate)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.asarray(np.convolve(np.abs(signal), kernel, mode="same"), dtype=np.float64)


@dataclass(frozen=True)
class Onset:
    """One signal and the frame its attack begins at, which every shape reading here is taken from.

    Each signal states its own onset, so a stored copy the codec delayed is measured from where its own
    attack starts and the readings stay about the shape of that attack.
    """

    signal: Signal
    sample_rate: int
    at: int

    @classmethod
    def read(cls, signal: Signal, sample_rate: int) -> Onset:
        """``signal`` placed at the frame it first reaches a tenth of its own peak."""
        amplitude = _smoothed(signal, sample_rate)
        peak = float(amplitude.max()) if amplitude.size else 0.0
        reached = np.flatnonzero(amplitude >= _ONSET_SHARE * peak) if peak > _QUIET else np.array([], dtype=np.int_)
        return cls(signal=signal, sample_rate=sample_rate, at=int(reached[0]) if reached.size else 0)

    @property
    def window(self) -> Signal:
        """The stretch a transient is read over: a little before the onset, and the attack after it."""
        low = max(0, self.at - _frames(_PRE_MS, self.sample_rate))
        return np.asarray(
            self.signal[low : low + _frames(_PRE_MS + _ONSET_MS, self.sample_rate)],
            dtype=np.float64,
        )

    @property
    def crest_db(self) -> float:
        """How far the attack's peak stands above the power it carries, in decibels.

        A transient is a peak a spectrum-averaged reading spreads out, so the crest is what says whether
        the stored copy still has one. Being a ratio inside one window, it reads the same at any level.
        """
        window = self.window
        power = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
        peak = float(np.max(np.abs(window))) if window.size else 0.0
        return 20.0 * float(np.log10(max(peak, _QUIET) / max(power, _QUIET)))

    @property
    def attack_ms(self) -> float:
        """How long the attack runs from a tenth of its peak to nine tenths of it, in milliseconds."""
        amplitude = _smoothed(self.window, self.sample_rate)
        peak = float(amplitude.max()) if amplitude.size else 0.0
        if peak <= _QUIET:
            return 0.0

        low = np.flatnonzero(amplitude >= _RISE_LOW * peak)
        high = np.flatnonzero(amplitude >= _RISE_HIGH * peak)
        if not low.size or not high.size:
            return 0.0

        return 1000.0 * max(0, int(high[0]) - int(low[0])) / self.sample_rate

    @property
    def floor_db(self) -> float:
        """The power sitting ahead of the onset, in decibels under the whole take's peak."""
        ahead = self.signal[max(0, self.at - _frames(_PRE_MS, self.sample_rate)) : self.at]
        power = float(np.sqrt(np.mean(np.square(ahead)))) if ahead.size else 0.0
        peak = float(np.max(np.abs(self.signal))) if self.signal.size else 0.0
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
    reference = Onset.read(recording.signal, recording.sample_rate)
    stored = Onset.read(written, recording.sample_rate)
    whole = evaluate(recording.signal, written, recording.sample_rate, composite)
    onset = evaluate(reference.window, stored.window, recording.sample_rate, composite)
    return TransientRecord(
        instrument_id=recording.instrument_id,
        material_class=recording.material_class,
        key=str(recording.key),
        storage=storage.label,
        depth=storage.depth,
        carrier=storage.carrier,
        fidelity=whole.fidelity,
        onset_fidelity=onset.fidelity,
        crest_loss_db=reference.crest_db - stored.crest_db,
        attack_growth_ms=stored.attack_ms - reference.attack_ms,
        pre_onset_rise_db=stored.floor_db - reference.floor_db,
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
    carry. Scoring the onset apart is the whole point: every term of the composite is a mean over frames,
    so a transient lasting a thousandth of a note reaches the objective as a thousandth of it, however
    much of what a listener recognises lives there.
    """
    target = export_target(config.export.tracker)
    grid = envelope_grid(target, tempo=_TEMPO, release_s=config.export.envelope.release_s)
    encoded = encoder(config, CurveSettings(config=config.encode, target=target, grid=grid))
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
