from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from optisample.config.root import OptiConfig
from optisample.dsp.spectral import bandlimit
from optisample.dsp.surrogate import EncodeContext, EncodingParams, render
from optisample.io.audio import mono, read_wav
from optisample.io.tracker.envelope import NO_ENVELOPE, envelope_grid
from optisample.io.tracker.target import export_target
from optisample.keys import SampleKey
from optisample.metrics.composite import build_composite, evaluate
from optisample.optimize.carrier import CurveSettings, PlayedCurve, clip_envelope, stored_carrier

Signal = NDArray[np.float64]

_TEMPO: Final = 125
_SEED: Final = 137
_STEP: Final = 0.5  # the height of that discontinuity, as a share of the span's own peak
_CLICKS: Final = (1, 8)  # discontinuities a badly wrapped loop makes over one span
_DEPTHS: Final = (16, 8)  # grids a stored sample is offered at
_KEEP: Final = (0.25, 0.5, 0.75)  # shares of a span a hard trim keeps
_BANDS_HZ: Final = (2000.0, 4000.0, 8000.0, 16000.0)  # brickwall edges, which cost band without costing rate


@dataclass(frozen=True)
class Recording:
    """One take the bench measures on, and what kind of sound it stands for."""

    instrument_id: str
    material_class: str
    key: SampleKey
    signal: Signal
    sample_rate: int


@dataclass(frozen=True)
class Degradation:
    """One way of spending fewer bytes, named by the axis it spends them on."""

    axis: str
    label: str
    apply: Callable[[Recording], Signal]


class BenchRecord(BaseModel):
    """What one degradation of one recording measured, on the composite and beside it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str
    material_class: str
    key: str
    axis: str
    label: str
    fidelity: float
    breakdown: dict[str, float]
    segmental_snr_db: float | None
    si_sdr_db: float | None
    loudness_delta_lu: float | None


def load_recordings(catalogue: Sequence[tuple[str, str, Path]], *, longest: int) -> tuple[Recording, ...]:
    """The takes the bench reads, ``longest`` per instrument, chosen by how long each one sounds."""
    gathered: list[Recording] = []
    for instrument_id, material_class, folder in catalogue:
        takes = sorted(folder.glob("*.wav"))
        opened = [(path, *read_wav(path)) for path in takes]
        ranked = sorted(opened, key=lambda item: item[1].shape[0] / item[2], reverse=True)[:longest]
        for path, samples, rate in ranked:
            gathered.append(
                Recording(
                    instrument_id=instrument_id,
                    material_class=material_class,
                    key=_key_of(path.name),
                    signal=mono(samples),
                    sample_rate=rate,
                )
            )

    return tuple(gathered)


def _key_of(name: str) -> SampleKey:
    """The key a dataset filename spells, read off its ``p``/``v`` tokens."""
    tokens = name.split("_")
    pitch = next(int(token[1:]) for token in tokens if token.startswith("p") and token[1:].isdigit())
    velocity = next(int(token[1:]) for token in tokens if token.startswith("v") and token[1:].isdigit())
    return SampleKey(pitch, velocity)


def _stored(recording: Recording, params: EncodingParams, config: OptiConfig, curve: PlayedCurve) -> Signal:
    """``recording`` put through the codec under ``params`` and rendered back at its own pitch."""
    context = EncodeContext(root_pitch=recording.key.pitch, config=config.encode, rng=np.random.default_rng(_SEED))
    return render(
        stored_carrier(recording.signal, recording.sample_rate, params, context, curve),
        recording.sample_rate,
        pitch=recording.key.pitch,
    )


def _curve(recording: Recording, settings: CurveSettings) -> PlayedCurve:
    return PlayedCurve(clip_envelope(recording.signal, recording.key, recording.sample_rate, settings), _TEMPO)


def _clicked(signal: Signal, wraps: int) -> Signal:
    """``signal`` with ``wraps`` discontinuities stepped into it, which is what a bad wrap sounds like."""
    stepped = np.array(signal, dtype=np.float64)
    peak = float(np.max(np.abs(stepped))) or 1.0
    for index in range(1, wraps + 1):
        at = int(stepped.size * index / (wraps + 1))
        stepped[at:] = stepped[at:] + _STEP * peak * (-1.0) ** index

    return stepped


def _trimmed(signal: Signal, keep: float) -> Signal:
    """``signal`` cut to ``keep`` of its length and played out to the end, which is a sample running out.

    The span is held to the length the note sounds for rather than to the length it kept, since a player
    asked for a note longer than the sample it stores sounds silence from there -- and a comparison that
    stopped where the sample does would read a truncation as no loss at all.
    """
    kept = max(1, int(signal.size * keep))
    played = np.zeros(signal.size, dtype=np.float64)
    played[:kept] = signal[:kept]
    return played


def _band(edge: float) -> Callable[[Recording], Signal]:
    """A brickwall at ``edge``, which costs a recording band without costing it a rate."""

    def _apply(recording: Recording) -> Signal:
        return bandlimit(recording.signal, recording.sample_rate, 0.0, edge)

    return _apply


def _trim(keep: float) -> Callable[[Recording], Signal]:
    """A span cut to ``keep`` of its length, which is a note ending before its material does."""

    def _apply(recording: Recording) -> Signal:
        return _trimmed(recording.signal, keep)

    return _apply


def _click(wraps: int) -> Callable[[Recording], Signal]:
    """``wraps`` discontinuities stepped into a span, which is what a badly landed wrap sounds like."""

    def _apply(recording: Recording) -> Signal:
        return _clicked(recording.signal, wraps)

    return _apply


def _encoder(config: OptiConfig, settings: CurveSettings) -> Callable[..., Callable[[Recording], Signal]]:
    """A factory for the degradations the codec itself makes, held to one config and one curve reading."""
    plain = PlayedCurve(NO_ENVELOPE, _TEMPO)

    def _encoded(rate: int, depth: int, *, compress: bool, carrier: bool) -> Callable[[Recording], Signal]:
        def _apply(recording: Recording) -> Signal:
            params = EncodingParams(target_rate=rate, depth_bits=depth, trim_s=None, compress=compress)
            return _stored(recording, params, config, _curve(recording, settings) if carrier else plain)

        return _apply

    return _encoded


def degradations(config: OptiConfig, rates: Sequence[int]) -> tuple[Degradation, ...]:
    """Every way of spending fewer bytes the bench measures, one per axis and setting.

    The codec-shaped ones run through the very encoder the pipeline stores with, so what they measure is
    what a run would land on. The bands, the trims and the clicks are built by hand: no encoding a run
    offers produces them, and what they are for is reading what the composite makes of a fault the codec
    never causes and of band loss held apart from the rate that usually carries it.
    """
    target = export_target(config.export.tracker)
    grid = envelope_grid(target, tempo=_TEMPO, release_s=config.export.envelope.release_s)
    encoded = _encoder(config, CurveSettings(config=config.encode, target=target, grid=grid))
    top = max(rates)

    ladder = [Degradation("rate", f"{rate}", encoded(rate, 16, compress=False, carrier=False)) for rate in rates]
    for depth in _DEPTHS:
        ladder.append(Degradation("depth", f"{depth}", encoded(top, depth, compress=False, carrier=False)))
        ladder.append(Degradation("carrier", f"{depth}", encoded(top, depth, compress=False, carrier=True)))

    ladder.append(Degradation("compress", "8", encoded(top, 8, compress=True, carrier=False)))
    ladder += [Degradation("band", f"{edge:.0f}", _band(edge)) for edge in _BANDS_HZ]
    ladder += [Degradation("trim", f"{keep:g}", _trim(keep)) for keep in _KEEP]
    ladder += [Degradation("click", f"{wraps}", _click(wraps)) for wraps in _CLICKS]
    return tuple(ladder)


def measure(recordings: Sequence[Recording], ladder: Sequence[Degradation], config: OptiConfig) -> list[BenchRecord]:
    """Score every degradation of every recording against the recording it was made from."""
    composite = build_composite(config.analysis.metrics)
    measured: list[BenchRecord] = []
    for recording in recordings:
        for degradation in ladder:
            report = evaluate(recording.signal, degradation.apply(recording), recording.sample_rate, composite)
            measured.append(
                BenchRecord(
                    instrument_id=recording.instrument_id,
                    material_class=recording.material_class,
                    key=str(recording.key),
                    axis=degradation.axis,
                    label=degradation.label,
                    fidelity=report.fidelity,
                    breakdown=dict(report.breakdown),
                    segmental_snr_db=report.diagnostics["segmental_snr_db"],
                    si_sdr_db=report.diagnostics["si_sdr_db"],
                    loudness_delta_lu=report.diagnostics["loudness_delta_lu"],
                )
            )

    return measured


def write(records: Sequence[BenchRecord], path: Path) -> None:
    """Write the bench's readings as one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")


def read(path: Path) -> list[BenchRecord]:
    """Every reading the bench left at ``path``."""
    return [BenchRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
