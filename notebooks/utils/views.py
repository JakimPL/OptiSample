"""Turn samples, instruments, and comparisons into plain table rows for the notebook to render.

Everything here returns built-in ``dict``/``list`` values (no plotting, no disk access beyond the
signals handed in), so it is straightforward to unit-test and the notebook stays a thin display
layer. The comparison rows mirror the P1 smoke-test columns on purpose: fidelity, the per-metric
breakdown, and the interpretable diagnostics side by side.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from notebooks.utils.degrade import DegradeSpec, apply
from notebooks.utils.loading import sample_label
from optisample.config.dsp import SpectralConfig
from optisample.dsp.spectral import spectral_centroid, spectral_flatness, spectral_rolloff
from optisample.metrics import CompositeFidelity, SampleSize, bytes_to_kib, evaluate, integrated_loudness, kib_to_bytes
from optisample.metrics.size import INSTRUMENT_HEADER_BYTES
from optisample.model import InstrumentSpec, SourceSample

Signal = NDArray[np.float64]

Row = dict[str, float | int | str | bool]


def _cc_summary(cc_averages: dict[int, float]) -> str:
    """Compact ``cc:value`` listing for a table cell (empty when no controllers are tracked)."""
    return " ".join(f"{cc}:{value:g}" for cc, value in sorted(cc_averages.items()))


def sample_row(sample: SourceSample, signal: Signal, sample_rate: int, spectral: SpectralConfig) -> Row:
    """Descriptors + storage footprint for one sample (a single table row)."""
    data = np.asarray(signal, dtype=np.float64)
    frames = int(data.size)
    peak = float(np.max(np.abs(data))) if frames else 0.0
    rms = float(np.sqrt(np.mean(data**2))) if frames else 0.0
    return {
        "sample": sample_label(sample),
        "pitch": sample.pitch,
        "velocity": sample.velocity,
        "cc": _cc_summary(sample.cc_averages),
        "dur_s": frames / sample_rate if sample_rate else 0.0,
        "frames": frames,
        "peak": peak,
        "rms": rms,
        "lufs": integrated_loudness(data, sample_rate),
        "centroid_hz": spectral_centroid(data, sample_rate, spectral.stft),
        "rolloff_hz": spectral_rolloff(data, sample_rate, spectral.stft, spectral.rolloff_percent),
        "flatness": spectral_flatness(data, spectral.stft),
        "kib_16": bytes_to_kib(SampleSize(frames, 16).total_bytes),
        "kib_8": bytes_to_kib(SampleSize(frames, 8).total_bytes),
    }


def material_rows(instrument: InstrumentSpec) -> list[Row]:
    """One row per note event the song plays through this instrument."""
    events = instrument.material or []
    return [
        {
            "pitch": event.pitch,
            "velocity": event.velocity,
            "cc": _cc_summary(event.cc_averages),
            "dur_s": event.duration_s,
            "count": event.count,
            "weight": event.weight,
        }
        for event in events
    ]


def stored_bytes(frame_counts: Sequence[int], depth_bits: int) -> int:
    """Uncompressed bytes for one instrument at ``depth_bits``: its samples + its instrument header."""
    return sum(SampleSize(int(f), depth_bits).total_bytes for f in frame_counts) + INSTRUMENT_HEADER_BYTES


def budget_summary(instrument: InstrumentSpec, frame_counts: Sequence[int]) -> Row:
    """How far full-length 16-/8-bit storage is over (or under) the instrument's byte budget."""
    budget = kib_to_bytes(instrument.budget_kb)
    bytes_16 = stored_bytes(frame_counts, 16)
    bytes_8 = stored_bytes(frame_counts, 8)
    return {
        "instrument": instrument.id,
        "n_samples": len(frame_counts),
        "budget_kib": bytes_to_kib(budget),
        "stored_kib_16": bytes_to_kib(bytes_16),
        "stored_kib_8": bytes_to_kib(bytes_8),
        "over_ratio_16": bytes_16 / budget if budget else float("inf"),
        "over_ratio_8": bytes_8 / budget if budget else float("inf"),
        "fits_16": bytes_16 <= budget,
        "fits_8": bytes_8 <= budget,
    }


def compare(reference: Signal, candidate: Signal, sample_rate: int, label: str, composite: CompositeFidelity) -> Row:
    """Evaluate ``candidate`` against ``reference`` and flatten it into one comparison row."""
    report = evaluate(reference, candidate, sample_rate, composite)
    breakdown, diagnostics = report.breakdown, report.diagnostics
    return {
        "candidate": label,
        "fidelity": report.fidelity,
        "mrstft": breakdown["mrstft"],
        "logmel_l1": breakdown["logmel_l1"],
        "mcd": breakdown["mcd"],
        "spectral_shape": breakdown["spectral_shape"],
        "snr_db": diagnostics["snr_db"],
        "seg_snr_db": diagnostics["segmental_snr_db"],
        "si_sdr_db": diagnostics["si_sdr_db"],
        "loudness_dLU": diagnostics["loudness_delta_lu"],
    }


def compare_specs(
    reference: Signal, sample_rate: int, specs: Sequence[DegradeSpec], composite: CompositeFidelity
) -> list[Row]:
    """Apply each degradation to ``reference`` and return one comparison row per spec."""
    return [
        compare(reference, apply(spec, reference, sample_rate), sample_rate, spec.label, composite) for spec in specs
    ]
