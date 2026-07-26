import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from notebooks.utils.views import Row
from optisample.config.dsp import StftParams
from optisample.dsp.spectral import stft_magnitude

Signal = NDArray[np.float64]

_CONTRIBUTION_KEYS = ("mrstft", "logmel_l1", "spectral_shape", "mcd")


def waveform_figure(signal: Signal, sample_rate: int, *, title: str | None = None) -> Figure:
    """Amplitude-vs-time plot of a single signal."""
    data = np.asarray(signal, dtype=np.float64)
    times = np.arange(data.size) / sample_rate if sample_rate else np.arange(data.size, dtype=np.float64)
    figure = Figure(figsize=(8.0, 2.2))
    axes = figure.add_subplot()
    axes.plot(times, data, linewidth=0.6)
    axes.set_xlabel("time (s)")
    axes.set_ylabel("amplitude")
    axes.set_xlim(0.0, float(times[-1]) if data.size else 1.0)
    if title:
        axes.set_title(title)
    figure.set_layout_engine("tight")
    return figure


def spectrogram_figure(
    signal: Signal,
    sample_rate: int,
    *,
    params: StftParams,
    dynamic_range_db: float,
    title: str | None = None,
) -> Figure:
    """Log-magnitude spectrogram on a **log-frequency** axis, in dB relative to the peak (floored below).

    The floor is ``dynamic_range_db`` below the peak — the same depth the log-spectral metrics use. A
    logarithmic frequency axis matches musical pitch (octaves are evenly spaced). It is drawn with
    ``pcolormesh`` rather than ``imshow`` because ``imshow`` can only map a linear axis; the DC bin is
    dropped so the axis can be logarithmic (``log 0`` is undefined).
    """
    magnitude = stft_magnitude(np.asarray(signal, dtype=np.float64), params)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    floor = max(peak * 10.0 ** (-dynamic_range_db / 20.0), 1e-12)
    decibels = 20.0 * np.log10(np.maximum(magnitude, floor) / (peak if peak > 0.0 else 1.0))
    nyquist = sample_rate / 2.0 if sample_rate else float(magnitude.shape[1])
    freqs = np.fft.rfftfreq(params.n_fft, 1.0 / sample_rate) if sample_rate else np.arange(magnitude.shape[1] + 1.0)
    times = np.arange(magnitude.shape[0]) * params.hop_length / (sample_rate or 1)
    figure = Figure(figsize=(8.0, 3.0))
    axes = figure.add_subplot()
    mesh = axes.pcolormesh(
        times, freqs[1:], decibels[:, 1:].T, vmin=-dynamic_range_db, vmax=0.0, cmap="magma", shading="nearest"
    )
    axes.set_yscale("log")
    axes.set_ylim(float(freqs[1]), nyquist)
    axes.set_xlabel("time (s)")
    axes.set_ylabel("frequency (Hz)")
    figure.colorbar(mesh, ax=axes, label="dB rel. peak")
    if title:
        axes.set_title(title)
    figure.set_layout_engine("tight")
    return figure


def contribution_bar(comparisons: list[Row], weights: dict[str, float], *, title: str | None = None) -> Figure:
    """Stacked weighted per-metric contributions; each bar's total height ≈ that candidate's fidelity."""
    labels = [str(row["candidate"]) for row in comparisons]
    positions = np.arange(len(comparisons), dtype=np.float64)
    figure = Figure(figsize=(8.0, 3.2))
    axes = figure.add_subplot()
    bottom = np.zeros(len(comparisons), dtype=np.float64)
    for key in _CONTRIBUTION_KEYS:
        weight = weights.get(key, 0.0)
        values = np.array([weight * float(row[key]) for row in comparisons], dtype=np.float64)
        axes.bar(positions, values, bottom=bottom, label=key)
        bottom += values
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=20.0, ha="right")
    axes.set_ylabel("weighted contribution")
    axes.legend(loc="upper left", fontsize="small")
    axes.set_title(title or "Composite fidelity decomposition")
    figure.set_layout_engine("tight")
    return figure


def figure_png(figure: Figure, *, dpi: int = 110) -> bytes:
    """Render a figure to PNG bytes via Agg — display in marimo with ``mo.image(...)``."""
    FigureCanvasAgg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()
