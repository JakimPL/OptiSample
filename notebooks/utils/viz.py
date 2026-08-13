import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from notebooks.utils.views import Row
from optisample.config.spectral import StftParams
from optisample.dsp.level import decay_trend, level_readings
from optisample.dsp.piecewise import fit_piecewise
from optisample.dsp.spectral import stft_magnitude
from optisample.dsp.trajectory import reading_window_s

Signal = NDArray[np.float64]

_CONTRIBUTION_KEYS = ("mrstft", "logmel_l1", "spectral_shape", "mcd")


@dataclass(frozen=True)
class SpectrogramStyle:
    """The two readings a spectrogram is drawn under: the transform it is taken with and the depth it states.

    Carrying the pair as one value lets a caller holding a config settle it once and hand the same
    reading to every spectrogram it draws, so the panels of one notebook are read on one scale.
    """

    params: StftParams
    dynamic_range_db: float


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
    style: SpectrogramStyle,
    title: str | None = None,
) -> Figure:
    """Log-magnitude spectrogram on a **log-frequency** axis, in dB relative to the peak (floored below).

    The floor is ``style.dynamic_range_db`` below the peak — the same depth the log-spectral metrics use.
    A logarithmic frequency axis matches musical pitch (octaves are evenly spaced). It is drawn with
    ``pcolormesh`` rather than ``imshow`` because ``imshow`` can only map a linear axis; the DC bin is
    dropped so the axis can be logarithmic (``log 0`` is undefined).
    """
    magnitude = stft_magnitude(np.asarray(signal, dtype=np.float64), style.params)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    floor = max(peak * 10.0 ** (-style.dynamic_range_db / 20.0), 1e-12)
    decibels = 20.0 * np.log10(np.maximum(magnitude, floor) / (peak if peak > 0.0 else 1.0))
    nyquist = sample_rate / 2.0 if sample_rate else float(magnitude.shape[1])
    freqs = (
        np.fft.rfftfreq(style.params.n_fft, 1.0 / sample_rate) if sample_rate else np.arange(magnitude.shape[1] + 1.0)
    )
    times = np.arange(magnitude.shape[0]) * style.params.hop_length / (sample_rate or 1)
    figure = Figure(figsize=(8.0, 3.0))
    axes = figure.add_subplot()
    mesh = axes.pcolormesh(
        times, freqs[1:], decibels[:, 1:].T, vmin=-style.dynamic_range_db, vmax=0.0, cmap="magma", shading="nearest"
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


def rd_scatter(rows: list[Row], *, title: str | None = None) -> Figure:
    """Where every kept item landed on the rate-distortion plane, marked by the depth it was stored at.

    A plan is one point per item; reading them together says which keys the budget bought resolution
    for and which it left carrying the distortion, and the depth split shows where the allocation
    judged bandwidth worth more than bit depth.
    """
    figure = Figure(figsize=(8.0, 3.4))
    axes = figure.add_subplot()
    for depth in sorted({int(row["depth"]) for row in rows}):
        at_depth = [row for row in rows if int(row["depth"]) == depth]
        axes.scatter(
            [float(row["kib"]) for row in at_depth],
            [float(row["distortion"]) for row in at_depth],
            s=26.0,
            alpha=0.8,
            label=f"{depth}-bit",
        )
    axes.set_xlabel("stored size (KiB)")
    axes.set_ylabel("distortion")
    axes.set_xscale("log")
    axes.legend(loc="upper right", fontsize="small")
    axes.set_title(title or "Rate-distortion of the kept items")
    figure.set_layout_engine("tight")
    return figure


def objective_bar(rows: list[Row], *, title: str | None = None) -> Figure:
    """Each pitch's share of the objective, in keyboard order, so the costly keys stand out by name."""
    labels = [str(row["note"]) for row in rows]
    positions = np.arange(len(rows), dtype=np.float64)
    figure = Figure(figsize=(9.0, 3.2))
    axes = figure.add_subplot()
    axes.bar(positions, [float(row["objective"]) for row in rows])
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=90.0, fontsize="x-small")
    axes.set_ylabel("weighted distortion")
    axes.set_title(title or "Objective contribution per pitch")
    figure.set_layout_engine("tight")
    return figure


def decline_figure(signal: Signal, sample_rate: int, *, nodes: int, title: str | None = None) -> Figure:
    """A recording's level against the straight decline fitted to it and the curve fitted through it.

    All three are read in decibels, the domain a ringing note falls straight in, so the gap between the
    readings and the line is what ``curvature_db`` states as one number: a note holding its own line runs
    flat against it, and one that races away and then settles bows off it.
    """
    readings = level_readings(signal, sample_rate, window_s=reading_window_s(signal.size, sample_rate))
    curve = fit_piecewise(readings, nodes=nodes)
    trend = decay_trend(signal, sample_rate)
    figure = Figure(figsize=(8.0, 2.6))
    axes = figure.add_subplot()
    axes.plot(readings.seconds, readings.values, linewidth=1.0, label="level read")
    axes.plot(readings.seconds, curve.at(readings.seconds), linewidth=1.4, label=f"fitted curve ({nodes} corners)")
    if trend is not None:
        line = trend.mean_db + trend.slope_db * (readings.seconds - trend.mean_s)
        axes.plot(readings.seconds, line, linewidth=1.2, linestyle="--", label=f"decline {trend.slope_db:.1f} dB/s")

    axes.set_xlabel("time (s)")
    axes.set_ylabel("level (dB)")
    axes.legend(loc="upper right", fontsize="small")
    if title:
        axes.set_title(title)

    figure.set_layout_engine("tight")
    return figure


def profile_figure(
    sustain: Signal,
    *,
    depths_db: Sequence[float],
    reached: NDArray[np.bool_],
    title: str | None = None,
) -> Figure:
    """What a recording sounded like at each depth of its own decline, one line per depth it arrived at.

    Reading the lines together says how a note's timbre changes as it falls away: partials that drop out
    early pull their line down as the decline deepens, and a sound that holds its balance keeps its lines
    together. The values stand past each frame's own mean, so the picture is the balance itself rather than
    the level the note was played at.
    """
    figure = Figure(figsize=(8.0, 2.8))
    axes = figure.add_subplot()
    columns = np.arange(sustain.shape[1])
    for index, depth in enumerate(depths_db):
        if bool(reached[index]):
            axes.plot(columns, sustain[index], linewidth=1.1, label=f"-{depth:g} dB")

    axes.set_xlabel("reading")
    axes.set_ylabel("level past the frame's mean (dB)")
    axes.legend(loc="upper right", fontsize="x-small", ncols=2)
    if title:
        axes.set_title(title)

    figure.set_layout_engine("tight")
    return figure


def figure_png(figure: Figure, *, dpi: int = 110) -> bytes:
    """Render a figure to PNG bytes via Agg — display in marimo with ``mo.image(...)``."""
    FigureCanvasAgg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()
