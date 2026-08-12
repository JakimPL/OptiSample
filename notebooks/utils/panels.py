from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import marimo as mo
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from notebooks.utils.audioio import to_wav_bytes
from notebooks.utils.views import Row
from notebooks.utils.viz import (
    SpectrogramStyle,
    figure_png,
    spectrogram_figure,
    waveform_figure,
)
from optisample.io.audio import read_wav

Signal = NDArray[np.float64]


def image(figure: Figure) -> mo.Html:
    """``figure`` as the raster a notebook displays it as.

    Figures are built through the object-oriented API and rasterized here, so a notebook draws without an
    interactive matplotlib backend to hold open.
    """
    return mo.image(figure_png(figure))


def table(rows: Sequence[Row], *, page_size: int | None = None) -> mo.ui.table:
    """``rows`` as the read-only table a panel lists its findings in.

    Selection stays off because these tables report what a stage decided; a panel offering a choice makes
    it through a control of its own, which keeps what is read apart from what is picked. Naming no
    ``page_size`` takes the table's own.
    """
    return mo.ui.table(list(rows), selection=None, page_size=page_size)


def player(signal: Signal, sample_rate: int, *, label: str, normalize: bool) -> mo.Html:
    """``signal`` as a titled player, encoded to WAV in memory.

    Normalizing makes a quiet clip audible at the cost of the level it was played at, so the caller
    states which of the two its panel is asking about.
    """
    return mo.vstack(
        [
            mo.md(f"**{label}**"),
            mo.audio(io.BytesIO(to_wav_bytes(signal, sample_rate, normalize=normalize))),
        ]
    )


def file_player(path: Path, *, label: str, normalize: bool) -> mo.Html:
    """The recording at ``path`` as a titled player."""
    signal, sample_rate = read_wav(path)
    return player(signal, sample_rate, label=label, normalize=normalize)


def waveform(signal: Signal, sample_rate: int, *, title: str) -> mo.Html:
    """``signal``'s amplitude against time, as the image a notebook shows."""
    return image(waveform_figure(signal, sample_rate, title=title))


def spectrogram(signal: Signal, sample_rate: int, *, style: SpectrogramStyle, title: str) -> mo.Html:
    """``signal``'s spectrogram under ``style``, as the image a notebook shows."""
    return image(spectrogram_figure(signal, sample_rate, style=style, title=title))


def signal_panel(path: Path, *, label: str, style: SpectrogramStyle, normalize: bool) -> mo.Html:
    """The recording at ``path`` heard and seen at once: how long it runs, a player, and its spectrogram.

    The two sides of a comparison built this way stand on one reading, so what is heard between them and
    what is visible between them are the same difference.
    """
    signal, sample_rate = read_wav(path)
    return mo.vstack(
        [
            mo.md(f"**{label}** — {signal.size / sample_rate:.2f}s at {sample_rate} Hz"),
            mo.audio(io.BytesIO(to_wav_bytes(signal, sample_rate, normalize=normalize))),
            spectrogram(signal, sample_rate, style=style, title=label),
        ]
    )
