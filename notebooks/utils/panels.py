from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Final

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

_WAV_MIME: Final = "audio/wav"


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


def _audio(clip: bytes, *, autoplay: bool) -> mo.Html:
    """``clip`` as the player a panel shows, sounding as it is drawn where the caller asks it to.

    Marimo's own player is what a panel shows when it waits to be pressed. A player asked to sound at once
    is written as the audio element the browser starts on its own, which is what makes a click on a picture
    audible without a second press; the browser plays it once the reader has touched the page.
    """
    if not autoplay:
        return mo.audio(io.BytesIO(clip))

    return mo.Html(f'<audio src="data:{_WAV_MIME};base64,{base64.b64encode(clip).decode("ascii")}" controls autoplay>')


def player(signal: Signal, sample_rate: int, *, label: str, normalize: bool, autoplay: bool) -> mo.Html:
    """``signal`` as a titled player, encoded to WAV in memory.

    Normalizing makes a quiet clip audible at the cost of the level it was played at, so the caller
    states which of the two its panel is asking about. ``autoplay`` states whether the clip sounds as the
    panel is drawn, which is what a panel answering a click on a picture asks for.
    """
    return mo.vstack(
        [
            mo.md(f"**{label}**"),
            _audio(to_wav_bytes(signal, sample_rate, normalize=normalize), autoplay=autoplay),
        ]
    )


def file_player(path: Path, *, label: str, normalize: bool, autoplay: bool) -> mo.Html:
    """The recording at ``path`` as a titled player."""
    signal, sample_rate = read_wav(path)
    return player(signal, sample_rate, label=label, normalize=normalize, autoplay=autoplay)


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
            _audio(to_wav_bytes(signal, sample_rate, normalize=normalize), autoplay=False),
            spectrogram(signal, sample_rate, style=style, title=label),
        ]
    )
