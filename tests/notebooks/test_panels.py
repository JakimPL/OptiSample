from collections.abc import Callable
from pathlib import Path

import marimo as mo
import numpy as np
from numpy.typing import NDArray

from notebooks.utils import panels, viz
from optisample.io.audio import write_wav

SR = 44_100

PNG_URI = "data:image/png"


def test_player_titles_the_clip_and_carries_its_audio(tone: Callable[..., NDArray[np.float64]]) -> None:
    html = panels.player(tone(), SR, label="original", normalize=True)
    assert isinstance(html, mo.Html)
    assert "original" in html.text
    assert "<audio" in html.text


def test_file_player_reads_the_recording_beside_it(tmp_path: Path, tone: Callable[..., NDArray[np.float64]]) -> None:
    path = tmp_path / "take.wav"
    write_wav(path, tone(), SR)
    html = panels.file_player(path, label="take", normalize=False)
    assert "take" in html.text
    assert "<audio" in html.text


def test_every_figure_helper_embeds_a_png(
    tone: Callable[..., NDArray[np.float64]], spectrogram_style: viz.SpectrogramStyle
) -> None:
    signal = tone()
    images = (
        panels.image(viz.waveform_figure(signal, SR)),
        panels.waveform(signal, SR, title="waveform"),
        panels.spectrogram(signal, SR, style=spectrogram_style, title="spectrogram"),
    )
    for html in images:
        assert PNG_URI in html.text


def test_signal_panel_states_the_stretch_it_read_and_shows_both_readings(
    tmp_path: Path, tone: Callable[..., NDArray[np.float64]], spectrogram_style: viz.SpectrogramStyle
) -> None:
    path = tmp_path / "reference.wav"
    write_wav(path, tone(duration=0.5), SR)
    html = panels.signal_panel(path, label="reference", style=spectrogram_style, normalize=False)
    assert "reference" in html.text
    assert f"0.50s at {SR} Hz" in html.text
    assert "<audio" in html.text
    assert PNG_URI in html.text


def test_table_lists_every_row_and_leaves_selection_off() -> None:
    rows: list[dict[str, float | int | str | bool]] = [{"key": "A4", "kib": 1.0}, {"key": "B4", "kib": 2.0}]
    listed = panels.table(rows, page_size=1)
    assert listed.data == rows
    assert listed.value is None
