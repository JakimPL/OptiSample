from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from notebooks.utils import degrade, views, viz
from optisample.config.metrics import MetricsConfig
from optisample.config.spectral import SpectralConfig
from optisample.metrics import CompositeFidelity

SR = 44_100


def test_waveform_and_spectrogram_render_to_png(
    spectral_config: SpectralConfig, metrics_config: MetricsConfig, tone: Callable[..., NDArray[np.float64]]
) -> None:
    signal = tone()
    figures = (
        viz.waveform_figure(signal, SR),
        viz.spectrogram_figure(
            signal, SR, params=spectral_config.stft, dynamic_range_db=metrics_config.preprocess.dynamic_range_db
        ),
    )
    for figure in figures:
        png = viz.figure_png(figure)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_contribution_bar_totals_match_fidelity(
    composite: CompositeFidelity, metrics_config: MetricsConfig, tone: Callable[..., NDArray[np.float64]]
) -> None:
    signal = tone()
    weights = metrics_config.weights
    rows = views.compare_specs(signal, SR, degrade.smoke_specs(), composite)
    for row in rows:
        stacked = sum(weights.get(key, 0.0) * float(row[key]) for key in viz._CONTRIBUTION_KEYS)
        assert stacked == pytest.approx(float(row["fidelity"]), rel=1e-9, abs=1e-9)
    assert viz.figure_png(viz.contribution_bar(rows, weights))[:8] == b"\x89PNG\r\n\x1a\n"
