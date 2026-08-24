from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from notebooks.utils import degrade, views, viz
from optisample.config import OptiConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.spectral import SpectralConfig
from optisample.metrics import CompositeFidelity

SR = 44_100

_DECLINE_LINES = 3  # the readings, the curve fitted through them and the straight line drawn against both
_REACHED = 4  # depths the fixture take arrives at, which is what it draws a line apiece for


def test_waveform_and_spectrogram_render_to_png(
    spectral_config: SpectralConfig, metrics_config: MetricsConfig, tone: Callable[..., NDArray[np.float64]]
) -> None:
    signal = tone()
    style = viz.SpectrogramStyle(
        params=spectral_config.stft, dynamic_range_db=metrics_config.preprocess.dynamic_range_db
    )
    figures = (
        viz.waveform_figure(signal, SR),
        viz.spectrogram_figure(signal, SR, style=style),
    )
    for figure in figures:
        png = viz.figure_png(figure)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_decline_is_drawn_against_the_line_and_the_curve_fitted_to_it(
    tone: Callable[..., NDArray[np.float64]], config: OptiConfig
) -> None:
    """All three read the same level curve, so what stands between them is what the curvature states."""
    signal = tone(duration=1.5) * np.exp(-np.linspace(0.0, 3.0, int(1.5 * SR)))
    figure = viz.decline_figure(signal, SR, nodes=config.cluster.descriptor.envelope_nodes, title="falling")
    lines = figure.axes[0].get_lines()

    assert [line.get_label() for line in lines][:2] == ["level read", "fitted curve (6 corners)"]
    assert len(lines) == _DECLINE_LINES
    assert viz.figure_png(figure)[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_level_holding_steady_is_drawn_without_a_decline_to_fit(
    tone: Callable[..., NDArray[np.float64]], config: OptiConfig
) -> None:
    """Material too short for a line to be drawn through is drawn as the readings and the curve alone."""
    figure = viz.decline_figure(tone(duration=0.02), SR, nodes=config.cluster.descriptor.envelope_nodes)

    assert len(figure.axes[0].get_lines()) == _DECLINE_LINES - 1


def test_the_profile_draws_one_line_per_depth_the_recording_reached(config: OptiConfig) -> None:
    """A depth a take fell short of leaves the picture, so every line drawn is a reading it truly holds."""
    depths = config.cluster.descriptor.anchor_depths_db
    sustain = np.tile(np.linspace(-6.0, 6.0, 12), (len(depths), 1))
    reached = np.array([index < _REACHED for index in range(len(depths))], dtype=np.bool_)
    figure = viz.profile_figure(sustain, depths_db=depths, reached=reached, title="at each depth")

    assert len(figure.axes[0].get_lines()) == _REACHED
    assert viz.figure_png(figure)[:8] == b"\x89PNG\r\n\x1a\n"


def test_contribution_bar_totals_match_fidelity(
    composite: CompositeFidelity, metrics_config: MetricsConfig, tone: Callable[..., NDArray[np.float64]]
) -> None:
    """A bar states the whole of a candidate's score, so its stack sums to the fidelity beside it.

    The terms are the ones the composite was built from, so a term added to the objective reaches the
    picture without the drawing being told about it.
    """
    signal = tone()
    weights = metrics_config.weights
    rows = views.compare_specs(signal, SR, degrade.smoke_specs(), composite)
    for row in rows:
        stacked = sum(weight * float(row[key]) for key, weight in weights.items())
        assert stacked == pytest.approx(float(row["fidelity"]), rel=1e-9, abs=1e-9)
    assert viz.figure_png(viz.contribution_bar(rows, weights))[:8] == b"\x89PNG\r\n\x1a\n"
