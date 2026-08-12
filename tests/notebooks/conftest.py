from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from notebooks.utils import loading, viz
from optisample.config import OptiConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.spectral import SpectralConfig
from optisample.model import Manifest

SR = 44_100

Demo = tuple[Path, Manifest]


@pytest.fixture
def spectrogram_style(spectral_config: SpectralConfig, metrics_config: MetricsConfig) -> viz.SpectrogramStyle:
    """The reading every spectrogram in these tests is drawn under, taken off the shipped config."""
    return viz.SpectrogramStyle(
        params=spectral_config.stft, dynamic_range_db=metrics_config.preprocess.dynamic_range_db
    )


@pytest.fixture
def tone() -> Callable[..., NDArray[np.float64]]:
    """Factory: a pure sine at ``SR`` (default 440 Hz, 1 s, 0.5 amplitude)."""

    def _tone(freq: float = 440.0, duration: float = 1.0, amplitude: float = 0.5) -> NDArray[np.float64]:
        times = np.arange(int(duration * SR)) / SR
        return amplitude * np.sin(2.0 * np.pi * freq * times)

    return _tone


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory, config: OptiConfig) -> Demo:
    root = tmp_path_factory.mktemp("demo")
    manifest = loading.load(loading.ensure_demo(root, config.synth))
    return root, manifest
