"""Shared fixtures for the notebook-helper tests: a demo project and a tone generator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from notebooks.utils import loading
from optisample.config import OptiConfig
from optisample.model import Manifest

SR = 44_100

Demo = tuple[Path, Manifest]


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
