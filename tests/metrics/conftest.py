"""Shared signal degradations for the metrics tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def quantize() -> Callable[..., NDArray[np.float64]]:
    """Factory: snap a signal to a ``bits``-deep uniform grid (the reference degradation)."""

    def _quantize(signal: NDArray[np.float64], bits: int) -> NDArray[np.float64]:
        step = 2.0 / (2**bits)
        return np.round(signal / step) * step

    return _quantize
