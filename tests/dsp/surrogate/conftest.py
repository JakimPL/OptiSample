"""Shared signal generator for the surrogate-codec tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

SR = 44_100


@pytest.fixture
def sine() -> Callable[..., NDArray[np.float64]]:
    """Factory: a pure sine at ``SR`` (default 440 Hz, 1 s, unit amplitude)."""

    def _sine(freq: float = 440.0, dur: float = 1.0, amp: float = 1.0) -> NDArray[np.float64]:
        times = np.arange(int(dur * SR), dtype=np.float64) / SR
        return amp * np.sin(2.0 * np.pi * freq * times)

    return _sine
