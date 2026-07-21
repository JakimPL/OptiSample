from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.surrogate import StoredSample
from optisample.metrics.size import SampleSize


def test_stored_sample_properties() -> None:
    stored = StoredSample(pcm=np.zeros(1000, dtype=np.float64), sample_rate=22_050, depth_bits=8, root_pitch=60)
    assert stored.frames == 1000
    assert stored.size == SampleSize(frames=1000, depth_bits=8)
    assert stored.stored_bytes == SampleSize(frames=1000, depth_bits=8).total_bytes
    assert stored.duration_s == pytest.approx(1000 / 22_050)
