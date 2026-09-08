from __future__ import annotations

import numpy as np
import pytest
from trackmod import BitDepth

from optisample.dsp.surrogate import StoredSample


def test_stored_sample_properties() -> None:
    stored = StoredSample(pcm=np.zeros(1000, dtype=np.float64), sample_rate=22_050, depth=BitDepth.EIGHT, root_pitch=60)
    assert stored.frames == 1000
    assert stored.depth is BitDepth.EIGHT
    assert stored.duration_s == pytest.approx(1000 / 22_050)


def test_a_sample_with_no_rate_reports_no_duration() -> None:
    stored = StoredSample(pcm=np.zeros(8, dtype=np.float64), sample_rate=0, depth=BitDepth.SIXTEEN, root_pitch=60)
    assert stored.duration_s == 0.0


def test_a_sample_stored_hot_plays_back_by_the_reciprocal_of_what_lifted_it() -> None:
    """Storing four times louder than recorded means playing a quarter as loud to sound as recorded."""
    stored = StoredSample(
        pcm=np.zeros(8, dtype=np.float64), sample_rate=22_050, depth=BitDepth.EIGHT, root_pitch=60, gain=4.0
    )
    assert stored.playback_gain == pytest.approx(0.25)


def test_a_sample_scaled_by_nothing_plays_as_it_stands() -> None:
    stored = StoredSample(
        pcm=np.zeros(8, dtype=np.float64), sample_rate=22_050, depth=BitDepth.EIGHT, root_pitch=60, gain=0.0
    )
    assert stored.playback_gain == 1.0
