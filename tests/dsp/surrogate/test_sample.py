from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.surrogate import StoredSample
from trackmod.core.samples.depth import BitDepth


def test_stored_sample_properties() -> None:
    stored = StoredSample(pcm=np.zeros(1000, dtype=np.float64), sample_rate=22_050, depth_bits=8, root_pitch=60)
    assert stored.frames == 1000
    assert stored.depth is BitDepth.EIGHT
    assert stored.duration_s == pytest.approx(1000 / 22_050)


@pytest.mark.parametrize("depth_bits", [0, 24, 32])
def test_a_depth_no_tracker_format_stores_is_rejected(depth_bits: int) -> None:
    stored = StoredSample(pcm=np.zeros(8, dtype=np.float64), sample_rate=22_050, depth_bits=depth_bits, root_pitch=60)
    with pytest.raises(ValueError):
        _ = stored.depth


def test_a_sample_with_no_rate_reports_no_duration() -> None:
    stored = StoredSample(pcm=np.zeros(8, dtype=np.float64), sample_rate=0, depth_bits=16, root_pitch=60)
    assert stored.duration_s == 0.0


def test_a_sample_stored_hot_plays_back_by_the_reciprocal_of_what_lifted_it() -> None:
    """Storing four times louder than recorded means playing a quarter as loud to sound as recorded."""
    stored = StoredSample(pcm=np.zeros(8, dtype=np.float64), sample_rate=22_050, depth_bits=8, root_pitch=60, gain=4.0)
    assert stored.playback_gain == pytest.approx(0.25)


def test_a_sample_scaled_by_nothing_plays_as_it_stands() -> None:
    stored = StoredSample(pcm=np.zeros(8, dtype=np.float64), sample_rate=22_050, depth_bits=8, root_pitch=60, gain=0.0)
    assert stored.playback_gain == 1.0
