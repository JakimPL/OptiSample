from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.optimize.velocity_map import derive_velocity_map, loudness_by_velocity

SR = 44_100


def sine(amp: float, freq: float = 220.0, dur: float = 0.6) -> NDArray[np.float64]:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def test_loudness_by_velocity_groups_and_orders_by_level() -> None:
    loud = loudness_by_velocity([(40, sine(0.1)), (80, sine(0.4)), (115, sine(1.0))], SR)
    assert set(loud) == {40, 80, 115}
    assert loud[40] < loud[80] < loud[115]


def test_loudness_by_velocity_averages_duplicates() -> None:
    loud = loudness_by_velocity([(80, sine(0.5)), (80, sine(0.5))], SR)
    single = loudness_by_velocity([(80, sine(0.5))], SR)
    assert loud[80] == pytest.approx(single[80])


def test_map_matches_amplitude_ratios() -> None:
    # Same sine at 0.1 / 0.4 / 1.0 → loudness differs by exactly 20*log10(ratio),
    # so volume = round(64 * ratio): 6.4→6, 25.6→26, 64→64.
    vmap = derive_velocity_map(loudness_by_velocity([(40, sine(0.1)), (80, sine(0.4)), (115, sine(1.0))], SR))
    assert vmap.volume(115) == 64
    assert vmap.volume(80) == 26
    assert vmap.volume(40) == 6


def test_map_is_monotone_bounded_and_full_length() -> None:
    vmap = derive_velocity_map(loudness_by_velocity([(30, sine(0.2)), (70, sine(0.5)), (120, sine(1.0))], SR))
    assert len(vmap.volumes) == 128
    assert all(0 <= v <= 64 for v in vmap.volumes)
    assert all(a <= b for a, b in zip(vmap.volumes, vmap.volumes[1:]))  # non-decreasing in velocity


def test_flat_extrapolation_beyond_anchors() -> None:
    vmap = derive_velocity_map(loudness_by_velocity([(40, sine(0.25)), (100, sine(1.0))], SR))
    assert vmap.volume(127) == vmap.volume(100)  # above the loudest anchor → held flat
    assert vmap.volume(0) == vmap.volume(40)  # below the quietest anchor → held flat


def test_single_velocity_is_full_volume_everywhere() -> None:
    vmap = derive_velocity_map(loudness_by_velocity([(64, sine(0.5))], SR))
    assert set(vmap.volumes) == {64}


def test_all_silent_maps_to_zero() -> None:
    silence = np.zeros(int(0.6 * SR), dtype=np.float64)
    vmap = derive_velocity_map(loudness_by_velocity([(40, silence), (100, silence)], SR))
    assert set(vmap.volumes) == {0}
    assert all(anchor.volume == 0 for anchor in vmap.anchors)


def test_anchors_record_the_measured_volume() -> None:
    vmap = derive_velocity_map(loudness_by_velocity([(50, sine(0.5)), (100, sine(1.0))], SR))
    for anchor in vmap.anchors:
        assert anchor.volume == vmap.volume(anchor.velocity)


def test_derive_requires_measurements() -> None:
    with pytest.raises(ValueError):
        derive_velocity_map({})


def test_volume_rejects_out_of_range_velocity() -> None:
    vmap = derive_velocity_map(loudness_by_velocity([(64, sine(0.5))], SR))
    with pytest.raises(ValueError):
        vmap.volume(128)
    with pytest.raises(ValueError):
        vmap.volume(-1)
