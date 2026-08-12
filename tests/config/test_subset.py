from __future__ import annotations

import pytest

from optisample.config.subset import SubsetConfig


def test_a_floor_of_zero_admits_whatever_the_source_recorded() -> None:
    assert SubsetConfig(min_duration_s=0.0).min_duration_s == 0.0


def test_a_floor_standing_under_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        SubsetConfig(min_duration_s=-0.1)
