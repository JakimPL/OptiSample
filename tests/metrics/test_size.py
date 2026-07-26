from __future__ import annotations

import pytest

from optisample.metrics.size import bytes_to_kib, kib_to_bytes


def test_kib_round_trip() -> None:
    assert kib_to_bytes(128.0) == 131072
    assert bytes_to_kib(131072) == pytest.approx(128.0)


def test_a_fractional_kib_budget_rounds_up_to_whole_bytes() -> None:
    assert kib_to_bytes(0.5) == 512
    assert kib_to_bytes(1.0009) == 1025  # a budget covers everything that was asked for
