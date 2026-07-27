from typing import Any

import pytest
from pydantic import ValidationError

from optisample.config.dynamics import DynamicsConfig


def raw(**overrides: Any) -> dict[str, Any]:
    """A valid dynamics payload with the named fields overridden."""
    base: dict[str, Any] = {
        "threshold_db": -18.0,
        "ratio": 3.0,
        "knee_db": 6.0,
        "rms_window_s": 0.01,
        "gain_smoothing_s": 0.025,
    }
    return {**base, **overrides}


def test_the_bundled_shape_validates() -> None:
    assert DynamicsConfig.model_validate(raw()).ratio == 3.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"ratio": 0.5},  # a ratio under one would expand rather than compress
        {"knee_db": -1.0},
        {"rms_window_s": 0.0},  # a detector has to average over some span
        {"gain_smoothing_s": 0.0},
        {"threshold_db": None},
        {"attack_s": 0.005},  # symmetric ballistics, so there is no separate attack to set
    ],
)
def test_a_setting_outside_the_schema_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DynamicsConfig.model_validate(raw(**overrides))


def test_one_to_one_is_allowed_as_the_setting_that_compresses_nothing() -> None:
    assert DynamicsConfig.model_validate(raw(ratio=1.0)).ratio == 1.0


def test_a_square_knee_is_allowed_as_the_setting_that_bends_at_the_threshold_alone() -> None:
    assert DynamicsConfig.model_validate(raw(knee_db=0.0)).knee_db == 0.0
