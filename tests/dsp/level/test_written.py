from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.level import Clock, Level, constant_level, loudest_db, written_level
from optisample.dsp.series import Readings, Series

_SPAN_S = 2.0
_EXACT_DB = 1e-9  # decibels two levels stand apart while stating the same thing


def _falling(*, from_db: float, to_db: float) -> Level:
    """A level running straight from ``from_db`` down to ``to_db``, turning only at its two ends."""
    return Level(
        readings=Readings(
            values=np.asarray([from_db, to_db], dtype=np.float64),
            seconds=np.asarray([0.0, _SPAN_S], dtype=np.float64),
        ),
        clock=Clock.PLAYED,
    )


def _moments() -> Series:
    """Moments spread across the stretch a shape is compared over."""
    return np.linspace(0.0, _SPAN_S, num=9, dtype=np.float64)


def test_a_written_shape_only_attenuates() -> None:
    """A tracker's envelope holds steps up to a full one, so the loudest moment it states is unity."""
    written = written_level(_falling(from_db=-3.0, to_db=-24.0), reference_db=-3.0)
    assert written.shape.peak_db == pytest.approx(0.0)
    assert np.all(written.shape.db(_moments()) <= 0.0)


def test_a_written_shape_stood_back_up_is_the_level_it_came_from() -> None:
    level = _falling(from_db=-3.0, to_db=-24.0)
    restored = written_level(level, reference_db=6.0).absolute
    assert restored.db(_moments()) == pytest.approx(level.db(_moments()), abs=_EXACT_DB)


def test_a_level_rising_above_its_reference_has_no_step_to_state_it() -> None:
    with pytest.raises(ValueError, match="attenuates"):
        written_level(_falling(from_db=0.0, to_db=-24.0), reference_db=-3.0)


def test_two_levels_written_against_one_reference_stay_as_far_apart_as_they_stand() -> None:
    """One reference across a plan is what keeps the balance two recordings hold from moving."""
    loud = _falling(from_db=0.0, to_db=-20.0)
    soft = _falling(from_db=-12.0, to_db=-32.0)
    reference = loudest_db([loud, soft])
    apart = written_level(loud, reference_db=reference).shape.db(_moments()) - written_level(
        soft, reference_db=reference
    ).shape.db(_moments())
    assert apart == pytest.approx(12.0, abs=_EXACT_DB)


def test_the_reference_a_set_is_written_against_is_the_loudest_moment_any_of_them_reaches() -> None:
    assert loudest_db([constant_level(-6.0, Clock.PLAYED), constant_level(-2.0, Clock.PLAYED)]) == pytest.approx(-2.0)


def test_a_set_holding_nothing_names_unity_itself() -> None:
    assert loudest_db([]) == pytest.approx(0.0)
