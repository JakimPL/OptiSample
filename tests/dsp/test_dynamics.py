from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dynamics import DynamicsConfig
from optisample.dsp.dynamics import compress
from optisample.dsp.level import peak_amplitude

SR = 44_100
_QUIET = 0.1  # the amplitude a scaled-down copy of a test signal peaks at

DynamicsFactory = Callable[..., DynamicsConfig]


@pytest.fixture
def dynamics(dynamics_config: DynamicsConfig) -> DynamicsFactory:
    """Factory: the bundled dynamics config with the named fields overridden (re-validated)."""

    def _build(**overrides: object) -> DynamicsConfig:
        return DynamicsConfig.model_validate({**dynamics_config.model_dump(), **overrides})

    return _build


def _crest_factor(signal: NDArray[np.float64]) -> float:
    """How far a signal's peak stands above its own average level, which is what compression narrows."""
    return peak_amplitude(signal) / float(np.sqrt(np.mean(signal**2)))


def _swelling_tone(seconds: float = 1.0) -> NDArray[np.float64]:
    """A tone rising from near-silence to full scale, so one pass covers both sides of the threshold."""
    times = np.arange(int(seconds * SR), dtype=np.float64) / SR
    return np.asarray(np.linspace(0.02, 1.0, times.size) * np.sin(2.0 * np.pi * 220.0 * times), dtype=np.float64)


# --- what compression does to a signal ------------------------------------------------------------


def test_compression_narrows_the_gap_between_the_peak_and_the_average(dynamics: DynamicsFactory) -> None:
    """The whole point of the stage: more of the quantizer's grid ends up carrying material."""
    swell = _swelling_tone()
    assert _crest_factor(compress(swell, SR, dynamics())) < _crest_factor(swell)


def test_a_threshold_over_everything_the_clip_reaches_leaves_it_alone(
    sine: Callable[..., NDArray[np.float64]], dynamics: DynamicsFactory
) -> None:
    """Levels are read against the clip's own peak, so a threshold above that peak is never crossed."""
    steady = sine(440.0)
    assert np.allclose(compress(steady, SR, dynamics(threshold_db=6.0)), steady)


def test_a_harder_ratio_holds_the_peak_further_down(dynamics: DynamicsFactory) -> None:
    swell = _swelling_tone()
    gentle = compress(swell, SR, dynamics(ratio=2.0))
    firm = compress(swell, SR, dynamics(ratio=8.0))
    assert peak_amplitude(firm) < peak_amplitude(gentle) < peak_amplitude(swell)


def test_a_ratio_of_one_passes_every_level_through(dynamics: DynamicsFactory) -> None:
    """One-to-one is the setting that keeps what it is given, whichever side of the threshold it is on."""
    swell = _swelling_tone()
    assert np.allclose(compress(swell, SR, dynamics(ratio=1.0)), swell)


def test_a_square_knee_bends_at_the_threshold_alone(dynamics: DynamicsFactory) -> None:
    """A knee of no width still divides cleanly, which is what keeps the curve defined at the edge."""
    swell = _swelling_tone()
    assert _crest_factor(compress(swell, SR, dynamics(knee_db=0.0))) < _crest_factor(swell)


# --- what it leaves as it stands --------------------------------------------------------------------


def test_a_recording_compresses_the_same_way_however_hot_it_was_captured(dynamics: DynamicsFactory) -> None:
    """The threshold reads against the clip's own peak, so the stage is settled before the level is."""
    swell = _swelling_tone()
    config = dynamics()
    assert np.allclose(compress(_QUIET * swell, SR, config), _QUIET * compress(swell, SR, config))


def test_silence_is_returned_as_it_stands(dynamics: DynamicsFactory) -> None:
    silence = np.zeros(1024, dtype=np.float64)
    assert np.array_equal(compress(silence, SR, dynamics()), silence)


def test_a_clip_holding_nothing_answers_with_nothing(dynamics: DynamicsFactory) -> None:
    assert compress(np.zeros(0, dtype=np.float64), SR, dynamics()).size == 0


def test_the_same_clip_compresses_to_the_same_samples_every_time(dynamics: DynamicsFactory) -> None:
    """Pure arithmetic end to end, which is what lets a worker process narrow a grid on its own."""
    swell = _swelling_tone()
    config = dynamics()
    assert np.array_equal(compress(swell, SR, config), compress(swell, SR, config))
