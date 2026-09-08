from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import load_config
from optisample.config.subsonic import SubsonicConfig
from optisample.dsp.subsonic import remove_subsonic, subsonic_order, subsonic_sections

SR: Final = 44_100
_CONFIG: Final = load_config().subsonic
_PASSBAND_HZ: Final = 440.0  # a tone standing well over any cutoff a subsonic curve is set at
_FLAT_DB: Final = 0.05  # decibels a passed tone is held within, which is what "at the level captured" means
_SETTLED: Final = slice(SR // 4, -SR // 4)  # the middle of a one-second reading, where both ends have come to rest


def tone(freq: float, dur: float = 1.0, sample_rate: int = SR) -> NDArray[np.float64]:
    times = np.arange(int(dur * sample_rate), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq * times)


def level_db(signal: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """How far ``signal`` stands under ``reference`` in decibels, read over the settled middle of both."""
    return float(20.0 * np.log10(np.sqrt(np.mean(signal[_SETTLED] ** 2) / np.mean(reference[_SETTLED] ** 2))))


@dataclass(frozen=True)
class ShapeCase:
    """One point of the curve: a tone, and the band of decibels it is expected to arrive within."""

    freq: float
    lowest_db: float
    highest_db: float


def test_the_order_reaches_the_depth_the_two_stated_points_ask_for() -> None:
    """Both passes together stand at least the stated decibels down where the config measures them."""
    reached = 20.0 * np.log10(1.0 + (_CONFIG.cutoff_hz / _CONFIG.rejection_hz) ** (2 * subsonic_order(_CONFIG)))
    assert reached >= _CONFIG.rejection_db


def test_the_curve_is_the_shallowest_one_arriving_at_that_depth() -> None:
    """One order less falls short, so the response stays as gentle as reaching the depth allows."""
    order = subsonic_order(_CONFIG)
    shallower = 20.0 * np.log10(1.0 + (_CONFIG.cutoff_hz / _CONFIG.rejection_hz) ** (2 * (order - 1)))
    assert shallower < _CONFIG.rejection_db


def test_a_gentle_rejection_still_leaves_the_curve_a_pole() -> None:
    """A depth one pole already passes settles on the shallowest curve there is."""
    gentle = SubsonicConfig(cutoff_hz=30.0, rejection_hz=10.0, rejection_db=0.5)
    assert subsonic_order(gentle) == 1


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(ShapeCase(freq=440.0, lowest_db=-_FLAT_DB, highest_db=_FLAT_DB), id="a passed tone"),
        pytest.param(ShapeCase(freq=110.0, lowest_db=-_FLAT_DB, highest_db=_FLAT_DB), id="a low note"),
        pytest.param(ShapeCase(freq=55.0, lowest_db=-0.5, highest_db=0.0), id="the lowest keys"),
        pytest.param(ShapeCase(freq=30.0, lowest_db=-8.0, highest_db=-4.0), id="the cutoff itself"),
        pytest.param(ShapeCase(freq=10.0, lowest_db=-200.0, highest_db=-60.0), id="the rejected depth"),
    ],
)
def test_the_curve_passes_the_material_and_falls_away_under_it(case: ShapeCase) -> None:
    """The shipped shape holds the notes a keyboard sounds and arrives at its stated depth by 10 Hz."""
    signal = tone(case.freq)
    assert case.lowest_db <= level_db(remove_subsonic(signal, SR, _CONFIG), signal) <= case.highest_db


def test_a_steady_offset_leaves_the_recording() -> None:
    """A recording sitting off zero comes back centered, an offset being the deepest content there is."""
    signal = tone(_PASSBAND_HZ) + 0.5
    cleaned = remove_subsonic(signal, SR, _CONFIG)
    assert abs(float(np.mean(cleaned[_SETTLED]))) < 1.0e-6


def test_a_passed_tone_keeps_the_phase_it_was_captured_on() -> None:
    """Reading forward and back leaves the waveform where it stood, so an onset stays where it was."""
    signal = tone(_PASSBAND_HZ)
    cleaned = remove_subsonic(signal, SR, _CONFIG)
    assert np.max(np.abs((cleaned - signal)[_SETTLED])) < 1.0e-3


def test_reading_a_cleaned_recording_again_leaves_it_as_it_stands() -> None:
    """The band is gone once, so a recording carried through a second stage arrives unchanged."""
    once = remove_subsonic(tone(_PASSBAND_HZ) + 0.5, SR, _CONFIG)
    twice = remove_subsonic(once, SR, _CONFIG)
    assert np.max(np.abs((twice - once)[_SETTLED])) < 1.0e-6


def test_a_recording_of_one_frame_stands_as_it_is() -> None:
    """A depth takes frames to measure, so the shortest recordings come back the way they arrived."""
    single = np.array([0.5], dtype=np.float64)
    np.testing.assert_array_equal(remove_subsonic(single, SR, _CONFIG), single)
    assert remove_subsonic(np.zeros(0, dtype=np.float64), SR, _CONFIG).size == 0


def test_the_sections_are_laid_out_two_poles_at_a_time() -> None:
    """Second-order sections are what the curve stays numerically stable in however steep it stands."""
    sections = subsonic_sections(SR, _CONFIG)
    assert sections.shape == ((subsonic_order(_CONFIG) + 1) // 2, 6)


def test_a_cutoff_over_the_rate_leaves_no_band_to_pass() -> None:
    steep = SubsonicConfig(cutoff_hz=30.0, rejection_hz=10.0, rejection_db=60.0)
    with pytest.raises(ValueError):
        subsonic_sections(40, steep)


def test_the_rejection_point_stands_under_the_cutoff() -> None:
    with pytest.raises(ValueError, match="must stand under cutoff_hz"):
        SubsonicConfig(cutoff_hz=30.0, rejection_hz=30.0, rejection_db=60.0)
