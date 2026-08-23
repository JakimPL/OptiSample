from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dynamics import DynamicsConfig, LimitConfig
from optisample.config.loop import EnvelopeConfig
from optisample.dsp.dynamics import compress, held_back, reduction_db
from optisample.dsp.envelope import LevelReading, level_reading, local_level
from optisample.dsp.level import gain_to_db, peak_amplitude
from optisample.music import midi_to_freq

SR = 44_100
_QUIET = 0.1  # the amplitude a scaled-down copy of a test signal peaks at
_HELD_PITCH = 60  # the key the held-back tests read their material at, which settles the weighting
_SQUARE_KNEE_DB = 1e-9  # the width a knee asked to be square still bends over, so the quadratic divides
_NO_CEILING = 96.0  # dB over a signal's own body, past any level material reaches, so the ratio alone shapes it
_ROUNDING_DB = 1e-9  # the slack a decibel comparison leaves for the float arithmetic behind it

DynamicsFactory = Callable[..., DynamicsConfig]
HoldFactory = Callable[..., LimitConfig]


@pytest.fixture
def dynamics(dynamics_config: DynamicsConfig) -> DynamicsFactory:
    """Factory: the bundled dynamics config with the named fields overridden (re-validated)."""

    def _build(**overrides: object) -> DynamicsConfig:
        return DynamicsConfig.model_validate({**dynamics_config.model_dump(), **overrides})

    return _build


@pytest.fixture
def hold() -> HoldFactory:
    """Factory: the curve a level is held back along, stated where each test needs it."""

    def _build(
        *,
        threshold_db: float = 6.0,
        ratio: float = 8.0,
        knee_db: float = 6.0,
        attack_share: float = 0.0,
        release_share: float = 0.0,
        ceiling_db: float = _NO_CEILING,
    ) -> LimitConfig:
        return LimitConfig(
            threshold_db=threshold_db,
            ratio=ratio,
            knee_db=knee_db,
            attack_share=attack_share,
            release_share=release_share,
            ceiling_db=ceiling_db,
        )

    return _build


@pytest.fixture
def reading(envelope_config: EnvelopeConfig) -> LevelReading:
    """How the held-back tests have every level of their material read."""
    return level_reading(SR, envelope_config, midi_to_freq(_HELD_PITCH))


def _crest_factor(signal: NDArray[np.float64]) -> float:
    """How far a signal's peak stands above its own average level, which is what compression narrows."""
    return peak_amplitude(signal) / float(np.sqrt(np.mean(signal**2)))


def _level_range_db(signal: NDArray[np.float64], reading: LevelReading) -> float:
    """How far the level of ``signal`` travels between its loudest and its quietest moment."""
    level = local_level(signal, reading)
    return gain_to_db(float(np.max(level)) / float(np.min(level)))


def _spiked_tone(seconds: float = 1.0, *, spike: float = 8.0) -> NDArray[np.float64]:
    """A steady tone carrying one short burst, which is the shape a written curve has no room to state."""
    times = np.arange(int(seconds * SR), dtype=np.float64) / SR
    swell = np.ones_like(times)
    burst = slice(times.size // 2, times.size // 2 + SR // 50)
    swell[burst] = spike
    return np.asarray(swell * np.sin(2.0 * np.pi * midi_to_freq(_HELD_PITCH) * times), dtype=np.float64)


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


# --- the curve on its own ---------------------------------------------------------------------------


def test_a_level_under_the_threshold_is_left_where_it_is(hold: HoldFactory) -> None:
    """The curve opens at the threshold, so everything standing under it passes through as it is."""
    config = hold(threshold_db=6.0, knee_db=0.0)
    level = np.asarray([10.0 ** (3.0 / 20.0)], dtype=np.float64)
    assert reduction_db(level, reference=1.0, config=config) == pytest.approx(0.0, abs=_SQUARE_KNEE_DB)


def test_a_level_past_the_threshold_keeps_the_share_the_ratio_names(hold: HoldFactory) -> None:
    """Six decibels over a square knee at four to one arrive as one and a half, so four and a half go."""
    config = hold(threshold_db=6.0, ratio=4.0, knee_db=0.0)
    level = np.asarray([10.0 ** (12.0 / 20.0)], dtype=np.float64)
    assert reduction_db(level, reference=1.0, config=config) == pytest.approx(-4.5)


def test_the_curve_reads_the_same_however_hot_the_level_stands(hold: HoldFactory) -> None:
    """A reference the level carries with it is what makes the reading a property of the material."""
    config = hold()
    level = np.linspace(0.01, 1.0, 64, dtype=np.float64)
    assert np.allclose(
        reduction_db(level, reference=1.0, config=config),
        reduction_db(_QUIET * level, reference=_QUIET, config=config),
    )


# --- what a level-detecting hold does to a waveform --------------------------------------------------


def test_a_level_the_curve_already_states_comes_back_as_it_stands(
    sine: Callable[..., NDArray[np.float64]], reading: LevelReading, hold: HoldFactory
) -> None:
    """The threshold reads over the body, so material sitting where it sits throughout is held back by nothing."""
    steady = sine(midi_to_freq(_HELD_PITCH))
    assert np.allclose(held_back(steady, reading, hold()), 1.0)


def test_a_spike_the_curve_missed_is_pulled_toward_the_body(reading: LevelReading, hold: HoldFactory) -> None:
    """The whole point of the pass: what a written curve had no room for stops setting the peak."""
    spiked = _spiked_tone()
    held = spiked * held_back(spiked, reading, hold())
    assert _level_range_db(held, reading) < _level_range_db(spiked, reading)
    assert _crest_factor(held) < _crest_factor(spiked)


def test_a_harder_ratio_pulls_a_spike_further_down(reading: LevelReading, hold: HoldFactory) -> None:
    spiked = _spiked_tone()
    gentle = spiked * held_back(spiked, reading, hold(ratio=2.0))
    firm = spiked * held_back(spiked, reading, hold(ratio=16.0))
    assert _level_range_db(firm, reading) < _level_range_db(gentle, reading) < _level_range_db(spiked, reading)


def test_a_ratio_of_one_holds_nothing_back(reading: LevelReading, hold: HoldFactory) -> None:
    """One-to-one is the setting that stores the waveform exactly as the curve before it left it."""
    spiked = _spiked_tone()
    assert np.allclose(held_back(spiked, reading, hold(ratio=1.0)), 1.0)


def test_material_carrying_nothing_is_held_back_by_nothing(reading: LevelReading, hold: HoldFactory) -> None:
    """A silent stretch carries no body to read a threshold against, so it comes back as it stands."""
    silence = np.zeros(SR, dtype=np.float64)
    assert np.array_equal(held_back(silence, reading, hold()), np.ones(SR, dtype=np.float64))


def test_the_hold_reads_the_same_however_hot_the_waveform_stands(reading: LevelReading, hold: HoldFactory) -> None:
    """The reference comes off the level itself, so a set is held back before its level is settled."""
    spiked = _spiked_tone()
    config = hold()
    assert np.allclose(held_back(_QUIET * spiked, reading, config), held_back(spiked, reading, config))


def test_a_transient_is_taken_down_before_it_arrives(reading: LevelReading, hold: HoldFactory) -> None:
    """The weighting is symmetric, so the hold reaches back as far ahead of a burst as behind it."""
    spiked = _spiked_tone()
    gain = held_back(spiked, reading, hold())
    onset = int(np.argmax(np.abs(spiked) > 1.5))
    assert gain[onset - reading.reach // 2] < gain[0]


# --- the attack, the release and the ceiling ---------------------------------------------------------


def test_an_attack_opens_the_hold_further_ahead_of_the_peak(reading: LevelReading, hold: HoldFactory) -> None:
    """A sub-unit attack is a share of the reach the detector already spans, spent as lookahead."""
    spiked = _spiked_tone()
    onset = int(np.argmax(np.abs(spiked) > 1.5))
    ahead = onset - reading.reach

    assert held_back(spiked, reading, hold(attack_share=0.5))[ahead] < held_back(spiked, reading, hold())[ahead]


def test_a_release_keeps_the_hold_open_behind_the_peak(reading: LevelReading, hold: HoldFactory) -> None:
    """The level settles once behind a transient rather than following the decay back up."""
    spiked = _spiked_tone()
    onset = int(np.argmax(np.abs(spiked) > 1.5))
    behind = onset + 2 * reading.reach

    assert held_back(spiked, reading, hold(release_share=4.0))[behind] < held_back(spiked, reading, hold())[behind]


def test_shares_of_zero_hold_the_reduction_to_the_moment_the_curve_asks_for_it(
    reading: LevelReading, hold: HoldFactory
) -> None:
    """The curve on its own is what a run states by asking for neither an attack nor a release."""
    spiked = _spiked_tone()
    assert np.array_equal(
        held_back(spiked, reading, hold(attack_share=0.0, release_share=0.0)),
        held_back(spiked, reading, hold()),
    )


def test_a_ceiling_holds_a_peak_the_ratio_alone_would_pass(reading: LevelReading, hold: HoldFactory) -> None:
    """The absolute limit behind the curve: no frame is left standing further over the body than it names."""
    spiked = _spiked_tone()
    ceiling_db = 3.0
    body = float(np.sqrt(np.mean(spiked**2)))
    held = local_level(spiked, reading) * held_back(spiked, reading, hold(ratio=1.0, ceiling_db=ceiling_db))

    assert float(np.max(gain_to_db(held / body))) <= ceiling_db + _ROUNDING_DB


def test_a_ceiling_past_every_level_leaves_the_ratio_to_shape_the_material(
    reading: LevelReading, hold: HoldFactory
) -> None:
    """A ceiling no material reaches is inert, so the curve alone answers."""
    spiked = _spiked_tone()
    assert np.array_equal(
        held_back(spiked, reading, hold(ceiling_db=_NO_CEILING)),
        held_back(spiked, reading, hold(ceiling_db=2.0 * _NO_CEILING)),
    )
