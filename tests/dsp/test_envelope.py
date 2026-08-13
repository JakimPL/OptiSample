from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.loop import EnvelopeConfig
from optisample.dsp.envelope import (
    LevelReading,
    decompose,
    level_reading,
    local_level,
    local_level_over,
    power_kernel,
    reading_frequency,
)
from optisample.dsp.level import db_to_gain, gain_to_db, peak_amplitude

SR = 8_000
FREQ = 200.0
FRAMES = SR  # one second, comfortably longer than the stretch the level is read over
_AMPLITUDE = 0.8
_STEADY_LEVEL = _AMPLITUDE / np.sqrt(2.0)
_BODY = slice(SR // 4, 3 * SR // 4)  # the stretch of a recording clear of both of its ends
_QUIETER_DB = -40.0


def _sine(frames: int = FRAMES, freq: float = FREQ, amp: float = _AMPLITUDE) -> NDArray[np.float64]:
    return amp * np.sin(2.0 * np.pi * freq * np.arange(frames, dtype=np.float64) / SR)


def _ringing(fall_db: float) -> NDArray[np.float64]:
    """A struck note: one pitch under a level falling ``fall_db`` over the length of the recording."""
    seconds = np.arange(FRAMES, dtype=np.float64) / SR
    return db_to_gain(fall_db * seconds * SR / FRAMES) * _sine()


def _reading(config: EnvelopeConfig, root_hz: float = FREQ) -> LevelReading:
    """How a note played at ``root_hz`` has its level read, which is what every reading below is taken under."""
    return level_reading(SR, config, root_hz)


# --- the weighting the level is read under ------------------------------------------------------------


def test_the_weighting_spans_two_periods_of_the_frequency_it_is_formed_at() -> None:
    """Two periods is what puts that frequency on the first zero of the weighting's own response."""
    kernel = power_kernel(SR, FREQ)

    assert float(np.sum(kernel)) == pytest.approx(1.0)
    assert (kernel.size - 1) / SR == pytest.approx(2.0 / FREQ)
    assert kernel.size % 2 == 1
    assert np.allclose(kernel, kernel[::-1])  # symmetric, so each frame is read from the material centred on it


def test_a_note_is_read_over_two_periods_of_its_own_pitch(envelope_config: EnvelopeConfig) -> None:
    """Reading each note over its own period is what follows a high note as closely as the one below it."""
    within = 0.5 * (envelope_config.lowest_hz + envelope_config.highest_hz)

    assert reading_frequency(envelope_config, within) == pytest.approx(within)
    assert _reading(envelope_config, within).kernel.size > _reading(envelope_config, 2.0 * within).kernel.size


def test_a_pitch_outside_the_band_is_read_at_the_edge_of_it(envelope_config: EnvelopeConfig) -> None:
    """The band caps how long the weighting runs and floors how short it gets, whatever pitch it is handed."""
    under, over = 0.5 * envelope_config.lowest_hz, 2.0 * envelope_config.highest_hz

    assert reading_frequency(envelope_config, under) == envelope_config.lowest_hz
    assert reading_frequency(envelope_config, over) == envelope_config.highest_hz


def test_the_reading_carries_the_material_its_weighting_spans(envelope_config: EnvelopeConfig) -> None:
    """A reading states its own reach, which is the material a stretch of a recording has to be read with."""
    reading = _reading(envelope_config)

    assert reading.reach == reading.kernel.size // 2
    assert reading.floor_db == envelope_config.floor_db


# --- the level a recording holds ----------------------------------------------------------------------


def test_the_level_of_a_steady_tone_is_the_level_that_tone_holds(envelope_config: EnvelopeConfig) -> None:
    """The mean is taken of power, so a tone reads at its own root mean square wherever it sounds."""
    level = local_level(_sine(), _reading(envelope_config))

    assert np.allclose(level[_BODY], _STEADY_LEVEL, rtol=1e-3)


def test_a_tone_at_the_frequency_the_weighting_was_formed_at_leaves_a_flat_level(
    envelope_config: EnvelopeConfig,
) -> None:
    """That frequency sits on a zero of the weighting, so the power it ripples at averages away."""
    lowest = envelope_config.lowest_hz
    tone = _sine(freq=lowest)

    level = local_level(tone, _reading(envelope_config, lowest))[_BODY]

    assert gain_to_db(float(level.max())) - gain_to_db(float(level.min())) < 0.01


def test_the_curve_holds_its_level_at_both_ends_of_the_recording(envelope_config: EnvelopeConfig) -> None:
    """The weighting is shared out over the material it covers, so the first and last frames read a true level."""
    level = local_level(_sine(), _reading(envelope_config))

    assert float(level[0]) == pytest.approx(_STEADY_LEVEL, rel=0.01)
    assert float(level[-1]) == pytest.approx(_STEADY_LEVEL, rel=0.01)


def test_the_quietest_level_read_sits_the_floor_under_the_recording_own_peak(
    envelope_config: EnvelopeConfig,
) -> None:
    """The floor scales with the material, which is what leaves a silent stretch at the level it was recorded at."""
    signal = np.concatenate([_sine(SR // 2), np.zeros(SR // 2)])

    level = local_level(signal, _reading(envelope_config))

    quietest = gain_to_db(float(level.min()))
    assert quietest == pytest.approx(gain_to_db(peak_amplitude(signal)) - envelope_config.floor_db, abs=0.01)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(0, 100, id="a stretch at the very start of the recording"),
        pytest.param(SR // 2, SR // 2 + 800, id="a stretch in the body of it"),
        pytest.param(FRAMES - 500, FRAMES, id="a stretch running to the last frame"),
    ],
)
def test_reading_a_stretch_answers_what_the_whole_recording_says_of_that_stretch(
    envelope_config: EnvelopeConfig, start: int, end: int
) -> None:
    """The weighting has a finite reach, so levelling a loop region costs the region's own length."""
    signal = _ringing(_QUIETER_DB)
    reading = _reading(envelope_config)

    windowed = local_level_over(signal, reading, start=start, end=end)

    assert np.allclose(windowed, local_level(signal, reading)[start:end])


# --- the split -----------------------------------------------------------------------------------------


def test_the_split_puts_the_recording_back_together_as_it_was(envelope_config: EnvelopeConfig) -> None:
    signal = _ringing(_QUIETER_DB)

    split = decompose(signal, _reading(envelope_config))

    assert np.allclose(split.recombined(), signal, atol=1e-12)


def test_the_carrier_holds_one_level_wherever_the_note_sounds(envelope_config: EnvelopeConfig) -> None:
    """Dividing the recording by the level it holds is what leaves the carrier at one loudness to read timbre off."""
    reading = _reading(envelope_config)
    split = decompose(_ringing(_QUIETER_DB), reading)

    carried = local_level(split.carrier, reading)[_BODY]

    assert np.allclose(carried, 1.0, atol=0.01)


def test_capturing_a_sound_quieter_shifts_its_level_and_leaves_its_carrier(envelope_config: EnvelopeConfig) -> None:
    """The floor scales with the material, so what the split reads of a sound is a property of the sound."""
    signal = _ringing(_QUIETER_DB)
    reading = _reading(envelope_config)

    loud = decompose(signal, reading)
    quiet = decompose(db_to_gain(_QUIETER_DB) * signal, reading)

    assert np.allclose(quiet.carrier, loud.carrier)
    assert np.allclose(quiet.level_db - loud.level_db, _QUIETER_DB)


def test_material_carrying_no_sound_reads_at_a_level_the_split_stays_defined_at(
    envelope_config: EnvelopeConfig,
) -> None:
    split = decompose(np.zeros(FRAMES), _reading(envelope_config))

    assert np.all(split.level > 0.0)
    assert np.array_equal(split.carrier, np.zeros(FRAMES))
