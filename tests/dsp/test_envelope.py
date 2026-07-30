from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.loop import EnvelopeConfig
from optisample.dsp.envelope import decompose, local_level, local_level_over, power_kernel
from optisample.dsp.levels import db_to_gain, gain_to_db, peak_amplitude

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


# --- the weighting the level is read under ------------------------------------------------------------


def test_the_weighting_spans_two_periods_of_the_lowest_frequency_it_reads(envelope_config: EnvelopeConfig) -> None:
    """Two periods is what puts that frequency on the first zero of the weighting's own response."""
    kernel = power_kernel(SR, envelope_config.lowest_hz)

    assert float(np.sum(kernel)) == pytest.approx(1.0)
    assert (kernel.size - 1) / SR == pytest.approx(2.0 / envelope_config.lowest_hz)
    assert kernel.size % 2 == 1
    assert np.allclose(kernel, kernel[::-1])  # symmetric, so each frame is read from the material centred on it


# --- the level a recording holds ----------------------------------------------------------------------


def test_the_level_of_a_steady_tone_is_the_level_that_tone_holds(envelope_config: EnvelopeConfig) -> None:
    """The mean is taken of power, so a tone reads at its own root mean square wherever it sounds."""
    level = local_level(_sine(), SR, envelope_config)

    assert np.allclose(level[_BODY], _STEADY_LEVEL, rtol=1e-3)


def test_a_tone_at_the_lowest_frequency_read_leaves_a_flat_level(envelope_config: EnvelopeConfig) -> None:
    """The lowest frequency sits on a zero of the weighting, so the power it ripples at averages away."""
    tone = _sine(freq=envelope_config.lowest_hz)

    level = local_level(tone, SR, envelope_config)[_BODY]

    assert gain_to_db(float(level.max())) - gain_to_db(float(level.min())) < 0.01


def test_the_curve_holds_its_level_at_both_ends_of_the_recording(envelope_config: EnvelopeConfig) -> None:
    """The weighting is shared out over the material it covers, so the first and last frames read a true level."""
    level = local_level(_sine(), SR, envelope_config)

    assert float(level[0]) == pytest.approx(_STEADY_LEVEL, rel=0.01)
    assert float(level[-1]) == pytest.approx(_STEADY_LEVEL, rel=0.01)


def test_the_quietest_level_read_sits_the_floor_under_the_recording_own_peak(
    envelope_config: EnvelopeConfig,
) -> None:
    """The floor scales with the material, which is what leaves a silent stretch at the level it was recorded at."""
    signal = np.concatenate([_sine(SR // 2), np.zeros(SR // 2)])

    level = local_level(signal, SR, envelope_config)

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

    windowed = local_level_over(signal, SR, envelope_config, start=start, end=end)

    assert np.allclose(windowed, local_level(signal, SR, envelope_config)[start:end])


# --- the split -----------------------------------------------------------------------------------------


def test_the_split_puts_the_recording_back_together_as_it_was(envelope_config: EnvelopeConfig) -> None:
    signal = _ringing(_QUIETER_DB)

    split = decompose(signal, SR, envelope_config)

    assert np.allclose(split.recombined(), signal, atol=1e-12)


def test_the_carrier_holds_one_level_wherever_the_note_sounds(envelope_config: EnvelopeConfig) -> None:
    """Dividing the recording by the level it holds is what leaves the carrier at one loudness to read timbre off."""
    split = decompose(_ringing(_QUIETER_DB), SR, envelope_config)

    carried = local_level(split.carrier, SR, envelope_config)[_BODY]

    assert np.allclose(carried, 1.0, atol=0.01)


def test_capturing_a_sound_quieter_shifts_its_level_and_leaves_its_carrier(envelope_config: EnvelopeConfig) -> None:
    """The floor scales with the material, so what the split reads of a sound is a property of the sound."""
    signal = _ringing(_QUIETER_DB)

    loud = decompose(signal, SR, envelope_config)
    quiet = decompose(db_to_gain(_QUIETER_DB) * signal, SR, envelope_config)

    assert np.allclose(quiet.carrier, loud.carrier)
    assert np.allclose(quiet.level_db - loud.level_db, _QUIETER_DB)


def test_material_carrying_no_sound_reads_at_a_level_the_split_stays_defined_at(
    envelope_config: EnvelopeConfig,
) -> None:
    split = decompose(np.zeros(FRAMES), SR, envelope_config)

    assert np.all(split.level > 0.0)
    assert np.array_equal(split.carrier, np.zeros(FRAMES))
