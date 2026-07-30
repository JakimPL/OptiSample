from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.codec import EncodeConfig
from optisample.config.tracker import TrackerFormat
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.optimize.orchestrate.staging import instrument_peak, staged_encode
from optisample.optimize.tasks import AudioMap

_LOUD = 0.8
_QUIET = 0.2

Retarget = Callable[[TrackerFormat], ExportTarget]


def _tone(amplitude: float) -> NDArray[np.float64]:
    times = np.arange(256, dtype=np.float64) / 256.0
    return np.asarray(amplitude * np.sin(2.0 * np.pi * times), dtype=np.float64)


@pytest.fixture
def audio() -> AudioMap:
    """Two survivors an octave apart, recorded at levels far enough apart to tell one reference from two."""
    return {SampleKey(60, 100): _tone(_LOUD), SampleKey(72, 100): _tone(_QUIET)}


# --- what an instrument peaks at --------------------------------------------------------------------


def test_an_instrument_peaks_at_its_loudest_recording(audio: AudioMap) -> None:
    assert instrument_peak(audio) == pytest.approx(_LOUD)


def test_an_instrument_holding_no_recording_peaks_at_nothing() -> None:
    assert instrument_peak({}) == 0.0


# --- what each format is staged for -----------------------------------------------------------------


def test_a_format_restoring_level_on_playback_leaves_each_clip_on_its_own_peak(
    audio: AudioMap, encode_config: EncodeConfig, retarget: Retarget
) -> None:
    """Impulse Tracker writes the balance beside the samples, so each one is stored as hot as it can be."""
    assert staged_encode(encode_config, retarget(TrackerFormat.IT), audio) == encode_config


def test_a_format_without_one_normalizes_every_clip_against_the_loudest(
    audio: AudioMap, encode_config: EncodeConfig, retarget: Retarget
) -> None:
    """FastTracker 2 pins its per-sample gain to full, so the margins have to survive into the PCM."""
    staged = staged_encode(encode_config, retarget(TrackerFormat.XM), audio)
    assert staged.peak_reference == pytest.approx(_LOUD)


def test_staging_moves_the_reference_and_leaves_every_other_setting_alone(
    audio: AudioMap, encode_config: EncodeConfig, retarget: Retarget
) -> None:
    staged = staged_encode(encode_config, retarget(TrackerFormat.XM), audio)
    assert (staged.seam, staged.dynamics, staged.headroom_db) == (
        encode_config.seam,
        encode_config.dynamics,
        encode_config.headroom_db,
    )
