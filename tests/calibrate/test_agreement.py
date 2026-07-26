from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.calibrate import (
    CalibrationContext,
    NoteProbe,
    RendererAgreement,
    distortion_vs_source,
    rank_correlation,
    render_note_openmpt,
    render_note_surrogate,
    renderer_agreement,
)
from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.io.render import openmpt123_available

SAMPLE_RATE = 44_100

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")


# --- unit tests (no openmpt123 needed) ------------------------------------------------------------


def test_render_note_surrogate_matches_requested_duration(stored: Callable[..., StoredSample]) -> None:
    sample = stored(root_pitch=60)
    out = render_note_surrogate(sample, NoteProbe(pitch=60, duration_s=0.5), out_rate=8_000)
    assert out.size == 4_000  # 0.5 s at 8 kHz


def test_rank_correlation_is_one_for_same_order_and_minus_one_for_reverse() -> None:
    surrogate = np.array([0.1, 0.2, 0.3, 0.9])
    assert rank_correlation(surrogate, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)
    assert rank_correlation(surrogate, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_rank_correlation_is_nan_with_fewer_than_two_points() -> None:
    assert np.isnan(rank_correlation(np.array([0.5]), np.array([0.5])))


# --- ground-truth calibration (openmpt123 required) ----------------------------------------------


@requires_openmpt
def test_render_note_openmpt_matches_requested_duration(
    make_encode_ctx: Callable[..., EncodeContext],
    recording: Callable[..., NDArray[np.float64]],
    calibration_context: CalibrationContext,
) -> None:
    stored = encode(recording("piano", 60, 1.5), SAMPLE_RATE, EncodingParams(44_100, 16), make_encode_ctx(60))
    out = render_note_openmpt(stored, NoteProbe(pitch=60, duration_s=1.0), calibration_context)
    assert out.size == int(round(1.0 * calibration_context.render.sample_rate))


@requires_openmpt
@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_surrogate_agrees_with_openmpt_at_root_pitch(
    archetype: str,
    make_encode_ctx: Callable[..., EncodeContext],
    recording: Callable[..., NDArray[np.float64]],
    calibration_context: CalibrationContext,
) -> None:
    stored = encode(recording(archetype, 60, 2.0), SAMPLE_RATE, EncodingParams(44_100, 16), make_encode_ctx(60))
    agree = renderer_agreement(stored, NoteProbe(pitch=60, duration_s=1.5), calibration_context)
    assert isinstance(agree, RendererAgreement)
    assert agree.distance < 0.05  # observed ~0.001-0.002; the two engines are near-identical at root
    assert abs(agree.loudness_delta_lu) < 25.0  # a real (constant) level gap from IT gain staging exists
    assert {"mrstft", "logmel_l1", "spectral_shape", "mcd"} <= set(agree.breakdown)  # diagnostic per-metric view


@requires_openmpt
def test_surrogate_agrees_with_openmpt_when_transposed(
    make_encode_ctx: Callable[..., EncodeContext],
    recording: Callable[..., NDArray[np.float64]],
    calibration_context: CalibrationContext,
) -> None:
    stored = encode(recording("piano", 60, 2.0), SAMPLE_RATE, EncodingParams(44_100, 16), make_encode_ctx(60))
    agree = renderer_agreement(stored, NoteProbe(pitch=67, duration_s=1.0), calibration_context)  # +7 semitones
    assert agree.distance < 0.1  # observed ~0.013; larger than root (resampler differences) but small


@requires_openmpt
@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_surrogate_ranks_operating_points_like_openmpt(
    archetype: str,
    make_encode_ctx: Callable[..., EncodeContext],
    recording: Callable[..., NDArray[np.float64]],
    calibration_context: CalibrationContext,
) -> None:
    signal = recording(archetype, 60, 1.5)
    reference = resample_to(signal, SAMPLE_RATE, calibration_context.render.sample_rate)
    probe = NoteProbe(pitch=60, duration_s=1.0)
    # Operating points that are each meaningfully separated (distortions span ~0.002 -> ~3.0), so a
    # correct ranking is well-defined. We deliberately do not pit 44.1/16 against 22/16: both are
    # effectively lossless (~0.002, inaudible), so their relative order is numerical noise, not a claim
    # ground truth can adjudicate.
    grid = [
        EncodingParams(target_rate=44_100, depth_bits=16),
        EncodingParams(target_rate=11_025, depth_bits=16),
        EncodingParams(target_rate=11_025, depth_bits=8),
        EncodingParams(target_rate=22_050, depth_bits=8),
    ]
    surrogate, openmpt = [], []
    for params in grid:
        stored = encode(signal, SAMPLE_RATE, params, make_encode_ctx(60, seed=0))
        d_surrogate, d_openmpt = distortion_vs_source(reference, stored, probe, calibration_context)
        surrogate.append(d_surrogate)
        openmpt.append(d_openmpt)
    # The headline calibration claim: the objective ranks encodings the way ground truth does.
    assert rank_correlation(np.array(surrogate), np.array(openmpt)) >= 0.9
