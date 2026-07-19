from __future__ import annotations

import numpy as np
import pytest

from optisample.calibrate import (
    NoteProbe,
    RendererAgreement,
    distortion_vs_source,
    rank_correlation,
    render_note_openmpt,
    render_note_surrogate,
    renderer_agreement,
    single_note_module,
)
from optisample.config.render import RenderConfig
from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate import MAX_VOLUME, EncodingParams, StoredSample, encode
from optisample.io.render import openmpt123_available
from optisample.optimize.export import c5speed_for_pitch
from optisample.synth import SAMPLE_RATE, NoteSpec, render_sample

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")
SETTINGS = RenderConfig(sample_rate=48_000, interpolation="sinc", gain_db=0.0)


def _stored(root_pitch: int = 60, rate: int = 22_050, frames: int = 8_000) -> StoredSample:
    time = np.arange(frames) / rate
    pcm = 0.6 * np.sin(2 * np.pi * 220 * time)
    return StoredSample(pcm=pcm, sample_rate=rate, depth_bits=16, root_pitch=root_pitch)


def _recording(archetype: str, pitch: int, dur: float) -> np.ndarray:
    return render_sample(archetype, NoteSpec(pitch, 100, 0.0, dur, SAMPLE_RATE), np.random.default_rng(pitch))


# --- unit tests (no openmpt123 needed) ------------------------------------------------------------


def test_note_probe_defaults_to_full_volume() -> None:
    assert NoteProbe(pitch=60).volume == MAX_VOLUME


def test_single_note_module_wires_one_sample_to_the_probed_key() -> None:
    stored = _stored(root_pitch=48, rate=22_050)
    module = single_note_module(stored, NoteProbe(pitch=60, volume=50, duration_s=0.5))
    assert len(module.samples) == 1
    assert module.samples[0].c5speed == c5speed_for_pitch(22_050, 48)  # root key plays natural
    assert module.instruments[0].note_map[60] == (60, 1)  # probed key -> (identity note, sample 1)
    row, channel, cell = module.patterns[0].cells[0]
    assert (row, channel) == (0, 0)
    assert (cell.note, cell.instrument, cell.volume) == (60, 1, 50)


def test_single_note_module_rows_cover_the_duration() -> None:
    stored = _stored()
    short = single_note_module(stored, NoteProbe(pitch=60, duration_s=0.5))
    longer = single_note_module(stored, NoteProbe(pitch=60, duration_s=2.0))
    assert longer.patterns[0].rows > short.patterns[0].rows
    assert short.patterns[0].rows == int(0.5 / (6 * 2.5 / 125)) + 2  # speed 6, tempo 125 -> 0.12 s/row


def test_render_note_surrogate_matches_requested_duration() -> None:
    stored = _stored(root_pitch=60)
    out = render_note_surrogate(stored, NoteProbe(pitch=60, duration_s=0.5), out_rate=8_000)
    assert out.size == 4_000  # 0.5 s at 8 kHz


def test_rank_correlation_is_one_for_same_order_and_minus_one_for_reverse() -> None:
    surrogate = np.array([0.1, 0.2, 0.3, 0.9])
    assert rank_correlation(surrogate, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)
    assert rank_correlation(surrogate, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_rank_correlation_is_nan_with_fewer_than_two_points() -> None:
    assert np.isnan(rank_correlation(np.array([0.5]), np.array([0.5])))


# --- ground-truth calibration (openmpt123 required) ----------------------------------------------


@requires_openmpt
def test_render_note_openmpt_matches_requested_duration(make_encode_ctx) -> None:
    stored = encode(_recording("piano", 60, 1.5), SAMPLE_RATE, EncodingParams(44_100, 16), make_encode_ctx(60))
    out = render_note_openmpt(stored, NoteProbe(pitch=60, duration_s=1.0), SETTINGS)
    assert out.size == int(round(1.0 * SETTINGS.sample_rate))


@requires_openmpt
@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_surrogate_agrees_with_openmpt_at_root_pitch(archetype: str, make_encode_ctx) -> None:
    stored = encode(_recording(archetype, 60, 2.0), SAMPLE_RATE, EncodingParams(44_100, 16), make_encode_ctx(60))
    agree = renderer_agreement(stored, NoteProbe(pitch=60, duration_s=1.5), SETTINGS)
    assert isinstance(agree, RendererAgreement)
    assert agree.distance < 0.05  # observed ~0.001-0.002; the two engines are near-identical at root
    assert abs(agree.loudness_delta_lu) < 25.0  # a real (constant) level gap from IT gain staging exists
    assert {"mrstft", "logmel_l1", "spectral_shape", "mcd"} <= set(agree.breakdown)  # diagnostic per-metric view


@requires_openmpt
def test_surrogate_agrees_with_openmpt_when_transposed(make_encode_ctx) -> None:
    stored = encode(_recording("piano", 60, 2.0), SAMPLE_RATE, EncodingParams(44_100, 16), make_encode_ctx(60))
    agree = renderer_agreement(stored, NoteProbe(pitch=67, duration_s=1.0), SETTINGS)  # +7 semitones
    assert agree.distance < 0.1  # observed ~0.013; larger than root (resampler differences) but small


@requires_openmpt
@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_surrogate_ranks_operating_points_like_openmpt(archetype: str, make_encode_ctx) -> None:
    recording = _recording(archetype, 60, 1.5)
    reference = resample_to(recording, SAMPLE_RATE, SETTINGS.sample_rate)
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
        stored = encode(recording, SAMPLE_RATE, params, make_encode_ctx(60, seed=0))
        d_surrogate, d_openmpt = distortion_vs_source(reference, stored, probe, SETTINGS)
        surrogate.append(d_surrogate)
        openmpt.append(d_openmpt)
    # The headline calibration claim: the objective ranks encodings the way ground truth does.
    assert rank_correlation(np.array(surrogate), np.array(openmpt)) >= 0.9
