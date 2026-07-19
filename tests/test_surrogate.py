from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.surrogate import MAX_VOLUME, EncodingParams, StoredSample, encode, render, semitone_ratio
from optisample.metrics.diagnostics import snr
from optisample.metrics.size import SampleSize

SR = 44_100


def sine(freq: float, dur: float = 1.0, amp: float = 1.0) -> NDArray[np.float64]:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def test_semitone_ratio_octaves() -> None:
    assert semitone_ratio(0.0) == pytest.approx(1.0)
    assert semitone_ratio(12.0) == pytest.approx(2.0)
    assert semitone_ratio(-12.0) == pytest.approx(0.5)


def test_stored_sample_properties() -> None:
    stored = StoredSample(pcm=np.zeros(1000, dtype=np.float64), sample_rate=22_050, depth_bits=8, root_pitch=60)
    assert stored.frames == 1000
    assert stored.size == SampleSize(frames=1000, depth_bits=8)
    assert stored.stored_bytes == SampleSize(frames=1000, depth_bits=8).total_bytes
    assert stored.duration_s == pytest.approx(1000 / 22_050)


def test_encode_sets_rate_depth_and_gain() -> None:
    stored = encode(0.25 * sine(440.0), SR, EncodingParams(target_rate=22_050, depth_bits=8), root_pitch=57)
    assert stored.sample_rate == 22_050
    assert stored.depth_bits == 8
    assert stored.root_pitch == 57
    assert stored.frames == pytest.approx(int(round(SR * 1.0 * 22_050 / SR)), abs=2)
    assert stored.gain == pytest.approx(1.0 / 0.25, rel=1e-6)  # normalized to full scale


def test_encode_trims_to_duration() -> None:
    stored = encode(sine(440.0, dur=2.0), SR, EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5), root_pitch=60)
    assert stored.frames == pytest.approx(int(round(0.5 * SR)), abs=2)


def test_encode_render_round_trip_is_near_lossless() -> None:
    # Native rate, 16-bit, no dither, no transpose → only 16-bit quantization noise.
    source = sine(440.0)
    normalized = source / float(np.max(np.abs(source)))
    stored = encode(source, SR, EncodingParams(target_rate=SR, depth_bits=16, dither=False), root_pitch=60)
    rendered = render(stored, SR, pitch=60)
    length = min(rendered.size, normalized.size)
    assert snr(normalized[:length], rendered[:length]) > 80.0


def test_render_transpose_octave_halves_length() -> None:
    stored = encode(sine(440.0), SR, EncodingParams(target_rate=SR, depth_bits=16), root_pitch=60)
    base = render(stored, SR, pitch=60)
    octave_up = render(stored, SR, pitch=72)  # +12 semitones → plays twice as fast
    assert octave_up.size == pytest.approx(base.size // 2, abs=2)


def test_render_volume_scales_amplitude() -> None:
    stored = encode(sine(440.0), SR, EncodingParams(target_rate=SR, depth_bits=16, dither=False), root_pitch=60)
    full = render(stored, SR, pitch=60, volume=MAX_VOLUME)
    half = render(stored, SR, pitch=60, volume=MAX_VOLUME // 2)
    length = min(full.size, half.size)
    assert np.allclose(half[:length], 0.5 * full[:length], atol=1e-9)


def test_render_zero_duration_is_empty() -> None:
    stored = encode(sine(440.0), SR, EncodingParams(target_rate=SR, depth_bits=16), root_pitch=60)
    assert render(stored, SR, pitch=60, duration_s=0.0).size == 0


def test_render_duration_pads_and_truncates() -> None:
    stored = encode(sine(440.0, dur=1.0), SR, EncodingParams(target_rate=SR, depth_bits=16), root_pitch=60)
    padded = render(stored, SR, pitch=60, duration_s=2.0)  # longer than the sample → zero-padded
    truncated = render(stored, SR, pitch=60, duration_s=0.25)  # shorter → cut
    assert padded.size == pytest.approx(int(round(2.0 * SR)), abs=1)
    assert truncated.size == pytest.approx(int(round(0.25 * SR)), abs=1)
    assert float(np.max(np.abs(padded[-100:]))) == 0.0  # tail is silence


# --- looping -------------------------------------------------------------------------------------


def test_encode_loop_stores_attack_plus_loop_and_drops_the_tail() -> None:
    stored = encode(sine(440.0, dur=2.0), SR, EncodingParams(target_rate=SR, depth_bits=16, loop=True), root_pitch=60)
    assert stored.loop is not None
    assert stored.frames == stored.loop.end  # storage is trimmed to [0, loop.end)
    assert stored.frames < int(0.5 * SR)  # ... a small fraction of the 2 s recording


def test_render_loop_sustains_a_note_held_past_the_stored_length() -> None:
    stored = encode(sine(440.0, dur=2.0), SR, EncodingParams(target_rate=SR, depth_bits=16, loop=True), root_pitch=60)
    held = render(stored, SR, pitch=60, duration_s=3.0)  # far longer than the ~0.1 s stored
    assert held.size == pytest.approx(int(round(3.0 * SR)), abs=1)
    assert float(np.sqrt(np.mean(held[-SR:] ** 2))) > 0.1  # the last second still sounds (loop sustained it)


def test_loop_falls_back_to_trim_on_non_periodic_material() -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SR)
    looped = encode(noise, SR, EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5, loop=True), root_pitch=60)
    plain = encode(noise, SR, EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5, loop=False), root_pitch=60)
    assert looped.loop is None  # noise is not periodic enough to loop
    assert looped.frames == plain.frames  # ... so it is identical to the non-looped trim
