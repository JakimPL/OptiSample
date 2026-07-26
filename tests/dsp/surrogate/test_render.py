from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode, render
from optisample.metrics.diagnostics import snr
from trackmod.spec.levels import MAX_VOLUME

SR = 44_100


def test_encode_render_round_trip_is_near_lossless(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    # Native rate, 16-bit, no dither, no transpose → only 16-bit quantization noise.
    source = sine(440.0)
    normalized = source / float(np.max(np.abs(source)))
    stored = encode(source, SR, EncodingParams(target_rate=SR, depth_bits=16, dither=False), make_encode_ctx(60))
    rendered = render(stored, SR, pitch=60)
    length = min(rendered.size, normalized.size)
    assert snr(normalized[:length], rendered[:length]) > 80.0


def test_render_transpose_octave_halves_length(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    stored = encode(sine(440.0), SR, EncodingParams(target_rate=SR, depth_bits=16), make_encode_ctx(60))
    base = render(stored, SR, pitch=60)
    octave_up = render(stored, SR, pitch=72)  # +12 semitones → plays twice as fast
    assert octave_up.size == pytest.approx(base.size // 2, abs=2)


def test_render_volume_scales_amplitude(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    stored = encode(sine(440.0), SR, EncodingParams(target_rate=SR, depth_bits=16, dither=False), make_encode_ctx(60))
    full = render(stored, SR, pitch=60, volume=MAX_VOLUME)
    half = render(stored, SR, pitch=60, volume=MAX_VOLUME // 2)
    length = min(full.size, half.size)
    assert np.allclose(half[:length], 0.5 * full[:length], atol=1e-9)


def test_render_zero_duration_is_empty(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    stored = encode(sine(440.0), SR, EncodingParams(target_rate=SR, depth_bits=16), make_encode_ctx(60))
    assert render(stored, SR, pitch=60, duration_s=0.0).size == 0


def test_render_duration_pads_and_truncates(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    stored = encode(sine(440.0, dur=1.0), SR, EncodingParams(target_rate=SR, depth_bits=16), make_encode_ctx(60))
    padded = render(stored, SR, pitch=60, duration_s=2.0)  # longer than the sample → zero-padded
    truncated = render(stored, SR, pitch=60, duration_s=0.25)  # shorter → cut
    assert padded.size == pytest.approx(int(round(2.0 * SR)), abs=1)
    assert truncated.size == pytest.approx(int(round(0.25 * SR)), abs=1)
    assert float(np.max(np.abs(padded[-100:]))) == 0.0  # tail is silence


def test_render_loop_sustains_a_note_held_past_the_stored_length(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    params = EncodingParams(target_rate=SR, depth_bits=16, loop=True)
    stored = encode(sine(440.0, dur=2.0), SR, params, make_encode_ctx(60))
    held = render(stored, SR, pitch=60, duration_s=3.0)  # far longer than the ~0.1 s stored
    assert held.size == pytest.approx(int(round(3.0 * SR)), abs=1)
    assert float(np.sqrt(np.mean(held[-SR:] ** 2))) > 0.1  # the last second still sounds (loop sustained it)
