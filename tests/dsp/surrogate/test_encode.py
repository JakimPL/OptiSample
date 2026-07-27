from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dsp import EncodeConfig
from optisample.dsp.levels import peak_amplitude
from optisample.dsp.quantize import headroom_peak
from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode

SR = 44_100
_QUIET = 0.25  # the amplitude a test recording peaks at, which normalization has to lift to the headroom
_HALF_S = 0.5


def _root_mean_square(pcm: NDArray[np.float64]) -> float:
    """How much of the stored grid's range the material occupies, on average."""
    return float(np.sqrt(np.mean(pcm**2)))


def _swelling_tone(seconds: float = 1.0) -> NDArray[np.float64]:
    """A tone rising from near-silence to full scale, so compression has a crest factor to narrow."""
    times = np.arange(int(seconds * SR), dtype=np.float64) / SR
    return np.asarray(np.linspace(0.02, 1.0, times.size) * np.sin(2.0 * np.pi * 220.0 * times), dtype=np.float64)


def test_encode_sets_rate_depth_and_gain(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    encode_config: EncodeConfig,
) -> None:
    stored = encode(_QUIET * sine(440.0), SR, EncodingParams(target_rate=22_050, depth_bits=8), make_encode_ctx(57))
    assert stored.sample_rate == 22_050
    assert stored.depth_bits == 8
    assert stored.root_pitch == 57
    assert stored.frames == pytest.approx(int(round(SR * 1.0 * 22_050 / SR)), abs=2)
    assert stored.gain * _QUIET == pytest.approx(headroom_peak(encode_config.headroom_db), rel=1e-3)


def test_encode_trims_to_duration(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5)
    stored = encode(sine(440.0, dur=2.0), SR, params, make_encode_ctx(60))
    assert stored.frames == pytest.approx(int(round(0.5 * SR)), abs=2)


def test_encode_loop_stores_attack_plus_loop_and_drops_the_tail(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    params = EncodingParams(target_rate=SR, depth_bits=16, loop=True)
    stored = encode(sine(440.0, dur=2.0), SR, params, make_encode_ctx(60))
    assert stored.loop is not None
    assert stored.frames == stored.loop.end  # storage is trimmed to [0, loop.end)
    assert stored.frames < int(SR)  # ... attack + a ~0.5 s loop, well under the 2 s recording


def test_loop_falls_back_to_trim_on_non_periodic_material(make_encode_ctx: Callable[..., EncodeContext]) -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SR)
    looped = encode(
        noise, SR, EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5, loop=True), make_encode_ctx(60)
    )
    plain = encode(
        noise, SR, EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5, loop=False), make_encode_ctx(60)
    )
    assert looped.loop is None  # noise is not periodic enough to loop
    assert looped.frames == plain.frames  # ... so it is identical to the non-looped trim


# --- where the level is set ---------------------------------------------------------------------------


def test_normalization_reads_the_span_the_sample_stores(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    encode_config: EncodeConfig,
) -> None:
    """A peak in a stretch the trim discards leaves the stored sample at exactly the level it asked for."""
    signal = np.concatenate([_QUIET * sine(440.0, dur=_HALF_S), sine(440.0, dur=_HALF_S)])
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=_HALF_S)
    stored = encode(signal, SR, params, make_encode_ctx(60))
    assert peak_amplitude(stored.pcm) == pytest.approx(headroom_peak(encode_config.headroom_db), rel=1e-3)


def test_a_shared_reference_keeps_the_margin_between_two_recordings(
    sine: Callable[..., NDArray[np.float64]], encode_config: EncodeConfig
) -> None:
    """What a format keeping no per-sample gain needs: the quieter recording stays quieter in the PCM."""
    loud = sine(440.0)
    config = encode_config.model_copy(update={"peak_reference": peak_amplitude(loud)})
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=_HALF_S, dither=False)
    stored_loud = encode(loud, SR, params, EncodeContext(root_pitch=60, config=config))
    stored_quiet = encode(_QUIET * loud, SR, params, EncodeContext(root_pitch=60, config=config))
    assert stored_loud.gain == stored_quiet.gain
    assert peak_amplitude(stored_quiet.pcm) == pytest.approx(_QUIET * peak_amplitude(stored_loud.pcm), rel=1e-3)


# --- what compression buys --------------------------------------------------------------------------


def test_a_compressed_encoding_fills_more_of_the_grid_at_the_same_peak(
    make_encode_ctx: Callable[..., EncodeContext],
) -> None:
    """Both encodings normalize to the same peak, so what compression buys shows as level under it."""
    swell = _swelling_tone()
    params = EncodingParams(target_rate=SR, depth_bits=8, trim_s=1.0, dither=False)
    plain = encode(swell, SR, params, make_encode_ctx(60))
    shaped = encode(swell, SR, replace(params, compress=True), make_encode_ctx(60))
    assert peak_amplitude(shaped.pcm) == pytest.approx(peak_amplitude(plain.pcm), rel=1e-2)
    assert _root_mean_square(shaped.pcm) > _root_mean_square(plain.pcm)


def test_compression_runs_before_the_loop_seam_is_crossfaded(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """Ordering the two that way is what keeps a looped sample's seam continuous on the audio stored."""
    params = EncodingParams(target_rate=SR, depth_bits=8, loop=True, compress=True, dither=False)
    stored = encode(sine(440.0, dur=2.0), SR, params, make_encode_ctx(60))
    assert stored.loop is not None
    assert stored.frames == stored.loop.end
