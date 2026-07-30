from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.codec import EncodeConfig
from optisample.config.loop import GeometryConfig
from optisample.dsp.levels import peak_amplitude
from optisample.dsp.quantize import headroom_peak
from optisample.dsp.surrogate import (
    NO_LOOP,
    NO_RELEASE_RAMP,
    EncodeContext,
    EncodingParams,
    SettledLoop,
    encode,
)

SR = 44_100
_QUIET = 0.25  # the amplitude a test recording peaks at, which normalization has to lift to the headroom
_HALF_S = 0.5
_TONE_S = 2.0  # a recording long enough for the stage to place a loop inside and still leave a tail
_TONE_HZ = 440.0  # the pitch every recording below is played at, which its loop is settled around

SettleLoop = Callable[..., SettledLoop | None]


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
    stored = encode(_QUIET * sine(_TONE_HZ), SR, EncodingParams(target_rate=22_050, depth_bits=8), make_encode_ctx(57))
    assert stored.sample_rate == 22_050
    assert stored.depth_bits == 8
    assert stored.root_pitch == 57
    assert stored.frames == pytest.approx(int(round(SR * 1.0 * 22_050 / SR)), abs=2)
    assert stored.gain * _QUIET == pytest.approx(headroom_peak(encode_config.headroom_db), rel=1e-3)


def test_encode_trims_to_duration(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5)
    stored = encode(sine(_TONE_HZ, dur=2.0), SR, params, make_encode_ctx(60))
    assert stored.frames == pytest.approx(int(round(0.5 * SR)), abs=2)


def test_encode_loop_stores_attack_plus_loop_and_drops_the_tail(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    settle: SettleLoop,
    geometry_config: GeometryConfig,
) -> None:
    recording = sine(_TONE_HZ, dur=_TONE_S)
    settled = settle(recording, SR, root_hz=_TONE_HZ, search_s=_TONE_S)
    params = EncodingParams(target_rate=SR, depth_bits=16, looped=True)
    stored = encode(recording, SR, params, make_encode_ctx(60, settled=settled))
    assert stored.loop is not None
    assert stored.frames == stored.loop.end  # storage is trimmed to [0, loop.end)
    assert stored.frames < recording.size  # ... the attack plus one loop, so the sustain tail is dropped
    assert stored.loop.length >= round(geometry_config.min_loop_s * SR)


def test_a_trimmed_sample_closes_on_the_release_ramp(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    encode_config: EncodeConfig,
) -> None:
    """A cut at the length the material asks for lands mid-tone, so the stored span ends on silence."""
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=_HALF_S, looped=False)
    faded = encode(sine(_TONE_HZ, dur=2.0), SR, params, make_encode_ctx(60))
    stepped = encode(sine(_TONE_HZ, dur=2.0), SR, params, make_encode_ctx(60, release_fade_s=0.0))

    assert abs(float(faded.pcm[-1])) < abs(float(stepped.pcm[-1]))
    assert float(faded.pcm[-1]) == pytest.approx(0.0, abs=headroom_peak(0.0) * 2.0**-15)
    assert faded.frames == stepped.frames  # the ramp shapes the span it is given and costs no bytes
    assert faded.release_frames == round(encode_config.release_fade_s * SR)  # ... and the sample records it


def test_a_looped_sample_keeps_the_wrap_point_the_crossfade_made(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    settle: SettleLoop,
) -> None:
    """A loop ends where playback returns to its start, so the span is stored as the crossfade left it."""
    recording = sine(_TONE_HZ, dur=_TONE_S)
    settled = settle(recording, SR, root_hz=_TONE_HZ, search_s=_TONE_S)
    params = EncodingParams(target_rate=SR, depth_bits=16, looped=True)
    faded = encode(recording, SR, params, make_encode_ctx(60, settled=settled))
    unfaded = encode(recording, SR, params, make_encode_ctx(60, release_fade_s=0.0, settled=settled))

    assert faded.loop is not None
    assert np.array_equal(faded.pcm, unfaded.pcm)
    assert faded.release_frames == NO_RELEASE_RAMP  # nothing closes it, so nothing closes its ground truth


def test_a_stored_copy_wraps_the_same_stretch_of_the_recording_at_a_lower_rate(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    settle: SettleLoop,
) -> None:
    """The loop is settled once on the recording, so a cheaper copy reaches it by scaling the bounds."""
    recording = sine(_TONE_HZ, dur=_TONE_S)
    settled = settle(recording, SR, root_hz=_TONE_HZ, search_s=_TONE_S)
    assert settled is not None
    context = make_encode_ctx(60, settled=settled)
    own = encode(recording, SR, EncodingParams(target_rate=SR, depth_bits=16, looped=True), context)
    cheap = encode(recording, SR, EncodingParams(target_rate=11_025, depth_bits=16, looped=True), context)

    assert own.loop == settled.loop  # stored at its own rate, the bounds are the settled ones
    assert cheap.loop is not None
    assert cheap.loop.start == pytest.approx(settled.loop.start / 4, abs=1)
    assert cheap.loop.end == pytest.approx(settled.loop.end / 4, abs=1)


def test_a_clip_the_stage_settled_no_loop_for_stores_the_trimmed_sample(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """Params ask to be looped and the settlement decides whether there is one, so the trim carries the rest."""
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=_HALF_S, looped=True)
    stored = encode(sine(_TONE_HZ, dur=_TONE_S), SR, params, make_encode_ctx(60, settled=NO_LOOP))

    assert stored.loop is None
    assert stored.frames == pytest.approx(int(round(_HALF_S * SR)), abs=2)


def test_non_periodic_material_is_settled_no_loop_and_stored_as_its_trim(
    make_encode_ctx: Callable[..., EncodeContext], settle: SettleLoop
) -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SR)
    settled = settle(noise, SR, root_hz=_TONE_HZ, search_s=1.0)
    assert settled is NO_LOOP  # noise is not periodic enough to loop

    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=0.5, looped=True)
    looped = encode(noise, SR, params, make_encode_ctx(60, settled=settled))
    plain = encode(noise, SR, replace(params, looped=False), make_encode_ctx(60))

    assert looped.loop is None
    assert looped.frames == plain.frames  # ... so it is identical to the non-looped trim


# --- where the level is set ---------------------------------------------------------------------------


def test_normalization_reads_the_span_the_sample_stores(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    encode_config: EncodeConfig,
) -> None:
    """A peak in a stretch the trim discards leaves the stored sample at exactly the level it asked for."""
    signal = np.concatenate([_QUIET * sine(_TONE_HZ, dur=_HALF_S), sine(_TONE_HZ, dur=_HALF_S)])
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=_HALF_S)
    stored = encode(signal, SR, params, make_encode_ctx(60))
    assert peak_amplitude(stored.pcm) == pytest.approx(headroom_peak(encode_config.headroom_db), rel=1e-3)


def test_a_shared_reference_keeps_the_margin_between_two_recordings(
    sine: Callable[..., NDArray[np.float64]], encode_config: EncodeConfig
) -> None:
    """What a format keeping no per-sample gain needs: the quieter recording stays quieter in the PCM."""
    loud = sine(_TONE_HZ)
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
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    settle: SettleLoop,
) -> None:
    """Ordering the two that way is what keeps a looped sample's seam continuous on the audio stored."""
    recording = sine(_TONE_HZ, dur=_TONE_S)
    settled = settle(recording, SR, root_hz=_TONE_HZ, search_s=_TONE_S)
    params = EncodingParams(target_rate=SR, depth_bits=8, looped=True, compress=True, dither=False)
    stored = encode(recording, SR, params, make_encode_ctx(60, settled=settled))
    assert stored.loop is not None
    assert stored.frames == stored.loop.end
