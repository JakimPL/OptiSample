from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.surrogate import (
    TRIMMED,
    EncodeContext,
    EncodingParams,
    StoredSample,
    closed_reference,
    encode,
    output_frame,
    render,
)
from optisample.metrics.diagnostics import snr
from trackmod.spec.levels import MAX_VOLUME

SR = 44_100
_TRIM_S = 0.5
_ROOT = 60
_OCTAVE_UP = 72


def test_encode_render_round_trip_is_near_lossless(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    # Native rate, 16-bit, no dither, no transpose, no release ramp → only 16-bit quantization noise.
    source = sine(440.0)
    normalized = source / float(np.max(np.abs(source)))
    context = make_encode_ctx(60, release_fade_s=0.0)
    stored = encode(source, SR, EncodingParams(target_rate=SR, depth_bits=16, dither=False), context)
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


def test_a_quiet_recording_plays_back_as_quietly_as_it_was_recorded(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """Storing hot spends the depth on one recording; playback undoes it, so the instrument keeps its balance."""
    quiet = 0.25 * sine(440.0)
    stored = encode(quiet, SR, EncodingParams(target_rate=SR, depth_bits=16, dither=False), make_encode_ctx(60))
    rendered = render(stored, SR, pitch=60)
    assert float(np.max(np.abs(rendered))) == pytest.approx(0.25, rel=1e-3)


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
    params = EncodingParams(target_rate=SR, depth_bits=16, loop_choice=0)
    stored = encode(sine(440.0, dur=2.0), SR, params, make_encode_ctx(60))
    held = render(stored, SR, pitch=60, duration_s=3.0)  # far longer than the ~0.1 s stored
    assert held.size == pytest.approx(int(round(3.0 * SR)), abs=1)
    assert float(np.sqrt(np.mean(held[-SR:] ** 2))) > 0.1  # the last second still sounds (loop sustained it)


def _decaying(sine: Callable[..., NDArray[np.float64]], *, half_life_s: float) -> NDArray[np.float64]:
    """Two seconds of a struck note: a steady pitch under an amplitude that falls away as it rings."""
    tone = sine(440.0, dur=2.0)
    return np.exp(-np.arange(tone.size, dtype=np.float64) / (half_life_s * SR)) * tone


def _tail_level(signal: NDArray[np.float64]) -> float:
    """Level of the last quarter second of ``signal`` -- where a held note's decline shows."""
    return float(np.sqrt(np.mean(signal[-SR // 4 :] ** 2)))


def test_a_held_loop_declines_the_way_the_recording_it_stands_for_did(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """A struck note stored as attack plus loop ends near the level its recording ended at."""
    source = _decaying(sine, half_life_s=0.6)
    params = EncodingParams(target_rate=SR, depth_bits=16, loop_choice=0)
    stored = encode(source, SR, params, make_encode_ctx(_ROOT))
    assert stored.decay is not None

    held = render(stored, SR, pitch=_ROOT, duration_s=2.0)
    ringing = render(replace(stored, decay=None), SR, pitch=_ROOT, duration_s=2.0)
    stored_frames = output_frame(stored, stored.frames, SR, _ROOT)

    assert np.allclose(held[:stored_frames], ringing[:stored_frames])  # the stored material sounds as stored
    assert _tail_level(held) == pytest.approx(_tail_level(source), rel=0.5)
    assert _tail_level(ringing) > 5.0 * _tail_level(source)  # the same loop, left to ring at its own level


def test_a_sample_carrying_no_decay_plays_at_the_level_it_was_stored_at(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """A steady recording states no decline, so nothing is put over the loop that sustains it."""
    params = EncodingParams(target_rate=SR, depth_bits=16, loop_choice=0)
    stored = encode(sine(440.0, dur=2.0), SR, params, make_encode_ctx(_ROOT))

    assert stored.decay is None


# --- the ground truth a stored sample is measured against -------------------------------------------


def _stored_half_second(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    *,
    loop_choice: int | None,
) -> StoredSample:
    """Half a second of a steady tone, stored the way ``loop_choice`` asks and ending as that leaves it."""
    params = EncodingParams(target_rate=SR, depth_bits=16, trim_s=_TRIM_S, loop_choice=loop_choice)
    return encode(sine(440.0, dur=2.0), SR, params, make_encode_ctx(_ROOT))


def _closed_unit_reference(stored: StoredSample, pitch: int) -> NDArray[np.float64]:
    """A flat unit ground truth over the whole stretch ``stored`` plays for at ``pitch``, closed as it closes."""
    span = np.ones(output_frame(stored, stored.frames, SR, pitch), dtype=np.float64)
    return closed_reference(span, stored, SR, pitch=pitch)


def _ramped_frames(closed: NDArray[np.float64]) -> int:
    """How many frames of a flat unit ground truth the closing ramp took below full level."""
    return int(np.count_nonzero(closed < 1.0))


def test_the_ground_truth_closes_where_the_stored_material_stops(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """The ramp a stored span stops on stands over the source as well, leaving the codec between them."""
    stored = _stored_half_second(sine, make_encode_ctx, loop_choice=TRIMMED)
    played_frames = output_frame(stored, stored.frames, SR, _ROOT)
    ramp_start = output_frame(stored, stored.frames - stored.release_frames, SR, _ROOT)
    span = np.ones(played_frames, dtype=np.float64)

    closed = closed_reference(span, stored, SR, pitch=_ROOT)

    assert np.array_equal(closed[:ramp_start], span[:ramp_start])  # everything before the ramp is the recording
    assert float(closed[-1]) == 0.0
    assert float(closed[(ramp_start + played_frames) // 2]) == pytest.approx(0.5, abs=0.01)  # linear across it


@pytest.mark.parametrize(
    ("loop_choice", "held"),
    [
        pytest.param(0, 1.0, id="a looped sample wraps at its seam, so nothing closes its ground truth"),
        pytest.param(TRIMMED, 0.5, id="the note ends before the ramp begins"),
    ],
)
def test_the_ground_truth_stands_as_the_recording_where_no_ramp_reaches_it(
    sine: Callable[..., NDArray[np.float64]],
    make_encode_ctx: Callable[..., EncodeContext],
    loop_choice: int | None,
    held: float,
) -> None:
    stored = _stored_half_second(sine, make_encode_ctx, loop_choice=loop_choice)
    span = np.ones(round(held * output_frame(stored, stored.frames, SR, _ROOT)), dtype=np.float64)

    assert np.array_equal(closed_reference(span, stored, SR, pitch=_ROOT), span)


def test_a_note_held_longer_than_its_sample_keeps_the_recording_past_the_ramp(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """The length the ramp closes stays charged, so past the stored material the ground truth is the recording."""
    stored = _stored_half_second(sine, make_encode_ctx, loop_choice=TRIMMED)
    played_frames = output_frame(stored, stored.frames, SR, _ROOT)
    ramp_start = output_frame(stored, stored.frames - stored.release_frames, SR, _ROOT)
    span = np.ones(round(1.5 * played_frames), dtype=np.float64)

    closed = closed_reference(span, stored, SR, pitch=_ROOT)

    assert np.array_equal(closed[:ramp_start], span[:ramp_start])
    assert float(closed[played_frames - 1]) == 0.0  # the ramp reaches silence where the material runs out
    assert np.array_equal(closed[played_frames:], span[played_frames:])  # the note carries on being measured


def test_the_ramp_moves_with_the_material_when_the_sample_is_repitched(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """Playing a sample an octave up runs its closing ramp twice as fast, and the ground truth follows it."""
    stored = _stored_half_second(sine, make_encode_ctx, loop_choice=TRIMMED)
    at_root = _closed_unit_reference(stored, _ROOT)
    an_octave_up = _closed_unit_reference(stored, _OCTAVE_UP)

    assert an_octave_up.size == pytest.approx(at_root.size // 2, abs=1)
    assert _ramped_frames(an_octave_up) == pytest.approx(_ramped_frames(at_root) / 2, abs=2)
