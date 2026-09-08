from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from trackmod import BitDepth

from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import UNLOOPED, EncodingParams, SettledLoops
from optisample.frontier import lower_convex_hull
from optisample.optimize.operating_points import (
    SourceClip,
    SweepContext,
    compresses,
    evaluate_encoding,
    sweep_rates,
)
from optisample.synth import NoteSpec, synthesize
from tests.conftest import TEST_CONFIG_DIR

SR = 44_100
_HELD_S = 3.0  # a pad long enough for the stage to place a loop inside and still leave a sustain tail

SettleLoops = Callable[..., SettledLoops]
_CHEAPEST = 0  # the offer a test reaches for: the shortest region the recording supports

# synthesize is a test-signal generator here; its synth config is fixture-independent test data.
_SYNTH = load_config(TEST_CONFIG_DIR).synth


def bright_piano(pitch: int = 84, velocity: int = 115, dur: float = 1.0) -> np.ndarray:
    spec = NoteSpec(pitch=pitch, velocity=velocity, controller=60.0, duration_s=dur, sample_rate=SR)
    return synthesize("piano", spec, np.random.default_rng(1), _SYNTH)


def seeded(context: SweepContext, seed: int = 0) -> SweepContext:
    """The same sweep context with a seeded dither RNG, so a sweep reproduces run to run."""
    return dataclasses.replace(context, rng=np.random.default_rng(seed))


_PERIODIC_HZ = 245.0


# 245 Hz has an exact 180-frame period at 44.1 kHz, so a whole-period loop reproduces it exactly
# (a non-integer period would make the loop play a slightly detuned pitch -- a real limitation, not a bug).
def harmonic_tone(freq: float = _PERIODIC_HZ, dur: float = 3.0) -> np.ndarray:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    )


def stored_as(rate: int, *, loop_index: int | None = UNLOOPED, trim_s: float | None = None) -> EncodingParams:
    """One encoding as the reduction settles it: a stored rate at 16 bits, held for ``trim_s``."""
    return EncodingParams(target_rate=rate, depth=BitDepth.SIXTEEN, trim_s=trim_s, dither=False, loop_index=loop_index)


def _looped_clip(settle: SettleLoops) -> SourceClip:
    """A periodic pad with the loops the stage offers for it, one of which a looped encoding is stored around."""
    signal = harmonic_tone(dur=_HELD_S)
    return SourceClip(
        signal=signal,
        sample_rate=SR,
        root_pitch=57,
        duration_s=_HELD_S,
        settled=settle(signal, SR, root_hz=_PERIODIC_HZ, search_s=_HELD_S),
    )


def test_looping_a_periodic_clip_saves_bytes_at_similar_quality(
    sweep_context: SweepContext, settle: SettleLoops
) -> None:
    clip = _looped_clip(settle)
    plain = evaluate_encoding(clip, stored_as(SR, trim_s=_HELD_S), sweep_context)
    looped = evaluate_encoding(clip, stored_as(SR, loop_index=_CHEAPEST, trim_s=_HELD_S), sweep_context)
    assert looped.stored_bytes < plain.stored_bytes // 2  # dropping the 3 s sustain tail is a big saving
    assert looped.distortion < 0.1  # the whole-period loop reconstructs the exactly-periodic tone


def test_looping_is_pareto_optimal_on_the_frontier_when_it_helps(
    sweep_context: SweepContext, settle: SettleLoops
) -> None:
    clip = _looped_clip(settle)
    offered = [
        stored_as(rate, loop_index=index, trim_s=_HELD_S) for rate in (SR, 11_025) for index in (UNLOOPED, _CHEAPEST)
    ]
    hull = lower_convex_hull([evaluate_encoding(clip, params, sweep_context) for params in offered])
    assert any(op.params.loop_index is not None for op in hull)  # a looped config survives onto the hull


def test_each_loop_a_clip_offers_is_an_operating_point_of_its_own(
    sweep_context: SweepContext, settle: SettleLoops
) -> None:
    """Loop length is a rate axis, so every offer has to reach the hull as its own cost/quality point."""
    clip = _looped_clip(settle)
    assert len(clip.settled) > 1

    points = [
        evaluate_encoding(clip, stored_as(SR, loop_index=index, trim_s=_HELD_S), sweep_context)
        for index in range(len(clip.settled))
    ]

    assert len({point.stored_bytes for point in points}) == len(points)
    assert [point.stored_bytes for point in points] == sorted(point.stored_bytes for point in points)


def test_the_swept_rates_are_the_ladder_a_recording_reaches_down_to(sweep: Callable[..., SweepConfig]) -> None:
    """A clip is offered the listed rates it can reach, highest first, its own among them."""
    ladder = sweep(rates=(8_000, 16_000, 22_050, 44_100))

    assert sweep_rates(ladder, 48_000) == [48_000, 44_100, 22_050, 16_000, 8_000]


def test_a_recording_is_offered_no_rate_above_its_own(sweep: Callable[..., SweepConfig]) -> None:
    """Resampling upward spends bytes on a band the recording never held, so the ladder stops at it."""
    ladder = sweep(rates=(8_000, 16_000, 22_050, 44_100))

    assert sweep_rates(ladder, 22_050) == [22_050, 16_000, 8_000]


def test_a_recording_at_a_listed_rate_is_offered_it_once(sweep: Callable[..., SweepConfig]) -> None:
    """The clip's own rate and the ladder's entry for it are one candidate, so the grid holds no duplicate."""
    ladder = sweep(rates=(8_000, 16_000, 22_050))

    assert sweep_rates(ladder, 16_000) == [16_000, 8_000]


def test_evaluate_encoding_lossless_beats_aggressive(sweep_context: SweepContext) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    lossless = evaluate_encoding(
        clip, EncodingParams(target_rate=SR, depth=BitDepth.SIXTEEN, dither=False), sweep_context
    )
    aggressive = evaluate_encoding(clip, EncodingParams(target_rate=5_512, depth=BitDepth.EIGHT), sweep_context)
    assert lossless.distortion < aggressive.distortion
    assert lossless.stored_bytes > aggressive.stored_bytes


def test_evaluate_encoding_without_duration_stores_full_clip(sweep_context: SweepContext) -> None:
    clip = SourceClip(signal=bright_piano(dur=0.5), sample_rate=SR, root_pitch=84)  # duration_s=None → no trim
    result = evaluate_encoding(
        clip, EncodingParams(target_rate=SR, depth=BitDepth.SIXTEEN, dither=False), sweep_context
    )
    assert result.frames == pytest.approx(int(0.5 * SR), abs=2)
    assert result.kib == pytest.approx(result.stored_bytes / 1024.0)


def test_evaluate_encoding_bytes_match_the_formats_cost_table(sweep_context: SweepContext) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    result = evaluate_encoding(clip, EncodingParams(target_rate=22_050, depth=BitDepth.SIXTEEN), sweep_context)
    assert result.stored_bytes == sweep_context.storage.sample_bytes(frames=result.frames, depth=BitDepth.SIXTEEN)


def test_a_shallow_depth_is_stored_compressed_where_the_config_asks_for_it(
    sweep: Callable[..., SweepConfig],
) -> None:
    assert compresses(sweep(compress=True), 8) is True


def test_a_deep_depth_keeps_the_waveform_as_recorded(sweep: Callable[..., SweepConfig]) -> None:
    """Sixteen bits leave the quantizer's floor below anything compression could protect, so it is left off."""
    assert compresses(sweep(compress=True), 16) is False


def test_a_config_asking_for_no_compression_leaves_every_depth_alone(
    sweep: Callable[..., SweepConfig],
) -> None:
    assert compresses(sweep(compress=False), 8) is False


def test_a_measured_frontier_is_monotone_and_convex(sweep_context: SweepContext) -> None:
    """Read over encodings the surrogate actually scored, so the hull's shape is a measured property."""
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    context = seeded(sweep_context)
    offered = [stored_as(rate, trim_s=1.0) for rate in (8_000, 11_025, 22_050, 44_100)]
    hull = lower_convex_hull([evaluate_encoding(clip, params, context) for params in offered])
    stored_bytes = [p.stored_bytes for p in hull]
    distortion = [p.distortion for p in hull]
    assert len(hull) >= 2
    assert all(a < b for a, b in zip(stored_bytes, stored_bytes[1:]))  # bytes strictly increase
    assert all(a > b for a, b in zip(distortion, distortion[1:]))  # distortion strictly decreases
    slopes = [
        (distortion[i + 1] - distortion[i]) / (stored_bytes[i + 1] - stored_bytes[i]) for i in range(len(hull) - 1)
    ]
    assert all(a < b for a, b in zip(slopes, slopes[1:]))  # slopes increase → convex (diminishing returns)
