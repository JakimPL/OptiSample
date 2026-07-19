from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import EncodingParams
from optisample.metrics.composite import CompositeFidelity
from optisample.metrics.size import SampleSize
from optisample.optimize import (
    OperatingPoint,
    SourceClip,
    default_rates,
    evaluate_encoding,
    lower_convex_hull,
    rd_frontier,
    sample_operating_points,
)
from optisample.synth import NoteSpec, default_synth_config, render_sample

SR = 44_100


def bright_piano(pitch: int = 84, velocity: int = 115, dur: float = 1.0) -> np.ndarray:
    spec = NoteSpec(pitch=pitch, velocity=velocity, controller=60.0, duration_s=dur, sample_rate=SR)
    return render_sample("piano", spec, np.random.default_rng(1), default_synth_config())


def point(stored_bytes: int, distortion: float) -> OperatingPoint:
    params = EncodingParams(target_rate=1, depth_bits=16)
    return OperatingPoint(params=params, stored_bytes=stored_bytes, distortion=distortion, frames=stored_bytes)


# 245 Hz has an exact 180-frame period at 44.1 kHz, so a whole-period loop reproduces it exactly
# (a non-integer period would make the loop play a slightly detuned pitch -- a real limitation, not a bug).
def harmonic_tone(freq: float = 245.0, dur: float = 3.0) -> np.ndarray:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    )


def test_looping_a_periodic_clip_saves_bytes_at_similar_quality(
    sweep: Callable[..., SweepConfig], composite: CompositeFidelity, encode_config: EncodeConfig
) -> None:
    clip = SourceClip(signal=harmonic_tone(dur=3.0), sample_rate=SR, root_pitch=57, duration_s=3.0)
    plain_sweep = sweep(rates=(SR,), depths=(16,), dither=False, loops=(False,))
    loop_sweep = sweep(rates=(SR,), depths=(16,), dither=False, loops=(True,))
    plain = sample_operating_points(clip, plain_sweep, composite=composite, encode_config=encode_config)[0]
    looped = sample_operating_points(clip, loop_sweep, composite=composite, encode_config=encode_config)[0]
    assert looped.params.loop is True
    assert looped.stored_bytes < plain.stored_bytes // 2  # dropping the 3 s sustain tail is a big saving
    assert looped.distortion < 0.1  # the whole-period loop reconstructs the exactly-periodic tone


def test_looping_is_pareto_optimal_on_the_frontier_when_it_helps(
    sweep: Callable[..., SweepConfig], composite: CompositeFidelity, encode_config: EncodeConfig
) -> None:
    clip = SourceClip(signal=harmonic_tone(dur=3.0), sample_rate=SR, root_pitch=57, duration_s=3.0)
    grid = sweep(rates=(SR, 11_025), depths=(16, 8), dither=False, loops=(False, True))
    hull = rd_frontier(clip, grid, composite=composite, encode_config=encode_config)
    assert any(op.params.loop for op in hull)  # a looped config survives onto the rate-distortion hull


def test_default_rates_are_capped_floored_and_sorted(sweep_config: SweepConfig) -> None:
    divisors, floor = sweep_config.rate_divisors, sweep_config.min_rate
    assert default_rates(44_100, divisors, floor) == [44_100, 22_050, 14_700, 11_025, 7_350, 5_512]
    assert default_rates(8_000, divisors, floor) == [8_000, 4_000]  # low divisors clamp to the floor and dedupe
    assert all(rate <= 44_100 for rate in default_rates(44_100, divisors, floor))


def test_evaluate_encoding_lossless_beats_aggressive(composite: CompositeFidelity, encode_config: EncodeConfig) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    params = {"composite": composite, "encode_config": encode_config}
    lossless = evaluate_encoding(clip, EncodingParams(target_rate=SR, depth_bits=16, dither=False), **params)
    aggressive = evaluate_encoding(clip, EncodingParams(target_rate=5_512, depth_bits=8), **params)
    assert lossless.distortion < aggressive.distortion
    assert lossless.stored_bytes > aggressive.stored_bytes


def test_evaluate_encoding_without_duration_stores_full_clip(
    composite: CompositeFidelity, encode_config: EncodeConfig
) -> None:
    clip = SourceClip(signal=bright_piano(dur=0.5), sample_rate=SR, root_pitch=84)  # duration_s=None → no trim
    result = evaluate_encoding(
        clip,
        EncodingParams(target_rate=SR, depth_bits=16, dither=False),
        composite=composite,
        encode_config=encode_config,
    )
    assert result.frames == pytest.approx(int(0.5 * SR), abs=2)
    assert result.kib == pytest.approx(result.stored_bytes / 1024.0)


def test_evaluate_encoding_bytes_match_size_model(composite: CompositeFidelity, encode_config: EncodeConfig) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    result = evaluate_encoding(
        clip, EncodingParams(target_rate=22_050, depth_bits=16), composite=composite, encode_config=encode_config
    )
    assert result.stored_bytes == SampleSize(frames=result.frames, depth_bits=16).total_bytes


def test_sample_operating_points_covers_the_grid(
    sweep: Callable[..., SweepConfig], composite: CompositeFidelity, encode_config: EncodeConfig
) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    grid = sweep(rates=(44_100, 22_050, 11_025), depths=(16, 8))
    points = sample_operating_points(
        clip, grid, composite=composite, encode_config=encode_config, rng=np.random.default_rng(0)
    )
    assert len(points) == 3 * 2
    assert all(p.stored_bytes > 0 for p in points)


def test_lower_convex_hull_drops_point_above_chord() -> None:
    hull = lower_convex_hull([point(100, 1.0), point(200, 0.5), point(300, 0.45), point(400, 0.1)])
    assert [p.stored_bytes for p in hull] == [100, 200, 400]  # (300, 0.45) sits above the 200→400 chord


def test_lower_convex_hull_drops_dominated_point() -> None:
    hull = lower_convex_hull([point(100, 1.0), point(200, 0.5), point(300, 0.6)])
    assert [p.stored_bytes for p in hull] == [100, 200]  # more bytes AND more distortion → dominated


def test_lower_convex_hull_drops_collinear_midpoint() -> None:
    hull = lower_convex_hull([point(0, 3.0), point(10, 2.0), point(20, 1.0)])
    assert [p.stored_bytes for p in hull] == [0, 20]  # the redundant middle vertex is removed


def test_lower_convex_hull_edge_cases() -> None:
    assert lower_convex_hull([]) == []
    solo = point(50, 0.2)
    assert lower_convex_hull([solo]) == [solo]


def test_rd_frontier_is_monotone_and_convex(
    sweep_config: SweepConfig, composite: CompositeFidelity, encode_config: EncodeConfig
) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    hull = rd_frontier(
        clip, sweep_config, composite=composite, encode_config=encode_config, rng=np.random.default_rng(0)
    )
    stored_bytes = [p.stored_bytes for p in hull]
    distortion = [p.distortion for p in hull]
    assert len(hull) >= 2
    assert all(a < b for a, b in zip(stored_bytes, stored_bytes[1:]))  # bytes strictly increase
    assert all(a > b for a, b in zip(distortion, distortion[1:]))  # distortion strictly decreases
    slopes = [
        (distortion[i + 1] - distortion[i]) / (stored_bytes[i + 1] - stored_bytes[i]) for i in range(len(hull) - 1)
    ]
    assert all(a < b for a, b in zip(slopes, slopes[1:]))  # slopes increase → convex (diminishing returns)
