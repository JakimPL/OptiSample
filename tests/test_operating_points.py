from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.surrogate import EncodingParams
from optisample.metrics.size import SampleSize
from optisample.optimize import (
    OperatingPoint,
    SourceClip,
    SweepGrid,
    default_rates,
    evaluate_encoding,
    lower_convex_hull,
    rd_frontier,
    sample_operating_points,
)
from optisample.synth import NoteSpec, render_sample

SR = 44_100


def bright_piano(pitch: int = 84, velocity: int = 115, dur: float = 1.0) -> np.ndarray:
    spec = NoteSpec(pitch=pitch, velocity=velocity, controller=60.0, duration_s=dur, sample_rate=SR)
    return render_sample("piano", spec, np.random.default_rng(1))


def point(stored_bytes: int, distortion: float) -> OperatingPoint:
    params = EncodingParams(target_rate=1, depth_bits=16)
    return OperatingPoint(params=params, stored_bytes=stored_bytes, distortion=distortion, frames=stored_bytes)


def test_default_rates_are_capped_floored_and_sorted() -> None:
    assert default_rates(44_100) == [44_100, 22_050, 14_700, 11_025, 7_350, 5_512]
    assert default_rates(8_000) == [8_000, 4_000]  # low divisors clamp to the floor and dedupe
    assert all(rate <= 44_100 for rate in default_rates(44_100))


def test_evaluate_encoding_lossless_beats_aggressive() -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    lossless = evaluate_encoding(clip, EncodingParams(target_rate=SR, depth_bits=16, dither=False))
    aggressive = evaluate_encoding(clip, EncodingParams(target_rate=5_512, depth_bits=8))
    assert lossless.distortion < aggressive.distortion
    assert lossless.stored_bytes > aggressive.stored_bytes


def test_evaluate_encoding_without_duration_stores_full_clip() -> None:
    clip = SourceClip(signal=bright_piano(dur=0.5), sample_rate=SR, root_pitch=84)  # duration_s=None → no trim
    result = evaluate_encoding(clip, EncodingParams(target_rate=SR, depth_bits=16, dither=False))
    assert result.frames == pytest.approx(int(0.5 * SR), abs=2)
    assert result.kib == pytest.approx(result.stored_bytes / 1024.0)


def test_evaluate_encoding_bytes_match_size_model() -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    result = evaluate_encoding(clip, EncodingParams(target_rate=22_050, depth_bits=16))
    assert result.stored_bytes == SampleSize(frames=result.frames, depth_bits=16).total_bytes


def test_sample_operating_points_covers_the_grid() -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    grid = SweepGrid(rates=(44_100, 22_050, 11_025), depths=(16, 8))
    points = sample_operating_points(clip, grid, rng=np.random.default_rng(0))
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


def test_rd_frontier_is_monotone_and_convex() -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    hull = rd_frontier(clip, rng=np.random.default_rng(0))
    stored_bytes = [p.stored_bytes for p in hull]
    distortion = [p.distortion for p in hull]
    assert len(hull) >= 2
    assert all(a < b for a, b in zip(stored_bytes, stored_bytes[1:]))  # bytes strictly increase
    assert all(a > b for a, b in zip(distortion, distortion[1:]))  # distortion strictly decreases
    slopes = [
        (distortion[i + 1] - distortion[i]) / (stored_bytes[i + 1] - stored_bytes[i]) for i in range(len(hull) - 1)
    ]
    assert all(a < b for a, b in zip(slopes, slopes[1:]))  # slopes increase → convex (diminishing returns)
