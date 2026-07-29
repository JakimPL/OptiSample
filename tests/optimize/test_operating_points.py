from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from optisample.config import load_config
from optisample.config.optimize import TRIMMED_ONLY, SweepConfig
from optisample.dsp.surrogate import TRIMMED, EncodingParams
from optisample.optimize.operating_points import (
    OperatingPoint,
    SourceClip,
    SweepContext,
    compression_at,
    evaluate_encoding,
    lower_convex_hull,
    rd_frontier,
    sample_operating_points,
    sweep_param_grid,
    sweep_rates,
)
from optisample.synth import NoteSpec, synthesize
from trackmod.core.samples.depth import BitDepth

SR = 44_100

# synthesize is a test-signal generator here; its synth config is fixture-independent test data.
_SYNTH = load_config().synth


def bright_piano(pitch: int = 84, velocity: int = 115, dur: float = 1.0) -> np.ndarray:
    spec = NoteSpec(pitch=pitch, velocity=velocity, controller=60.0, duration_s=dur, sample_rate=SR)
    return synthesize("piano", spec, np.random.default_rng(1), _SYNTH)


def point(stored_bytes: int, distortion: float) -> OperatingPoint:
    params = EncodingParams(target_rate=1, depth_bits=16)
    return OperatingPoint(params=params, stored_bytes=stored_bytes, distortion=distortion, frames=stored_bytes)


def seeded(context: SweepContext, seed: int = 0) -> SweepContext:
    """The same sweep context with a seeded dither RNG, so a sweep reproduces run to run."""
    return dataclasses.replace(context, rng=np.random.default_rng(seed))


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
    sweep: Callable[..., SweepConfig], sweep_context: SweepContext
) -> None:
    clip = SourceClip(signal=harmonic_tone(dur=3.0), sample_rate=SR, root_pitch=57, duration_s=3.0)
    grid = sweep(rates=(SR,), depths=(16,), dither=False, loop_choices=1)
    plain, looped = sample_operating_points(clip, grid, sweep_context)
    assert plain.params.loop_choice is TRIMMED
    assert looped.params.loop_choice == 0
    assert looped.stored_bytes < plain.stored_bytes // 2  # dropping the 3 s sustain tail is a big saving
    assert looped.distortion < 0.1  # the whole-period loop reconstructs the exactly-periodic tone


def test_looping_is_pareto_optimal_on_the_frontier_when_it_helps(
    sweep: Callable[..., SweepConfig], sweep_context: SweepContext
) -> None:
    clip = SourceClip(signal=harmonic_tone(dur=3.0), sample_rate=SR, root_pitch=57, duration_s=3.0)
    grid = sweep(rates=(SR, 11_025), depths=(16, 8), dither=False, loop_choices=1)
    hull = rd_frontier(clip, grid, sweep_context)
    assert any(op.params.loop_choice is not None for op in hull)  # a looped config survives onto the hull


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
    lossless = evaluate_encoding(clip, EncodingParams(target_rate=SR, depth_bits=16, dither=False), sweep_context)
    aggressive = evaluate_encoding(clip, EncodingParams(target_rate=5_512, depth_bits=8), sweep_context)
    assert lossless.distortion < aggressive.distortion
    assert lossless.stored_bytes > aggressive.stored_bytes


def test_evaluate_encoding_without_duration_stores_full_clip(sweep_context: SweepContext) -> None:
    clip = SourceClip(signal=bright_piano(dur=0.5), sample_rate=SR, root_pitch=84)  # duration_s=None → no trim
    result = evaluate_encoding(clip, EncodingParams(target_rate=SR, depth_bits=16, dither=False), sweep_context)
    assert result.frames == pytest.approx(int(0.5 * SR), abs=2)
    assert result.kib == pytest.approx(result.stored_bytes / 1024.0)


def test_evaluate_encoding_bytes_match_the_formats_cost_table(sweep_context: SweepContext) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    result = evaluate_encoding(clip, EncodingParams(target_rate=22_050, depth_bits=16), sweep_context)
    assert result.stored_bytes == sweep_context.storage.sample_bytes(frames=result.frames, depth=BitDepth.SIXTEEN)


def test_a_shallow_depth_is_swept_over_every_compression_the_config_asks_for(
    sweep: Callable[..., SweepConfig],
) -> None:
    assert compression_at(sweep(compress=(False, True)), 8) == (False, True)


def test_a_deep_depth_is_swept_uncompressed_once(sweep: Callable[..., SweepConfig]) -> None:
    """Sixteen bits leave the quantizer's floor below anything compression could protect, so it is skipped."""
    assert compression_at(sweep(compress=(False, True)), 16) == (False,)


def test_the_grid_enumerates_compression_only_where_it_is_swept(sweep: Callable[..., SweepConfig]) -> None:
    grid = sweep(rates=(SR,), depths=(16, 8), compress=(False, True), loop_choices=TRIMMED_ONLY)
    params = list(sweep_param_grid(grid, SR, trim_s=None))
    assert [(point.depth_bits, point.compress) for point in params] == [(16, False), (8, False), (8, True)]


def test_the_grid_offers_the_trimmed_sample_beside_every_loop_candidate(
    sweep: Callable[..., SweepConfig],
) -> None:
    """Storing no loop is an option of its own, so the frontier prices a loop against going without."""
    grid = sweep(rates=(SR,), depths=(16,), loop_choices=2)
    params = list(sweep_param_grid(grid, SR, trim_s=None))

    assert [point.loop_choice for point in params] == [TRIMMED, 0, 1]


def test_sample_operating_points_covers_the_grid(
    sweep: Callable[..., SweepConfig], sweep_context: SweepContext
) -> None:
    """One point per grid entry: three rates once at 16 bits, and both compressions of them at 8."""
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    grid = sweep(rates=(44_100, 22_050, 11_025), depths=(16, 8), compress=(False, True), loop_choices=TRIMMED_ONLY)
    points = sample_operating_points(clip, grid, seeded(sweep_context))
    assert len(points) == 3 + 3 * 2
    assert all(p.stored_bytes > 0 for p in points)


@dataclass(frozen=True)
class _HullCase:
    """A lower-convex-hull scenario: the ``(stored_bytes, distortion)`` points fed in and the bytes kept."""

    name: str
    points: tuple[tuple[int, float], ...]
    kept: list[int]


_HULL_CASES = (
    _HullCase("above-chord vertex dropped", ((100, 1.0), (200, 0.5), (300, 0.45), (400, 0.1)), [100, 200, 400]),
    _HullCase("dominated vertex dropped", ((100, 1.0), (200, 0.5), (300, 0.6)), [100, 200]),
    _HullCase("collinear midpoint dropped", ((0, 3.0), (10, 2.0), (20, 1.0)), [0, 20]),
)


@pytest.mark.parametrize("case", _HULL_CASES, ids=lambda case: case.name)
def test_lower_convex_hull_keeps_only_frontier_vertices(case: _HullCase) -> None:
    hull = lower_convex_hull([point(stored_bytes, distortion) for stored_bytes, distortion in case.points])
    assert [p.stored_bytes for p in hull] == case.kept


def test_lower_convex_hull_edge_cases() -> None:
    assert lower_convex_hull([]) == []
    solo = point(50, 0.2)
    assert lower_convex_hull([solo]) == [solo]


def test_rd_frontier_is_monotone_and_convex(sweep_config: SweepConfig, sweep_context: SweepContext) -> None:
    clip = SourceClip(signal=bright_piano(), sample_rate=SR, root_pitch=84, duration_s=1.0)
    hull = rd_frontier(clip, sweep_config, seeded(sweep_context))
    stored_bytes = [p.stored_bytes for p in hull]
    distortion = [p.distortion for p in hull]
    assert len(hull) >= 2
    assert all(a < b for a, b in zip(stored_bytes, stored_bytes[1:]))  # bytes strictly increase
    assert all(a > b for a, b in zip(distortion, distortion[1:]))  # distortion strictly decreases
    slopes = [
        (distortion[i + 1] - distortion[i]) / (stored_bytes[i + 1] - stored_bytes[i]) for i in range(len(hull) - 1)
    ]
    assert all(a < b for a, b in zip(slopes, slopes[1:]))  # slopes increase → convex (diminishing returns)
