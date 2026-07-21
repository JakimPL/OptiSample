from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode

SR = 44_100


def test_encode_sets_rate_depth_and_gain(
    sine: Callable[..., NDArray[np.float64]], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    stored = encode(0.25 * sine(440.0), SR, EncodingParams(target_rate=22_050, depth_bits=8), make_encode_ctx(57))
    assert stored.sample_rate == 22_050
    assert stored.depth_bits == 8
    assert stored.root_pitch == 57
    assert stored.frames == pytest.approx(int(round(SR * 1.0 * 22_050 / SR)), abs=2)
    assert stored.gain == pytest.approx(1.0 / 0.25, rel=1e-6)  # normalized to full scale


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
