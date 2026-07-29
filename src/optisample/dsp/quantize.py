from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.levels import db_to_gain, peak_amplitude
from optisample.seed import SURROGATE_SEED

Signal = NDArray[np.float64]

VALID_DEPTHS: Final = (8, 16)


def quantization_step(bits: int) -> float:
    """Grid spacing of a signed ``bits``-bit quantizer over ``[-1, 1)`` (one LSB)."""
    if bits not in VALID_DEPTHS:
        raise ValueError(f"IT samples are 8- or 16-bit, got {bits}")

    return 2.0 ** (1 - bits)


def headroom_peak(headroom_db: float) -> float:
    """The peak a stored sample is normalized to: full scale lowered by ``headroom_db``.

    A dithered encode adds up to one step to every sample and the signed grid stops one step below full
    scale, so a sample normalized to full scale meets the clip on its own loudest frame. Keeping the
    peak this far under leaves the dither somewhere to go, and costs ``headroom_db`` of the depth's
    signal-to-noise ratio to do it.
    """
    return db_to_gain(-headroom_db)


def normalize_peak(
    signal: Signal,
    target_peak: float,
    *,
    reference_peak: float | None = None,
) -> tuple[Signal, float]:
    """Scale ``signal`` so ``reference_peak`` lands on ``target_peak``; return ``(scaled, gain)``.

    The reference defaults to the signal's own peak, which stores every clip as hot as its depth
    allows and leaves the level to be restored on playback. Naming the loudest peak of a whole
    instrument instead scales every one of its clips by a single factor, so the balance between them
    survives into the PCM -- what a format keeping no per-sample multiplier needs.

    A reference of zero leaves the signal as it stands, at unit gain.
    """
    data = np.asarray(signal, dtype=np.float64)
    largest = peak_amplitude(data) if reference_peak is None else reference_peak
    if largest <= 0.0:
        return data.copy(), 1.0

    gain = target_peak / largest
    return np.asarray(data * gain, dtype=np.float64), gain


def apply_gain(signal: Signal, gain: float) -> Signal:
    """Scale amplitude by ``gain`` (the makeup gain is ``1 / normalize_peak``'s gain)."""
    return np.asarray(np.asarray(signal, dtype=np.float64) * gain, dtype=np.float64)


def release_fade(signal: Signal, fade_frames: int) -> Signal:
    """``signal`` with its last ``fade_frames`` ramped linearly to silence; returns a copy.

    A stored span is cut at the length its material asks for, which lands mid-decay wherever the note was
    let go before the sound was. Closing on a ramp puts the last frame at silence, so a tracker reaching
    the end of the sample plays it out. A span shorter than the ramp is faded over its whole length, and a
    ramp of no frames leaves the span as it stands.
    """
    data = np.asarray(signal, dtype=np.float64)
    fade = min(fade_frames, data.size)
    if fade <= 0:
        return data.copy()

    faded = data.copy()
    faded[data.size - fade :] *= np.linspace(1.0, 0.0, fade, endpoint=True)
    return faded


def _tpdf_dither(size: int, step: float, rng: np.random.Generator) -> Signal:
    """Triangular-PDF dither in ``(-step, step)`` = the difference of two uniform LSBs."""
    return np.asarray(step * (rng.random(size) - rng.random(size)), dtype=np.float64)


def _quantize_grid(values: Signal, step: float) -> Signal:
    """Round to the signed grid and clip to ``[-1, 1 - step]`` (IT's asymmetric signed range)."""
    grid = np.round(np.asarray(values, dtype=np.float64) / step) * step
    return np.asarray(np.clip(grid, -1.0, 1.0 - step), dtype=np.float64)


def _noise_shape(data: Signal, dither: Signal, step: float) -> Signal:
    """First-order error diffusion: carry the rounding residual forward so its spectrum is high-pass."""
    out = np.empty_like(data)
    carry = 0.0
    for index in range(data.size):
        desired = float(data[index]) + float(dither[index]) + carry
        quantized = float(np.clip(round(desired / step) * step, -1.0, 1.0 - step))
        carry = desired - quantized
        out[index] = quantized

    return out


def requantize(
    signal: Signal,
    bits: int,
    *,
    dither: bool = True,
    noise_shaping: bool = False,
    rng: np.random.Generator | None = None,
) -> Signal:
    """Requantize ``signal`` to ``bits`` bits over ``[-1, 1)`` with optional TPDF dither / noise shaping.

    ``rng`` defaults to a fixed seed so encoding is reproducible; pass one to vary the dither.
    """
    step = quantization_step(bits)
    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return data.copy()

    generator = rng if rng is not None else np.random.default_rng(SURROGATE_SEED)
    noise = (
        _tpdf_dither(data.size, step, generator)
        if dither
        else np.zeros(
            data.size,
            dtype=np.float64,
        )
    )
    if noise_shaping:
        return _noise_shape(data, noise, step)

    return _quantize_grid(data + noise, step)
