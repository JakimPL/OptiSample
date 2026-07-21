"""Encode a recording into a :class:`~optisample.dsp.surrogate.sample.StoredSample`.

Encoding composes the byte-reducing transforms into one stored sample: peak-normalize (store hot),
resample to a lower rate, bound the stored length by looping *or* trimming, and requantize to
8/16-bit with dither/noise shaping.
"""

from __future__ import annotations

from optisample.config.dsp import LoopConfig
from optisample.dsp.loop import Loop, crossfade_loop, detect_loop
from optisample.dsp.quantize import normalize_peak, requantize
from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate.params import EncodeContext, EncodingParams
from optisample.dsp.surrogate.sample import Signal, StoredSample


def _apply_loop(resampled: Signal, rate: int, config: LoopConfig) -> tuple[Signal, Loop | None]:
    """Detect a loop, crossfade its seam, and trim storage to attack + loop (or leave the signal be)."""
    detected = detect_loop(resampled, rate, config)
    if detected is None:
        return resampled, None
    fade_len = int(round(config.crossfade_s * rate))
    faded = crossfade_loop(resampled, detected, fade_len=fade_len)
    return faded[: detected.end], detected


def _loop_or_trim(resampled: Signal, params: EncodingParams, config: LoopConfig) -> tuple[Signal, Loop | None]:
    """Bound stored length by looping *or* trimming -- never both.

    A detected loop already trims storage to ``[0, loop.end)`` (attack + one loop region), so trimming
    on top would cut into or past the loop. Trimming to ``trim_s`` therefore applies only when looping
    was not requested, or was requested but the material was not periodic enough to loop.
    """
    if params.loop:
        looped, loop = _apply_loop(resampled, params.target_rate, config)
        if loop is not None:
            return looped, loop
    if params.trim_s is not None:
        return resampled[: max(0, int(round(params.trim_s * params.target_rate)))], None
    return resampled, None


def encode(signal: Signal, sample_rate: int, params: EncodingParams, ctx: EncodeContext) -> StoredSample:
    """Encode ``signal`` into a :class:`StoredSample`: normalize -> resample -> (loop | trim) -> requantize.

    With ``params.loop`` set, storage is trimmed to the attack plus a looped sustain region (if the
    signal is periodic enough); the loop then sustains notes held past the stored length. Otherwise
    it is trimmed to ``trim_s`` and a longer note simply ends. A loop request on non-periodic material
    silently falls back to the trimmed sample, so the config is never worse than its non-looped twin.
    """
    normalized, gain = normalize_peak(signal, ctx.config.target_peak)
    resampled = resample_to(normalized, sample_rate, params.target_rate)
    resampled, loop = _loop_or_trim(resampled, params, ctx.config.loop)
    pcm = requantize(
        resampled, params.depth_bits, dither=params.dither, noise_shaping=params.noise_shaping, rng=ctx.rng
    )
    return StoredSample(
        pcm=pcm,
        sample_rate=params.target_rate,
        depth_bits=params.depth_bits,
        root_pitch=ctx.root_pitch,
        gain=gain,
        loop=loop,
    )
