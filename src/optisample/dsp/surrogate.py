"""Fast numpy "surrogate" codec: encode a recording into an IT-style stored sample and render notes
back from it, standing in for the real tracker in the optimizer's inner loop.

Encoding composes the byte-reducing transforms -- peak-normalize (store hot), resample to a lower
rate, trim to only the duration the material needs, and requantize to 8/16-bit with dither/noise
shaping -- into a :class:`StoredSample`. Rendering reads that sample back at a target pitch (a
resample by ``2**(semitones / 12)``, the way a tracker repitches), scales by the note volume
(``0..64``, linear per IT), and fits the note to a requested duration.

Looping (P6) lets a short stored sample sustain a long note: with ``params.loop`` set, storage is
trimmed to the attack plus a looped region and :func:`render` repeats that region to fill the
duration; without a loop, a note held longer than the sample is zero-padded (it simply ends).
Volume-envelope support is still deferred. The surrogate is calibrated against ``openmpt123`` in
P4 -- until then it *is* the objective the rate-distortion search optimizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from optisample.config import load_config
from optisample.config.dsp import EncodeConfig, LoopConfig
from optisample.dsp.loop import Loop, crossfade_loop, detect_loop
from optisample.dsp.quantize import normalize_peak, requantize
from optisample.dsp.resample import resample_num, resample_to
from optisample.metrics.size import SampleSize

Signal = NDArray[np.float64]

MAX_VOLUME = 64  # IT note volume spans 0x00..0x40 (linear in amplitude).


@dataclass(frozen=True)
class EncodingParams:
    """One encoding configuration for a stored sample (the per-sample decision variables of P2)."""

    target_rate: int
    depth_bits: int
    trim_s: float | None = None
    dither: bool = True
    noise_shaping: bool = False
    loop: bool = False  # store attack + a looped sustain region instead of the whole trimmed sample


@dataclass(frozen=True)
class StoredSample:
    """An encoded, IT-ready sample: its PCM, stored rate/depth, natural pitch, and applied gain."""

    pcm: Signal
    sample_rate: int
    depth_bits: int
    root_pitch: int
    gain: float = 1.0  # normalization gain applied at encode time (stored = source * gain).
    loop: Loop | None = None  # forward loop over stored frames, sustaining notes held past the sample

    @property
    def frames(self) -> int:
        return int(self.pcm.size)

    @property
    def size(self) -> SampleSize:
        return SampleSize(frames=self.frames, depth_bits=self.depth_bits)

    @property
    def stored_bytes(self) -> int:
        return self.size.total_bytes

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


@dataclass(frozen=True)
class EncodeContext:
    """What :func:`encode` needs around one swept ``EncodingParams`` point: the root pitch to stamp on
    the stored sample, the encode config (loop detection + normalization peak), and the dither RNG."""

    root_pitch: int
    config: EncodeConfig
    rng: np.random.Generator | None = None


@lru_cache(maxsize=1)
def default_encode_config() -> EncodeConfig:
    """The bundled encode config, cached so the sweep does not reload YAML per operating point.

    Transitional bridge for call sites that do not yet thread an ``EncodeConfig``
    (operating_points/orchestrate/grouping -> phase 6, export -> phase 7, artifacts -> phase 9);
    removed once every caller passes config explicitly.
    """
    return load_config().encode


def semitone_ratio(semitones: float) -> float:
    """Playback speed / frequency ratio for a pitch shift of ``semitones`` (12 semitones = 2x)."""
    return float(2.0 ** (semitones / 12.0))


def _fit_length(signal: Signal, length: int) -> Signal:
    """Truncate or zero-pad ``signal`` to exactly ``length`` samples."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if signal.size == length:
        return signal
    if signal.size > length:
        return np.asarray(signal[:length], dtype=np.float64)
    return np.asarray(np.pad(signal, (0, length - signal.size)), dtype=np.float64)


def _apply_loop(resampled: Signal, rate: int, config: LoopConfig) -> tuple[Signal, Loop | None]:
    """Detect a loop, crossfade its seam, and trim storage to attack + loop (or leave the signal be)."""
    detected = detect_loop(resampled, rate, config)
    if detected is None:
        return resampled, None
    fade_len = int(round(config.crossfade_s * rate))
    faded = crossfade_loop(resampled, detected, fade_len=fade_len)
    return faded[: detected.end], detected


def encode(signal: Signal, sample_rate: int, params: EncodingParams, ctx: EncodeContext) -> StoredSample:
    """Encode ``signal`` into a :class:`StoredSample`: normalize -> resample -> (loop | trim) -> requantize.

    With ``params.loop`` set, storage is trimmed to the attack plus a looped sustain region (if the
    signal is periodic enough); the loop then sustains notes held past the stored length. Otherwise
    it is trimmed to ``trim_s`` and a longer note simply ends. A loop request on non-periodic material
    silently falls back to the trimmed sample, so the config is never worse than its non-looped twin.
    """
    normalized, gain = normalize_peak(signal, ctx.config.target_peak)
    resampled = resample_to(normalized, sample_rate, params.target_rate)
    loop: Loop | None = None
    if params.loop:
        resampled, loop = _apply_loop(resampled, params.target_rate, ctx.config.loop)
    if loop is None and params.trim_s is not None:
        resampled = resampled[: max(0, int(round(params.trim_s * params.target_rate)))]
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


def _sustain_with_loop(played: Signal, loop: Loop, scale: float, target: int) -> Signal:
    """Extend ``played`` to ``target`` frames by repeating its loop region (mapped to the output rate)."""
    start = max(0, min(int(round(loop.start * scale)), played.size))
    end = max(start + 1, min(int(round(loop.end * scale)), played.size))
    segment = played[start:end]
    if segment.size == 0 or target <= end:
        return played
    repeats = int(np.ceil((target - end) / segment.size))
    tail = np.tile(segment, repeats)[: target - end]
    return np.concatenate([played[:end], tail])


def render(
    stored: StoredSample,
    out_rate: int,
    *,
    pitch: int | None = None,
    volume: int = MAX_VOLUME,
    duration_s: float | None = None,
) -> Signal:
    """Render a note from ``stored`` at ``out_rate``: repitch to ``pitch``, scale by ``volume``, fit duration.

    ``pitch`` defaults to the sample's root (no transpose). Repitching plays the sample faster/slower
    (``2**((pitch - root) / 12)``), which shifts both pitch and length the way a tracker does. If the
    sample carries a loop and the note is held past the stored length, the loop region is repeated to
    sustain it (in the output domain, so it tracks the repitch); otherwise the note simply ends.
    """
    transpose = 0.0 if pitch is None else float(pitch - stored.root_pitch)
    effective_rate = stored.sample_rate * semitone_ratio(transpose)
    scale = out_rate / effective_rate if effective_rate > 0.0 else 0.0
    played = resample_num(stored.pcm, int(round(stored.frames * scale)))
    if duration_s is not None and stored.loop is not None:
        target = int(round(duration_s * out_rate))
        if target > played.size:
            played = _sustain_with_loop(played, stored.loop, scale, target)
    rendered = played * (volume / MAX_VOLUME)
    if duration_s is not None:
        rendered = _fit_length(rendered, int(round(duration_s * out_rate)))
    return np.asarray(rendered, dtype=np.float64)
