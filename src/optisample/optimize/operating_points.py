from dataclasses import dataclass
from typing import Final

import numpy as np
from trackmod.module.storage import Storage

from optisample.config.codec import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import (
    NO_LOOPS,
    EncodeContext,
    EncodingParams,
    SettledLoops,
    Signal,
    StoredSample,
    closed_reference,
    encode,
    render,
)
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.composite import CompositeFidelity, evaluate
from optisample.metrics.size import bytes_to_kib

_COMPRESSIBLE_DEPTH: Final = 8  # bits; a deeper grid's noise floor sits below what compression protects


@dataclass(frozen=True, eq=False)
class SourceClip:
    """A recording to encode, its rate/natural pitch, the longest duration the material needs, and its loops.

    ``settled`` is the frontier of loops the stage found for this recording, one of which each looped
    encoding is stored around: the sweep prices keeping the played span against keeping the attack plus
    each of them.
    """

    signal: Signal
    sample_rate: int
    root_pitch: int
    duration_s: float | None = None
    settled: SettledLoops = NO_LOOPS


@dataclass(frozen=True)
class SweepContext:
    """What scoring one encoding needs beyond the clip itself.

    ``composite`` is the metric the loss is measured with, ``encode`` the codec config the surrogate
    runs, ``storage`` the target format's cost table that prices the result, and ``rng`` the dither
    source (left unset, the surrogate falls back to its own fixed seed).
    """

    composite: CompositeFidelity
    encode: EncodeConfig
    storage: Storage
    rng: np.random.Generator | None = None


@dataclass(frozen=True)
class OperatingPoint:
    """One encoding's cost/quality: stored bytes vs. composite distortion (lower is better)."""

    params: EncodingParams
    stored_bytes: int
    distortion: float
    frames: int

    @property
    def kib(self) -> float:
        return bytes_to_kib(self.stored_bytes)


def sweep_rates(sweep: SweepConfig, sample_rate: int) -> list[int]:
    """Stored rates a clip recorded at ``sample_rate`` is swept over, highest first.

    The configured ladder states the rates worth stepping down to, and ``sample_rate`` joins them so
    storing the recording as it stands is always among the candidates. A listed rate above the recording
    would resample it upward, spending bytes on a band the recording never held, so the ladder is read as
    far as the recording reaches and no further.
    """
    return sorted({rate for rate in sweep.rates if rate < sample_rate} | {sample_rate}, reverse=True)


def compresses(sweep: SweepConfig, depth: int) -> bool:
    """Whether a sample stored at ``depth`` runs through dynamics on the way to the quantizer.

    Compression trades waveform for headroom against the quantizer, a bargain a depth shallow enough to
    hear its own noise floor stands to win. Deeper storage keeps the waveform as recorded, where the
    quantizer already sits below what compression would protect.
    """
    return sweep.compress and depth <= _COMPRESSIBLE_DEPTH


def _reference(clip: SourceClip, stored: StoredSample) -> Signal:
    """The ground truth ``clip``'s encoding is scored against: its material, closed the way ``stored`` closes.

    The recording is held for the duration the clip is stored to serve, and the ramp a stored span stops
    on is put over it as well, so the score reads what the encoding did to the waveform.
    """
    span = (
        clip.signal
        if clip.duration_s is None
        else np.asarray(clip.signal[: seconds_to_frames(clip.duration_s, clip.sample_rate)], dtype=np.float64)
    )
    return closed_reference(span, stored, clip.sample_rate, pitch=clip.root_pitch)


def evaluate_encoding(clip: SourceClip, params: EncodingParams, context: SweepContext) -> OperatingPoint:
    """Encode ``clip`` with ``params``, render it back at its own pitch, and score the encoding loss."""
    encode_context = EncodeContext(
        root_pitch=clip.root_pitch,
        config=context.encode,
        settled=clip.settled,
        rng=context.rng,
    )
    stored = encode(clip.signal, clip.sample_rate, params, encode_context)
    candidate = render(
        stored,
        clip.sample_rate,
        pitch=clip.root_pitch,
        duration_s=clip.duration_s,
    )
    report = evaluate(_reference(clip, stored), candidate, clip.sample_rate, context.composite)
    return OperatingPoint(
        params=params,
        stored_bytes=context.storage.sample_bytes(frames=stored.frames, depth=stored.depth),
        distortion=report.fidelity,
        frames=stored.frames,
    )
