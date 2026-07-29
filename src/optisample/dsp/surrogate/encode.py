from dataclasses import dataclass

from optisample.config.codec import EncodeConfig, LoopConfig
from optisample.dsp.dynamics import compress
from optisample.dsp.loop import Loop, crossfade_loop, detect_loop
from optisample.dsp.quantize import headroom_peak, normalize_peak, release_fade, requantize
from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate.params import EncodeContext, EncodingParams
from optisample.dsp.surrogate.sample import NO_RELEASE_RAMP, Signal, StoredSample


def _apply_loop(
    resampled: Signal,
    rate: int,
    config: LoopConfig,
) -> tuple[Signal, Loop | None]:
    """Detect a loop, crossfade its seam, and trim storage to attack + loop (or leave the signal be)."""
    detected = detect_loop(resampled, rate, config)
    if detected is None:
        return resampled, None

    fade_len = round(config.crossfade_s * rate)
    faded = crossfade_loop(resampled, detected, fade_len=fade_len)
    return faded[: detected.end], detected


def _loop_or_trim(
    resampled: Signal,
    params: EncodingParams,
    config: LoopConfig,
) -> tuple[Signal, Loop | None]:
    """Bound stored length by looping *or* trimming, whichever the config selects.

    A detected loop already trims storage to ``[0, loop.end)`` (attack + one loop region), so it sets
    the full stored length itself. Trimming to ``trim_s`` governs the remaining cases: a config that
    requests trimming, and a loop request that falls back to trimming when the material is too
    aperiodic to loop.
    """
    if params.loop:
        looped, loop = _apply_loop(resampled, params.target_rate, config)
        if loop is not None:
            return looped, loop

    if params.trim_s is not None:
        return resampled[: max(0, round(params.trim_s * params.target_rate))], None

    return resampled, None


@dataclass(frozen=True)
class _Span:
    """The stretch a sample holds and how it ends: its loop, and the frames its closing ramp covers."""

    signal: Signal
    loop: Loop | None
    release_frames: int


def _closed(span: Signal, loop: Loop | None, rate: int, config: EncodeConfig) -> _Span:
    """``span`` with the ramp closing a sample that plays to its end, and how far that ramp reaches.

    A looped span ends at its wrap point, where :func:`~optisample.dsp.loop.crossfade_loop` has already
    made the seam continuous, so it is stored as it stands. Every other span is played out and stops, and
    the ramp is what it stops on -- over its whole length where the span is the shorter of the two.
    """
    if loop is not None:
        return _Span(span, loop, NO_RELEASE_RAMP)

    frames = min(round(config.release_fade_s * rate), span.size)
    return _Span(release_fade(span, frames), None, frames)


def _stored_span(
    signal: Signal,
    sample_rate: int,
    params: EncodingParams,
    config: EncodeConfig,
) -> _Span:
    """The stretch of ``signal`` a sample holds: resampled, shaped where asked, then looped or trimmed.

    Compression runs on the resampled waveform, ahead of loop detection, so the seam is crossfaded over
    exactly the audio that will be stored and stays continuous. The release ramp closes what the trim
    leaves, measured over the span the sample keeps.
    """
    resampled = resample_to(signal, sample_rate, params.target_rate)
    shaped = compress(resampled, params.target_rate, config.dynamics) if params.compress else resampled
    span, loop = _loop_or_trim(shaped, params, config.loop)
    return _closed(span, loop, params.target_rate, config)


def encode(
    signal: Signal,
    sample_rate: int,
    params: EncodingParams,
    context: EncodeContext,
) -> StoredSample:
    """Encode ``signal`` into a :class:`StoredSample`: resample -> compress -> (loop | trim) -> fade -> normalize.

    The chain ends on the quantizer, which the steps before it exist to feed. Normalizing last measures
    the span actually stored, so a peak in a stretch the trim discards leaves the stored sample at the
    level it asked for; ``params.compress`` narrows the crest factor first, putting more of the grid's
    range under the material. :attr:`StoredSample.gain` records what the normalization applied, which
    playback undoes.

    With ``params.loop`` set, storage is trimmed to the attack plus a looped sustain region (when the
    signal is periodic enough); the loop then sustains notes held past the stored length. For a plain
    config, storage is trimmed to ``trim_s`` and a longer note ends there, closing on the release ramp
    (:func:`~optisample.dsp.quantize.release_fade`) so the sample plays out. A loop request on aperiodic
    material falls back to the trimmed sample, so a looped config always matches or beats its plain twin.
    """
    span = _stored_span(signal, sample_rate, params, context.config)
    normalized, gain = normalize_peak(
        span.signal,
        headroom_peak(context.config.headroom_db),
        reference_peak=context.config.peak_reference,
    )
    pcm = requantize(
        normalized,
        params.depth_bits,
        dither=params.dither,
        noise_shaping=params.noise_shaping,
        rng=context.rng,
    )
    return StoredSample(
        pcm=pcm,
        sample_rate=params.target_rate,
        depth_bits=params.depth_bits,
        root_pitch=context.root_pitch,
        gain=gain,
        loop=span.loop,
        release_frames=span.release_frames,
    )
