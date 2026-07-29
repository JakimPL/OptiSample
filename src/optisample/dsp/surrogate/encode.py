from dataclasses import dataclass

from optisample.config.codec import EncodeConfig, LoopConfig
from optisample.dsp.decay import NO_DECAY, LinearDecay, fit_linear_decay
from optisample.dsp.dynamics import compress
from optisample.dsp.loop import Loop, crossfade_loop, loop_candidates
from optisample.dsp.quantize import headroom_peak, normalize_peak, release_fade, requantize
from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate.params import EncodeContext, EncodingParams
from optisample.dsp.surrogate.sample import NO_RELEASE_RAMP, Signal, StoredSample


def _chosen_loop(
    resampled: Signal,
    params: EncodingParams,
    config: LoopConfig,
) -> Loop | None:
    """The candidate ``params.loop_choice`` names, as the resampled waveform lays them out.

    Candidates are found at the stored rate, so the loop is placed on exactly the audio being stored and
    its bounds are the frames a player wraps between. Material offering fewer candidates than the choice
    counts to answers with ``None``, which stores the trimmed sample.
    """
    if params.loop_choice is None:
        return None

    candidates = loop_candidates(resampled, params.target_rate, config)
    return candidates[params.loop_choice] if params.loop_choice < len(candidates) else None


def _apply_loop(resampled: Signal, rate: int, loop: Loop, config: LoopConfig) -> Signal:
    """Crossfade the loop's seam and trim storage to attack + loop, which is the span a loop keeps."""
    faded = crossfade_loop(resampled, loop, fade_len=round(config.crossfade_s * rate))
    return faded[: loop.end]


def _loop_or_trim(
    resampled: Signal,
    params: EncodingParams,
    config: LoopConfig,
) -> tuple[Signal, Loop | None]:
    """Bound stored length by looping *or* trimming, whichever the params select.

    A chosen loop already trims storage to ``[0, loop.end)`` (attack + one loop region), so it sets the
    full stored length itself. Trimming to ``trim_s`` governs the remaining cases: params asking for the
    trimmed sample, and a loop choice reaching past what the material offers.
    """
    loop = _chosen_loop(resampled, params, config)
    if loop is not None:
        return _apply_loop(resampled, params.target_rate, loop, config), loop

    if params.trim_s is not None:
        return resampled[: max(0, round(params.trim_s * params.target_rate))], None

    return resampled, None


@dataclass(frozen=True)
class _Span:
    """The stretch a sample holds and how it carries on: its loop, its closing ramp, and its decay."""

    signal: Signal
    loop: Loop | None
    release_frames: int
    decay: LinearDecay | None


def _looped(span: Signal, material: Signal, loop: Loop, rate: int) -> _Span:
    """A looped span, kept as it stands, beside the decline the material past it goes on making.

    The wrap point is where :func:`~optisample.dsp.loop.crossfade_loop` has already made the seam
    continuous, so the span reaches its end at the level it holds there. What the loop then repeats is
    played down by the fitted decay, which is how a held note declines while the PCM stays the material.
    """
    return _Span(span, loop, NO_RELEASE_RAMP, fit_linear_decay(material, rate, loop))


def _closed(span: Signal, rate: int, config: EncodeConfig) -> _Span:
    """``span`` with the ramp closing a sample that plays to its end, and how far that ramp reaches.

    A span with no loop is played out and stops, and the ramp is what it stops on -- over its whole
    length where the span is the shorter of the two. Its PCM runs to where the material does, so it
    carries every level it plays at and asks for no decay beside it.
    """
    frames = min(round(config.release_fade_s * rate), span.size)
    return _Span(release_fade(span, frames), None, frames, NO_DECAY)


def _stored_span(
    signal: Signal,
    sample_rate: int,
    params: EncodingParams,
    config: EncodeConfig,
) -> _Span:
    """The stretch of ``signal`` a sample holds: resampled, shaped where asked, then looped or trimmed.

    Compression runs on the resampled waveform, ahead of loop detection, so the seam is crossfaded over
    exactly the audio that will be stored and stays continuous. It is also what the decay is fitted from,
    for the same reason: the level a loop repeats and the levels the material falls to past it are read
    off the waveform actually being stored.
    """
    resampled = resample_to(signal, sample_rate, params.target_rate)
    shaped = compress(resampled, params.target_rate, config.dynamics) if params.compress else resampled
    span, loop = _loop_or_trim(shaped, params, config.loop)
    if loop is not None:
        return _looped(span, shaped, loop, params.target_rate)

    return _closed(span, params.target_rate, config)


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

    With ``params.loop_choice`` naming a candidate the material offers, storage is trimmed to the attack
    plus that looped region and the loop sustains notes held past the stored length -- cheap to store,
    and true to the recording as far as the loop's own timbre holds. The decline the recording goes on
    making past that point is carried beside the PCM as a :class:`~optisample.dsp.decay.LinearDecay`, so
    a held note falls away rather than ringing at the loop's level. Storing the trimmed sample keeps
    ``trim_s`` worth of the recording as it was played and ends a longer note there, closing on the
    release ramp (:func:`~optisample.dsp.quantize.release_fade`) so the sample plays out. Which of the
    two a note is better served by is the sweep's to price, and it enumerates both.
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
        decay=span.decay,
    )
