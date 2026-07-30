from dataclasses import dataclass

from optisample.config.codec import EncodeConfig
from optisample.dsp.decay import NO_DECAY, LinearDecay
from optisample.dsp.dynamics import compress
from optisample.dsp.envelope import level_reading
from optisample.dsp.loop import Loop, loop_at_rate, prepare_loop
from optisample.dsp.quantize import headroom_peak, normalize_peak, release_fade, requantize
from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate.params import NO_LOOP, EncodeContext, EncodingParams
from optisample.dsp.surrogate.sample import NO_RELEASE_RAMP, Signal, StoredSample
from optisample.music import midi_to_freq


@dataclass(frozen=True)
class _Span:
    """The stretch a sample holds and how it carries on: its loop, its closing ramp, and its decay."""

    signal: Signal
    loop: Loop | None
    release_frames: int
    decay: LinearDecay | None


def _apply_loop(shaped: Signal, rate: int, loop: Loop, context: EncodeContext) -> Signal:
    """Ready the loop region to wrap and trim storage to attack + loop, which is the span a loop keeps.

    Preparation runs on the resampled waveform, so the region is held at the level the stored copy carries
    and the seam is blended over exactly the frames a player wraps between. The level is read over two
    periods of the pitch the clip was recorded at and at the rate the copy is stored at, which is the same
    reading the loop stage levelled by and so the same waveform it measured.
    """
    config = context.config
    reading = level_reading(rate, config.envelope, midi_to_freq(context.root_pitch))
    return prepare_loop(shaped, loop, rate, config.seam, reading)[: loop.end]


def _looped_span(
    shaped: Signal,
    sample_rate: int,
    params: EncodingParams,
    context: EncodeContext,
) -> _Span | None:
    """The looped span these params ask for, over the loop the clip was settled around.

    The loop stage settled the region on the recording at the rate it was analysed at, so the bounds are
    scaled onto the copy being stored (:func:`~optisample.dsp.loop.loop_at_rate`) and name the frames a
    player wraps between. The decay settled with the loop rides along, which is what plays a held note down
    from the level the loop repeats.

    Answers ``None`` where the trimmed sample is what is wanted instead: params asking for it, a clip the
    loop stage settled no loop for, and a stored rate too low for the settled bounds to survive onto.
    """
    settled = context.settled
    if not params.looped or settled is None:
        return None

    loop = loop_at_rate(settled.loop, sample_rate, params.target_rate, frames=shaped.size)
    if loop is None:
        return None

    span = _apply_loop(shaped, params.target_rate, loop, context)
    return _Span(span, loop, NO_RELEASE_RAMP, settled.decay)


def _closed(span: Signal, rate: int, config: EncodeConfig) -> _Span:
    """``span`` with the ramp closing a sample that plays to its end, and how far that ramp reaches.

    A span with no loop is played out and stops, and the ramp is what it stops on -- over its whole
    length where the span is the shorter of the two. Its PCM runs to where the material does, so it
    carries every level it plays at and asks for no decay beside it.
    """
    frames = min(round(config.release_fade_s * rate), span.size)
    return _Span(release_fade(span, frames), NO_LOOP, frames, NO_DECAY)


def _stored_span(
    signal: Signal,
    sample_rate: int,
    params: EncodingParams,
    context: EncodeContext,
) -> _Span:
    """The stretch of ``signal`` a sample holds: resampled, shaped where asked, then looped or trimmed.

    Compression runs on the resampled waveform, ahead of the loop's preparation, so the region is levelled
    and its wrap blended over exactly the audio that will be stored. A looped span carries the decay settled
    with its loop, which is how a held note declines while the PCM stays the material; a trimmed one keeps
    ``trim_s`` of the recording as it was played.
    """
    config = context.config
    resampled = resample_to(signal, sample_rate, params.target_rate)
    shaped = compress(resampled, params.target_rate, config.dynamics) if params.compress else resampled
    looped = _looped_span(shaped, sample_rate, params, context)
    if looped is not None:
        return looped

    if params.trim_s is not None:
        return _closed(shaped[: max(0, round(params.trim_s * params.target_rate))], params.target_rate, config)

    return _closed(shaped, params.target_rate, config)


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

    With ``params.looped`` set, storage is trimmed to the attack plus the loop region the loop stage settled
    for this clip, and that loop sustains notes held past the stored length -- cheap to store, and true to
    the recording as far as the loop's own timbre holds. The region is stored at one level and the decline
    the recording makes from it rides beside the PCM as a :class:`~optisample.dsp.decay.LinearDecay`, so a
    held note falls away on the ramp the material states. Storing the trimmed sample keeps ``trim_s`` worth of the
    recording as it was played and ends a longer note there, closing on the release ramp
    (:func:`~optisample.dsp.quantize.release_fade`) so the sample plays out. Which of the two a note is
    better served by is the sweep's to price, and it enumerates both.
    """
    span = _stored_span(signal, sample_rate, params, context)
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
