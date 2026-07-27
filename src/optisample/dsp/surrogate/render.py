import numpy as np

from optisample.dsp.loop import Loop
from optisample.dsp.quantize import apply_gain
from optisample.dsp.resample import resample_num
from optisample.dsp.surrogate.sample import Signal, StoredSample
from optisample.music import semitone_ratio
from trackmod.spec.levels import MAX_VOLUME


def _fit_length(signal: Signal, length: int) -> Signal:
    """Truncate or zero-pad ``signal`` to exactly ``length`` samples."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)

    if signal.size == length:
        return signal

    if signal.size > length:
        return np.asarray(signal[:length], dtype=np.float64)

    return np.asarray(
        np.pad(signal, (0, length - signal.size)),
        dtype=np.float64,
    )


def _effective_rate(stored: StoredSample, pitch: int | None) -> float:
    """Rate the sample effectively plays at after repitching to ``pitch``.

    A tracker repitches by resampling: triggering key ``pitch`` transposes ``pitch - root_pitch``
    semitones, so the stored rate scaled by ``2**(semitones / 12)`` is the effective playback rate --
    the ratio that maps stored frames to output frames. ``pitch = None`` means no transpose.
    """
    transpose = 0.0 if pitch is None else float(pitch - stored.root_pitch)
    return stored.sample_rate * semitone_ratio(transpose)


def _repitch(stored: StoredSample, out_rate: int, pitch: int | None) -> tuple[Signal, float]:
    """Resample ``stored``'s PCM from its effective (repitched) rate to ``out_rate``.

    Returns the repitched playback together with the stored->output frame scale, which the loop-sustain
    step reuses to map the stored loop bounds into the output domain.
    """
    effective_rate = _effective_rate(stored, pitch)
    scale = out_rate / effective_rate if effective_rate > 0.0 else 0.0
    played = resample_num(stored.pcm, round(stored.frames * scale))
    return played, scale


def _sustain_with_loop(
    played: Signal,
    loop: Loop,
    scale: float,
    target: int,
) -> Signal:
    """Extend ``played`` to ``target`` frames by repeating its loop region (mapped to the output rate)."""
    start = max(0, min(round(loop.start * scale), played.size))
    end = max(start + 1, min(round(loop.end * scale), played.size))
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
    """Render a note from ``stored`` at ``out_rate``: repitch to ``pitch``, level it, fit the duration.

    ``pitch`` defaults to the sample's root (no transpose). Repitching plays the sample faster/slower
    (``2**((pitch - root) / 12)``), which shifts both pitch and length the way a tracker does. If the
    sample carries a loop and the note is held past the stored length, the loop region is repeated to
    sustain it (in the output domain, so it tracks the repitch); otherwise the note simply ends.

    The note sounds at :attr:`~optisample.dsp.surrogate.sample.StoredSample.playback_gain` times
    ``volume``: the first restores the level the recording was stored hot from, which is what a module
    reaches through its per-sample multiplier, and the second is the note's own dynamic.
    """
    played, scale = _repitch(stored, out_rate, pitch)
    if duration_s is not None and stored.loop is not None:
        target = round(duration_s * out_rate)
        if target > played.size:
            played = _sustain_with_loop(played, stored.loop, scale, target)

    rendered = apply_gain(played, stored.playback_gain * volume / MAX_VOLUME)
    if duration_s is not None:
        rendered = _fit_length(rendered, round(duration_s * out_rate))

    return np.asarray(rendered, dtype=np.float64)
