import numpy as np

from optisample.dsp.decay import LinearDecay
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


def _frame_scale(stored: StoredSample, out_rate: int, pitch: int | None) -> float:
    """Output frames per stored frame once ``stored`` is repitched to ``pitch`` and played at ``out_rate``."""
    effective_rate = _effective_rate(stored, pitch)
    return out_rate / effective_rate if effective_rate > 0.0 else 0.0


def output_frame(stored: StoredSample, frame: int, out_rate: int, pitch: int | None) -> int:
    """Where stored frame ``frame`` lands on the output timeline of a note played at ``pitch``.

    Repitching resamples the whole stored span, so the stored timeline stretches by one ratio and every
    frame moves with it. Passing :attr:`~optisample.dsp.surrogate.sample.StoredSample.frames` gives the
    length the playback runs for, which is where a sample carrying no loop falls silent.
    """
    return round(frame * _frame_scale(stored, out_rate, pitch))


def _repitch(stored: StoredSample, out_rate: int, pitch: int | None) -> tuple[Signal, float]:
    """Resample ``stored``'s PCM from its effective (repitched) rate to ``out_rate``.

    Returns the repitched playback together with the stored->output frame scale, which the loop-sustain
    step reuses to map the stored loop bounds into the output domain.
    """
    scale = _frame_scale(stored, out_rate, pitch)
    played = resample_num(stored.pcm, output_frame(stored, stored.frames, out_rate, pitch))
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


def _declining(played: Signal, decay: LinearDecay | None, out_rate: int) -> Signal:
    """``played`` brought down by the ramp its stored sample declines on, where it carries one.

    The ramp holds unit gain over the stored material and falls away past it, so what the loop repeats
    declines the way the recording did while the stored frames sound exactly as they were stored.
    """
    if decay is None:
        return played

    return np.asarray(played * decay.envelope(played.size, out_rate), dtype=np.float64)


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

    A sample carrying a decay (:class:`~optisample.dsp.decay.LinearDecay`) is played down by it, which is
    what lets a held loop fall away the way the recording it stands for did. The note then sounds at
    :attr:`~optisample.dsp.surrogate.sample.StoredSample.playback_gain` times ``volume``: the first
    restores the level the recording was stored hot from, which is what a module reaches through its
    per-sample multiplier, and the second is the note's own dynamic.
    """
    played, scale = _repitch(stored, out_rate, pitch)
    if duration_s is not None and stored.loop is not None:
        target = round(duration_s * out_rate)
        if target > played.size:
            played = _sustain_with_loop(played, stored.loop, scale, target)

    rendered = apply_gain(_declining(played, stored.decay, out_rate), stored.playback_gain * volume / MAX_VOLUME)
    if duration_s is not None:
        rendered = _fit_length(rendered, round(duration_s * out_rate))

    return np.asarray(rendered, dtype=np.float64)


def _ramp_shapes(scored_frames: int, ramp_start: int, played_frames: int) -> bool:
    """Whether the ramp closing a stored span reaches into the stretch a note is scored over.

    It does when the ramp begins before both the note's end and the end of the stored material. A sample
    closing on nothing begins its ramp where its material ends, which leaves every note clear of it.
    """
    return ramp_start < min(scored_frames, played_frames)


def closed_reference(span: Signal, stored: StoredSample, out_rate: int, *, pitch: int | None) -> Signal:
    """``span`` closed by the ramp that closes ``stored`` where it plays at ``pitch``.

    A stored sample played to its end stops on a ramp to silence, which lands inside any note held that
    far. Putting the same ramp over the same output frames of the ground truth leaves the two differing
    by what the codec did to the waveform, which is what a score comparing them is asked for.

    The length the ramp closes stays charged: past the end of the stored material the ground truth is the
    recording, so a note held longer than its sample is measured against the material it is missing. A
    looped sample wraps at its seam, and its ground truth is the recording throughout.
    """
    played_frames = output_frame(stored, stored.frames, out_rate, pitch)
    ramp_start = output_frame(stored, stored.frames - stored.release_frames, out_rate, pitch)
    if not _ramp_shapes(span.size, ramp_start, played_frames):
        return span

    ramp = np.linspace(1.0, 0.0, played_frames - ramp_start, endpoint=True)
    reach = min(span.size, played_frames)
    closed = np.array(span, dtype=np.float64)
    closed[ramp_start:reach] *= ramp[: reach - ramp_start]
    return closed
