from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Final

import numpy as np

from optisample.config.codec import EncodeConfig
from optisample.dsp.envelope import decompose, level_reading
from optisample.dsp.levels import gain_to_db, level_readings
from optisample.dsp.piecewise import PiecewiseCurve, fit_piecewise
from optisample.dsp.quantize import headroom_peak, normalize_peak
from optisample.dsp.series import Readings, Series
from optisample.dsp.trajectory import reading_window_s
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from optisample.music import midi_to_freq, sounded_note
from optisample.optimize.export.coverage import covered_routing
from optisample.optimize.export.envelope import (
    QUIETEST_STEP,
    EnvelopeGrid,
    shape_nodes,
    volume_envelope,
)
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.instruments.unit import InstrumentUnit
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.core.timing.clock import tick_seconds
from trackmod.limits.bound import Bound
from trackmod.spec.levels import MAX_VOLUME

STORED_DEPTH: Final = BitDepth.SIXTEEN  # the depth a reference rendering of a recording keeps its timbre at

_WHOLE_RECORDING: Final = None  # the loop a sample holding its material end to end states
_WHOLE_WAVEFORM: Final = 1.0  # the share a waveform keeps where the step beside it states the whole level
_EXACTLY_RESTORED: Final = 0.0  # the gap a pair sounding its recording at the level it was captured at leaves
_FIRST_SAMPLE: Final = 0  # the position the one waveform a written recording holds takes in its own unit


@dataclass(frozen=True)
class NormalizedRecording:
    """One recording read for playing back: the audio itself, beside the level it moves through.

    ``levels`` is that level in decibels, read through the weighting the recording's own pitch forms
    (:func:`~optisample.dsp.envelope.level_reading`) and gathered into the windows a curve is placed
    across, which is what an instrument's volume envelope is fitted to. The recording travels with it
    because what a sample stores is the recording divided by the curve the format ends up playing, so
    the two are settled together.
    """

    name: str
    root_pitch: int
    sample_rate: int
    signal: Signal
    levels: Readings


def normalized_recording(
    signal: Signal,
    sample_rate: int,
    *,
    name: str,
    root_pitch: int,
    encode: EncodeConfig,
) -> NormalizedRecording:
    """``signal`` beside the level a volume envelope has to reproduce for it.

    The level is read through the split every calibrated measurement of a recording is taken through
    (:func:`~optisample.dsp.envelope.decompose`), so the weighting spans two periods of the pitch the
    recording was played at and a deep note is followed as smoothly as a high one. It is then gathered
    into windows the recording's own length settles (:func:`~optisample.dsp.trajectory.reading_window_s`),
    which is the stretch a curve of a few dozen corners is placed across.
    """
    reading = level_reading(sample_rate, encode.envelope, midi_to_freq(root_pitch))
    window_s = reading_window_s(int(signal.size), sample_rate)
    return NormalizedRecording(
        name=name,
        root_pitch=root_pitch,
        sample_rate=sample_rate,
        signal=signal,
        levels=level_readings(decompose(signal, reading).level, sample_rate, window_s=window_s),
    )


def level_curve(recording: NormalizedRecording, target: ExportTarget) -> PiecewiseCurve:
    """The curve ``target`` has room to state ``recording``'s own level as, in decibels on its played clock.

    A format numbers a fixed count of envelope breakpoints and the release spends one of them
    (:func:`~optisample.optimize.export.envelope.shape_nodes`), so the shape is fitted to every corner
    that leaves -- twenty-four for Impulse Tracker, eleven for FastTracker 2.
    """
    return fit_piecewise(recording.levels, nodes=shape_nodes(target.envelope_point_bound))


def played_gain(envelope: Envelope, *, tempo: int, frames: int, sample_rate: int) -> Series:
    """What ``envelope`` multiplies a held voice by at each of ``frames``, on the clock ``tempo`` runs.

    A tracker walks straight between two breakpoints in amplitude and holds the last one it reaches for
    as long as the note is held, which is what a note sounding past the corner the shape closes on plays
    at. Reading the curve this way states the gain the format actually applies, so a waveform divided by
    it comes back out at the level the recording held.

    The breakpoints up to the sustain point are the shape itself; the one past it carries a released note
    to silence, so a note still sounding never reaches it.
    """
    shape = envelope.points[: envelope.length if envelope.sustain is None else envelope.sustain.begin + 1]
    length = tick_seconds(tempo)
    seconds = np.arange(frames, dtype=np.float64) / sample_rate
    moments = np.asarray([point.tick * length for point in shape], dtype=np.float64)
    values = np.asarray([point.value / MAX_VOLUME for point in shape], dtype=np.float64)
    return np.asarray(np.interp(seconds, moments, values), dtype=np.float64)


@dataclass(frozen=True)
class StoredLevel:
    """How one recording's own level is stated: the step a format applies, and what the waveform keeps.

    A voice sounds at ``waveform x envelope x step``, so delivering a recording at the amplitude it was
    captured at asks those three for exactly the level the envelope gave up. ``step`` takes as much of it
    as the format's own per-sample multiplier states and ``share`` is what the waveform stored at the
    headroom peak keeps, which is the rest. ``gap_db`` is how far the pair lands from the level the
    recording was captured at, which is nothing for every recording the headroom leaves room for.
    """

    step: int
    share: float
    gap_db: float

    @property
    def restored(self) -> bool:
        """Whether the written pair sounds the recording at the amplitude it was captured at."""
        return self.gap_db == _EXACTLY_RESTORED


def stored_level(room: float, *, steps: Bound) -> StoredLevel:
    """The level a recording asks for, split between the step ``steps`` numbers and its own waveform.

    ``room`` is how far the levelled waveform may be scaled up before it meets the headroom it is stored
    under, which is the whole of what playback has to give back. The step is rounded up onto the format's
    grid, so what is left for the waveform is always a scaling down and the waveform stays inside its
    headroom however coarse that grid is near the floor.

    A levelled waveform already filling the headroom at the top step is stored as hot as that allows and
    sounds under the level its recording was captured at, which ``gap_db`` states -- a recording peaking
    within a decibel or two of full scale, whose flattened form asks for more than a tracker can put out.
    """
    step = steps.clamp(ceil(MAX_VOLUME / room))
    asked = MAX_VOLUME / (step * room)
    share = min(asked, _WHOLE_WAVEFORM)
    return StoredLevel(step=step, share=share, gap_db=gain_to_db(share / asked))


@dataclass(frozen=True)
class RecordingInstrument:
    """One recording written as a standalone instrument, beside how its own level was stated.

    ``level`` travels with the unit so a caller reports the recordings a format had no room to sound at
    the amplitude they were captured at, rather than letting the shortfall pass unsaid.
    """

    unit: InstrumentUnit
    level: StoredLevel


def _keymap(root_pitch: int, target: ExportTarget) -> Keymap:
    """Every key ``target`` numbers routed to the one stored waveform, each sounding its own pitch.

    A recording reaches five octaves either way of the key it was played at, so widening its own key
    outward (:func:`~optisample.optimize.export.coverage.covered_routing`) gives a written instrument the
    whole stretch of keyboard that recording can be transposed across, and every key on it sounds the
    pitch it is named for.
    """
    root_key = target.key(root_pitch)
    routing = {root_key: KeyAssignment(sample=_FIRST_SAMPLE, note=sounded_note(root_key, root_key))}
    return routed_keymap(covered_routing(routing, target))


def recording_instrument(
    recording: NormalizedRecording,
    *,
    target: ExportTarget,
    grid: EnvelopeGrid,
    headroom_db: float,
) -> RecordingInstrument:
    """``recording`` as the standalone instrument ``target`` writes: one waveform, played down by one curve.

    The level is fitted to the corners the format numbers and written onto its own tick and amplitude
    grids (:func:`~optisample.optimize.export.envelope.volume_envelope`), against its own loudest moment,
    since an instrument written on its own carries the whole of its level and stands beside nothing to be
    balanced with. The waveform is then the recording divided by the gain that written curve applies
    (:func:`played_gain`), so the pair multiplies back to the recording and what the sample holds is the
    timbre: level-flat wherever the format's thirty-six decibels of envelope reach, and carrying its own
    decline below that, which is where a curve stating one more step has none left to state. A moment the
    curve silences outright plays as silence whatever the waveform holds there, so the division is taken
    against the quietest step that still sounds and the waveform stays finite throughout.

    What the recording was captured at is stated beside all that (:func:`stored_level`), so a note played
    on the written instrument sounds at the amplitude the recording holds and two instruments written from
    one dataset stand exactly as far apart as their recordings do.
    """
    curve = level_curve(recording, target)
    envelope = volume_envelope(curve, grid, peak_db=curve.peak)
    gain = played_gain(
        envelope,
        tempo=grid.tempo,
        frames=int(recording.signal.size),
        sample_rate=recording.sample_rate,
    )
    sounding = np.maximum(gain, QUIETEST_STEP / MAX_VOLUME)
    hot, room = normalize_peak(recording.signal / sounding, headroom_peak(headroom_db))
    level = stored_level(room, steps=target.sample_level_bound)
    levels = target.sample_levels(level.step)
    return RecordingInstrument(
        unit=InstrumentUnit(
            instrument=Instrument(
                name=recording.name,
                keymap=_keymap(recording.root_pitch, target),
                volume_envelope=envelope,
            ),
            samples=(
                Sample(
                    name=recording.name,
                    pcm=np.asarray(hot * level.share, dtype=np.float64),
                    rate=recording.sample_rate,
                    depth=STORED_DEPTH,
                    volume=levels.volume,
                    gain=levels.gain,
                    loop=_WHOLE_RECORDING,
                ),
            ),
        ),
        level=level,
    )
