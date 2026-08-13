from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, prod
from typing import Final

import numpy as np

from optisample.config.codec import EncodeConfig
from optisample.dsp.envelope import decompose, level_reading
from optisample.dsp.level import Clock, curve_level, gain_to_db, level_readings, written_level
from optisample.dsp.piecewise import PiecewiseCurve, fit_piecewise
from optisample.dsp.quantize import headroom_peak, normalize_peak
from optisample.dsp.series import Readings
from optisample.dsp.trajectory import reading_window_s
from optisample.io.tracker.envelope import (
    EnvelopeGrid,
    shape_nodes,
    sounding_gain,
    volume_envelope,
)
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from optisample.music import midi_to_freq, sounded_note
from optisample.optimize.export.coverage import covered_routing
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.instruments.unit import InstrumentUnit
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.limits.bound import Bound

STORED_DEPTH: Final = BitDepth.SIXTEEN  # the depth a reference rendering of a recording keeps its timbre at

_WHOLE_RECORDING: Final = None  # the loop a sample holding its material end to end states
_WHOLE_WAVEFORM: Final = 1.0  # the share a waveform keeps where the step beside it states the whole level
_EXACTLY_RESTORED: Final = 0.0  # the gap a pair sounding its recording at the level it was captured at leaves
_FIRST_SAMPLE: Final = 0  # the position the one waveform a written recording holds takes in its own unit
_SOUNDING_STEP: Final = 1  # the lowest step a level field states that still sounds the note it multiplies


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
    (:func:`~optisample.io.tracker.envelope.shape_nodes`), so the shape is fitted to every corner
    that leaves -- twenty-four for Impulse Tracker, eleven for FastTracker 2.
    """
    return fit_piecewise(recording.levels, nodes=shape_nodes(target.envelope_point_bound))


@dataclass(frozen=True)
class StoredLevel:
    """How one recording's own level is stated: the steps a format applies, and what the waveform keeps.

    A voice sounds at ``waveform x envelope x steps``, so delivering a recording at the amplitude it was
    captured at asks those for exactly the level the envelope gave up. ``steps`` takes as much of it as
    the format's own always-applied fields state and ``share`` is what the waveform stored at the headroom
    peak keeps, which is the rest. ``gap_db`` is how far the pair lands from the level the recording was
    captured at, which is nothing for every recording the headroom leaves room for.
    """

    steps: tuple[int, ...]
    share: float
    gap_db: float

    @property
    def restored(self) -> bool:
        """Whether the written pair sounds the recording at the amplitude it was captured at."""
        return self.gap_db == _EXACTLY_RESTORED


def _stated_level(wanted: float, steps: Sequence[Bound]) -> tuple[int, ...]:
    """The step of each field whose levels together state the lowest one at or above ``wanted``.

    Each field states its step as a share of the whole it reaches, so what the fields state together is
    the product of those shares -- a lattice far finer than either grid, which is what lets the waveform
    under it stay at the peak it is stored to. Walking every step of the trailing field and closing each
    with the lowest leading step that reaches ``wanted`` reads every product there is, since the leading
    step is settled once the trailing one is.

    A level past what the fields reach answers with every field at full, which is as close as the format
    comes to it.
    """
    leading, trailing = steps
    reached = tuple(bound.maximum for bound in steps)
    lowest = _WHOLE_WAVEFORM
    for step in range(max(_SOUNDING_STEP, trailing.minimum), trailing.maximum + 1):
        share = step / trailing.maximum
        opening = leading.clamp(ceil(wanted * leading.maximum / share))
        level = share * opening / leading.maximum
        if wanted <= level < lowest:
            reached, lowest = (opening, step), level

    return reached


def stored_level(room: float, *, steps: Sequence[Bound]) -> StoredLevel:
    """The level a recording asks for, split between the fields ``steps`` numbers and its own waveform.

    ``room`` is how far the levelled waveform may be scaled up before it meets the headroom it is stored
    under, which is the whole of what playback has to give back. The fields take the lowest level they
    state at or above that, so what is left for the waveform is always a scaling down and the waveform
    stays inside its headroom however coarse a single grid runs near the floor.

    How much of the headroom the waveform keeps follows from how finely the fields divide the level asked
    of them. A format multiplying two grids meets any level a recording down to about -35 dBFS asks for
    to within 0.12 dB and one at -49 dBFS to within 0.30 dB, so its waveform is stored at the peak; a
    format stating one grid meets the same levels to within a few decibels, which is what its waveform
    gives up.

    A levelled waveform already filling the headroom at the top of every field is stored as hot as that
    allows and sounds under the level its recording was captured at, which ``gap_db`` states -- a
    recording peaking within a decibel or two of full scale, whose flattened form asks for more than a
    tracker can put out.
    """
    wanted = _WHOLE_WAVEFORM / room
    reached = _stated_level(wanted, steps)
    level = prod(step / bound.maximum for step, bound in zip(reached, steps))
    asked = wanted / level
    share = min(asked, _WHOLE_WAVEFORM)
    return StoredLevel(steps=reached, share=share, gap_db=gain_to_db(share / asked))


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
    grids (:func:`~optisample.io.tracker.envelope.volume_envelope`), against its own loudest moment,
    since an instrument written on its own carries the whole of its level and stands beside nothing to be
    balanced with. The waveform is then the recording divided by the gain that written curve applies
    (:func:`played_gain`), so the pair multiplies back to the recording and what the sample holds is the
    timbre: level-flat wherever the format's thirty-six decibels of envelope reach, and carrying its own
    decline below that, which is where a curve stating one more step has none left to state. A moment the
    curve silences outright plays as silence whatever the waveform holds there, so the division is taken
    against the quietest step that still sounds and the waveform stays finite throughout.

    What the recording was captured at is stated beside all that (:func:`stored_level`), across every
    level the format applies to a note whatever a pattern says, so a note played on the written instrument
    sounds at the amplitude the recording holds and two instruments written from one dataset stand exactly
    as far apart as their recordings do.
    """
    shape = curve_level(level_curve(recording, target), Clock.PLAYED)
    envelope = volume_envelope(written_level(shape, reference_db=shape.peak_db), grid)
    sounding = sounding_gain(
        envelope,
        tempo=grid.tempo,
        frames=int(recording.signal.size),
        sample_rate=recording.sample_rate,
    )
    hot, room = normalize_peak(recording.signal / sounding, headroom_peak(headroom_db))
    level = stored_level(room, steps=target.sample_level_bounds)
    levels = target.sample_levels(level.steps)
    return RecordingInstrument(
        unit=InstrumentUnit(
            instrument=Instrument(
                name=recording.name,
                keymap=_keymap(recording.root_pitch, target),
                volume_envelope=envelope,
                global_volume=levels.instrument,
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
