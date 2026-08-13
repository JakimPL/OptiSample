from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import inf
from typing import Final

import numpy as np

from optisample.dsp.level import gain_to_db, level_readings
from optisample.dsp.series import Readings, Series
from optisample.dsp.surrogate import StoredSample
from optisample.dsp.timebase import seconds_to_frames
from optisample.dsp.trajectory import SharedTrajectory, TrajectoryMember, fit_shared_trajectory, reading_window_s
from optisample.keys import SampleKey, keys_by_pitch, nearest_key
from optisample.model import NoteEvent
from optisample.music import semitone_ratio
from optisample.optimize.layers.bands import VelocityBand
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout
from optisample.optimize.plans import SampleUnit
from optisample.optimize.tasks import StoredRecordings
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.spec.levels import MAX_VOLUME

NO_SHAPE: Final = None  # what an instrument the material never plays a recorded key of leaves behind
NO_DRIFT: Final = 0.0  # what one envelope costs an instrument carrying none

Reach = Callable[[int, SampleUnit, int], float]  # how long the waveform at a sample position sounds a pitch

_WRITTEN_LEVELS: Final = (0,)  # the one group every voice belongs to while the levels beside it are already written
_SILENT_VOLUME: Final = 0  # the note volume a velocity the pattern silences is written at


@dataclass(frozen=True)
class PlannedVoices:
    """What a slot's keys are read from before any of its samples exists.

    ``recordings`` supplies the ground truth each key is judged against, ``material`` how long the song
    holds that key and how often, and ``velocity_map`` the note volume the pattern will write for it --
    which together name the voices an instrument answers without needing a stored waveform to exist yet.
    """

    recordings: StoredRecordings
    material: Sequence[NoteEvent]
    velocity_map: VelocityVolumeMap


@dataclass(frozen=True)
class PlayedVoices:
    """Everything an instrument's own trajectory is read from beside the samples it starts.

    ``recordings`` supplies the ground truth each key is judged against, ``material`` how long the song
    holds that key and how often, and ``velocity_map`` with ``gains`` the two levels the module already
    writes -- the note volume the pattern states and the 0-64 multiplier the sample carries -- which
    together settle what the stored waveform actually delivers.
    """

    recordings: StoredRecordings
    material: Sequence[NoteEvent]
    velocity_map: VelocityVolumeMap
    gains: tuple[int, ...]

    @property
    def planned(self) -> PlannedVoices:
        """The same voices read without the levels already written, which is what a shape is fitted from."""
        return PlannedVoices(
            recordings=self.recordings,
            material=self.material,
            velocity_map=self.velocity_map,
        )


@dataclass(frozen=True)
class Voice:
    """One key an instrument answers: the recording it stands for, how long it rings, and what it is worth.

    ``key`` names the recording in the audio map this voice is judged against, and its own velocity is the
    dynamic the voice sounds at, so the level the pattern writes and the level the reference holds state
    the same note. ``sample`` is the position the waveform serving it takes in the song's sample table,
    which is where the 0-64 gain written beside it is read from.
    """

    pitch: int
    key: SampleKey
    sample: int
    span_s: float
    weight: float


@dataclass(frozen=True)
class _Served:
    """The waveform one voice plays through: its own level trajectory, and the level written beside it.

    ``levels`` reads the stored PCM on the recording's own clock, so a voice sounded away from the root
    reads the moments it comes from. ``written_db`` is what the module's two multipliers add -- the
    sample's 0-64 gain and the note volume the pattern states -- which is what the waveform arrives at.
    """

    stored: StoredSample
    levels: Readings
    written_db: float


def _played_keys(material: Sequence[NoteEvent], band: VelocityBand) -> dict[int, list[NoteEvent]]:
    """The notes the material plays through one velocity band, gathered by the key each of them sounds."""
    played: dict[int, list[NoteEvent]] = {}
    for note in material:
        if band.covers(note.velocity):
            played.setdefault(note.pitch, []).append(note)

    return played


def _reach_s(stored: StoredSample, pitch: int) -> float:
    """How long the stored waveform keeps sounding a note at ``pitch``.

    A loop repeats for as long as a note is held, so a looped sample reaches every moment the material
    asks of it. A sample stored whole reaches the end of what it holds, stretched by the transposition it
    is played at, and the note stops there -- which is the whole stretch its envelope has to state.
    """
    if stored.loop is not None:
        return inf

    return stored.duration_s * semitone_ratio(stored.root_pitch - pitch)


def _voice(
    pitch: int,
    sample: int,
    notes: Sequence[NoteEvent],
    available: Sequence[SampleKey],
    reach_s: float,
) -> Voice:
    """What one key asks of its instrument: the reference for the loudest dynamic the material plays there.

    The reference is chosen the way the reduction chose the one this key was scored against
    (:func:`~optisample.keys.nearest_key`), so the trajectory an envelope is fitted to is the trajectory
    the objective already measured. It rings for the longest note the material holds here, up to where the
    waveform serving it stops sounding, and is worth the whole playing time the band spends on the key.
    """
    return Voice(
        pitch=pitch,
        key=nearest_key(available, max(note.velocity for note in notes)),
        sample=sample,
        span_s=min(max(note.duration_s for note in notes), reach_s),
        weight=sum(note.weight for note in notes),
    )


def _delivered_db(served: _Served, seconds: Series, *, pitch: int) -> Series:
    """The level the written module puts out for a note at ``pitch``, moment by moment on the played clock.

    A stored sample sounded away from its own root plays proportionally faster or slower, so each played
    moment reads the recorded moment it comes from. Past the loop the waveform holds the level it opens
    the loop on, for as long as the note runs -- which is exactly the stretch the envelope is there to
    supply.
    """
    stored = served.stored
    recorded = seconds / semitone_ratio(stored.root_pitch - pitch)
    reach = stored.duration_s if stored.loop is None else stored.loop.start / stored.sample_rate
    held = np.interp(np.minimum(recorded, reach), served.levels.seconds, served.levels.values)
    return np.asarray(held + served.written_db, dtype=np.float64)


def _member(
    voice: Voice,
    served: _Served,
    recordings: StoredRecordings,
    *,
    window_s: float,
) -> TrajectoryMember | None:
    """What one key's envelope supplies: the level its recording holds, less the level the module delivers.

    Returns ``None`` for a key the material holds too briefly to read one whole window of, which states
    nothing a shape could follow.
    """
    sample_rate = recordings.sample_rate
    reference = recordings.audio[voice.key]
    true = level_readings(reference[: seconds_to_frames(voice.span_s, sample_rate)], sample_rate, window_s=window_s)
    if true.count == 0:
        return None

    asked = true.values - _delivered_db(served, true.seconds, pitch=voice.pitch)
    return TrajectoryMember(
        readings=Readings(values=asked, seconds=true.seconds),
        groups=_WRITTEN_LEVELS,
        weight=voice.weight,
    )


def slot_voices(slot: InstrumentSlot, sources: PlannedVoices, *, reach: Reach) -> list[Voice]:
    """Every key one written instrument answers that the material plays and the grid holds a recording for.

    How long each waveform keeps sounding is asked of ``reach`` rather than read off a stored sample, so
    the same list serves a shape fitted after the samples exist and one fitted before they do.

    A key the pattern writes at silence is left out, since the note it stands for is never heard and the
    trajectory it would state runs away from every level there is.
    """
    played = _played_keys(sources.material, slot.band)
    available = keys_by_pitch(sources.recordings.audio)
    voices = [
        _voice(pitch, sample, played[pitch], available[pitch], reach(sample, unit, pitch))
        for sample, unit in zip(slot.samples, slot.units)
        for pitch in unit.keys
        if pitch in played and pitch in available
    ]
    return [voice for voice in voices if sources.velocity_map.volume(voice.key.velocity) > _SILENT_VOLUME]


def _served(voice: Voice, stored: Sequence[StoredSample], levels: Readings, sources: PlayedVoices) -> _Served:
    """The waveform one voice plays through, at the two levels the module already writes for it."""
    volume = sources.velocity_map.volume(voice.key.velocity)
    return _Served(
        stored=stored[voice.sample],
        levels=levels,
        written_db=gain_to_db(sources.gains[voice.sample] / MAX_VOLUME) + gain_to_db(volume / MAX_VOLUME),
    )


def slot_members(
    slot: InstrumentSlot,
    stored: Sequence[StoredSample],
    sources: PlayedVoices,
) -> tuple[TrajectoryMember, ...]:
    """One trajectory per key this instrument answers, all read at the window its longest note settles.

    Every voice is read from its own onset in windows of one length (:func:`reading_window_s`), so the set
    states the opening stretch of a single clock -- the clock the one envelope an instrument carries is
    played on.
    """
    voices = slot_voices(slot, sources.planned, reach=lambda sample, _unit, pitch: _reach_s(stored[sample], pitch))
    if not voices:
        return ()

    sample_rate = sources.recordings.sample_rate
    longest = seconds_to_frames(max(voice.span_s for voice in voices), sample_rate)
    window_s = reading_window_s(longest, sample_rate)
    levels = {
        voice.sample: level_readings(stored[voice.sample].pcm, stored[voice.sample].sample_rate, window_s=window_s)
        for voice in voices
    }
    members = (
        _member(voice, _served(voice, stored, levels[voice.sample], sources), sources.recordings, window_s=window_s)
        for voice in voices
    )
    return tuple(member for member in members if member is not None)


def instrument_shape(
    slot: InstrumentSlot,
    stored: Sequence[StoredSample],
    sources: PlayedVoices,
    *,
    nodes: int,
) -> SharedTrajectory | None:
    """The one shape this written instrument plays every voice it starts down by.

    A tracker keeps one volume envelope per instrument, so the keys a slot answers are fitted together and
    what the shape leaves each of them is :attr:`~optisample.dsp.trajectory.SharedTrajectory.dispersion_db`
    (:func:`~optisample.dsp.trajectory.fit_shared_trajectory`). The two levels the module writes beside the
    envelope -- the sample's own gain and the pattern's note volume -- are already spent restoring the
    balance the recordings hold, so they stand as written and the shape carries what is left.

    Returns ``NO_SHAPE`` for an instrument the material plays no recorded key of, which leaves its voices
    at the level their own material carries.
    """
    members = slot_members(slot, stored, sources)
    if not members:
        return NO_SHAPE

    return fit_shared_trajectory(members, nodes=nodes)


def instrument_shapes(
    layout: SlotLayout,
    stored: Sequence[StoredSample],
    sources: PlayedVoices,
    *,
    nodes: int,
) -> tuple[SharedTrajectory | None, ...]:
    """One shape per written instrument, in the order the module numbers them."""
    return tuple(instrument_shape(slot, stored, sources, nodes=nodes) for slot in layout.slots)


@dataclass(frozen=True)
class WrittenInstruments:
    """The instruments a plan is written as, beside what each one's single volume envelope leaves its keys.

    The module and every artifact describing it answer for the same instruments, so the layout and the
    reading each slot earned travel together and a reader states both from one value.
    """

    layout: SlotLayout
    drifts: tuple[float, ...]


def written_instruments(
    layout: SlotLayout,
    stored: Sequence[StoredSample],
    sources: PlayedVoices,
    *,
    nodes: int,
) -> WrittenInstruments:
    """``layout`` beside the dispersion the one envelope each of its instruments carries leaves its keys.

    An instrument the material plays no recorded key of carries no envelope, so it leaves ``NO_DRIFT``.
    """
    shapes = instrument_shapes(layout, stored, sources, nodes=nodes)
    return WrittenInstruments(
        layout=layout,
        drifts=tuple(NO_DRIFT if shape is NO_SHAPE else shape.dispersion_db for shape in shapes),
    )
