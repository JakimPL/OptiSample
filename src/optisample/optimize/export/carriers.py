from __future__ import annotations

from collections.abc import Sequence
from math import inf
from typing import Final

from optisample.dsp.levels import gain_to_db, level_readings
from optisample.dsp.series import Readings
from optisample.dsp.timebase import seconds_to_frames
from optisample.dsp.trajectory import SharedTrajectory, TrajectoryMember, fit_shared_trajectory, reading_window_s
from optisample.music import semitone_ratio
from optisample.optimize.export.voices import NO_SHAPE, PlannedVoices, Voice, slot_voices
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout
from optisample.optimize.plans import SampleUnit
from optisample.optimize.tasks import StoredRecordings
from trackmod.spec.levels import MAX_VOLUME

_UNBOUNDED: Final = inf  # how long a looped waveform keeps sounding, which is every moment asked of it


def planned_reach_s(unit: SampleUnit, recordings: StoredRecordings, pitch: int) -> float:
    """How long the waveform this unit will be stored as keeps sounding a note at ``pitch``.

    A shape has to be fitted before anything is encoded, so the reach is read off what the plan asked for
    rather than off a stored sample: a unit naming a loop the stage settled wraps for as long as the note
    is held, and one storing the played span reaches the end of that span, stretched by the transposition
    it is played at.
    """
    settled = recordings.settled.get(unit.representative_key, ())
    if unit.params.loop_index is not None and settled:
        return _UNBOUNDED

    played_s = unit.params.trim_s
    if played_s is None:
        played_s = recordings.audio[unit.representative_key].size / recordings.sample_rate

    return played_s * semitone_ratio(unit.representative - pitch)


def _asked_of_the_curve(reference: Readings, *, written_db: float) -> Readings:
    """The level a curve has to supply for one key: what its recording holds, less what the pattern writes.

    A tracker multiplies three things onto every note -- the waveform, the step beside its sample, and the
    note volume the pattern states for the velocity it was struck at. The pattern's share is known before
    anything is stored, so it is taken out here and the curve is fitted to what is left; the sample's own
    step is not known yet, and it is exactly what each member's offset absorbs.
    """
    return Readings(values=reference.values - written_db, seconds=reference.seconds)


def _member(
    key_reference: Readings,
    *,
    group: int,
    weight: float,
) -> TrajectoryMember:
    """One key's own level as a trajectory answering to a level of its own.

    Each stored sample stands in a group by itself, so what the fit settles is the decline the keys of one
    instrument agree on while each sample's own loudness stays in its offset -- which is exactly the pair a
    tracker writes, one envelope and one step per sample.
    """
    return TrajectoryMember(readings=key_reference, groups=(group,), weight=weight)


def _slot_members(voices: Sequence[Voice], sources: PlannedVoices, *, window_s: float) -> tuple[TrajectoryMember, ...]:
    """One member per key this instrument answers, each read from that key's own recording.

    Every stored sample stands in a group of its own, so the fit settles the decline the keys agree on
    while each sample's loudness stays in its offset. A key read over less than one whole window states
    nothing a curve could follow and is left out.
    """
    recordings = sources.recordings
    sample_rate = recordings.sample_rate
    groups = {sample: group for group, sample in enumerate(sorted({voice.sample for voice in voices}))}
    members = []
    for voice in voices:
        span = seconds_to_frames(voice.span_s, sample_rate)
        readings = level_readings(recordings.audio[voice.key][:span], sample_rate, window_s=window_s)
        if readings.count == 0:
            continue

        written = gain_to_db(sources.velocity_map.volume(voice.key.velocity) / MAX_VOLUME)
        members.append(
            _member(
                _asked_of_the_curve(readings, written_db=written),
                group=groups[voice.sample],
                weight=voice.weight,
            )
        )

    return tuple(members)


def slot_trajectory(
    slot: InstrumentSlot,
    sources: PlannedVoices,
    *,
    nodes: int,
) -> SharedTrajectory | None:
    """The curve this instrument plays every voice down by, read from its keys' own recordings.

    This is the inversion the carrier needs. The shape a slot carries is fitted from the level its
    recordings actually hold, before any of them is stored, so each waveform can then be that recording
    divided by what the written curve plays it down by. Nothing iterates: what the curve cannot state is
    exactly what stays in the waveform.

    Returns ``NO_SHAPE`` for an instrument the material plays no recorded key of, which leaves its voices
    at the level their own material carries.
    """

    def reach(_sample: int, unit: SampleUnit, pitch: int) -> float:
        return planned_reach_s(unit, sources.recordings, pitch)

    voices = slot_voices(slot, sources, reach=reach)
    if not voices:
        return NO_SHAPE

    sample_rate = sources.recordings.sample_rate
    longest = seconds_to_frames(max(voice.span_s for voice in voices), sample_rate)
    members = _slot_members(voices, sources, window_s=reading_window_s(longest, sample_rate))
    if not members:
        return NO_SHAPE

    return fit_shared_trajectory(members, nodes=nodes)


def plan_trajectories(
    layout: SlotLayout,
    sources: PlannedVoices,
    *,
    nodes: int,
) -> tuple[SharedTrajectory | None, ...]:
    """One shape per written instrument, in the order the module numbers them, fitted before any encoding."""
    return tuple(slot_trajectory(slot, sources, nodes=nodes) for slot in layout.slots)
