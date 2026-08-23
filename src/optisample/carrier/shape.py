from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.carrier.source import CarrierSource
from optisample.dsp.level import Clock, curve_level, level_readings, written_level
from optisample.dsp.trajectory import SharedTrajectory, TrajectoryMember, fit_shared_trajectory, reading_window_s
from optisample.io.tracker.envelope import NO_ENVELOPE, EnvelopeGrid, shape_nodes, volume_envelope
from optisample.io.tracker.target import ExportTarget
from trackmod.core.envelopes.envelope import Envelope

NO_SHAPE: Final = None  # what a set holding nothing long enough to read one whole window of leaves behind
NO_DISPERSION: Final = 0.0  # what one envelope costs a set carrying none


def shared_rate(sources: Sequence[CarrierSource]) -> int:
    """The rate every source was read at, which is the clock their one shape is fitted on.

    Reading a set at two rates would place one moment at two positions, so the shape a single envelope
    states is fitted from recordings a stage already brought to one rate.

    Raises:
        ValueError: when the set holds nothing, or when its sources were read at rates that differ.
    """
    if not sources:
        raise ValueError("a carrier set holds at least one source")

    rates = {source.sample_rate for source in sources}
    if len(rates) > 1:
        raise ValueError(f"sources are read at one rate, against the {sorted(rates)} stated")

    return rates.pop()


def _members(sources: Sequence[CarrierSource], *, window_s: float, sample_rate: int) -> tuple[TrajectoryMember, ...]:
    """Each source's own level as a trajectory answering to a level of its own.

    Every source stands in a group by itself, so what the fit settles is the shape they agree on while each
    source's own loudness is carried by its offset -- which leaves the curve stating the decline alone. A
    source too short to read one whole window of states nothing a shape could follow and is left out.
    """
    read = (
        (index, level_readings(source.decomposition.level, sample_rate, window_s=window_s), source.weight)
        for index, source in enumerate(sources)
    )
    return tuple(
        TrajectoryMember(readings=readings, groups=(index,), weight=weight)
        for index, readings, weight in read
        if readings.count > 0
    )


def carrier_shape(sources: Sequence[CarrierSource], *, nodes: int) -> SharedTrajectory | None:
    """The one curve an instrument built from ``sources`` plays every voice it starts down by.

    A tracker keeps one volume envelope per instrument and one level step per sample, so a set of
    recordings is written as a single shape plus a ladder of offsets
    (:func:`~optisample.dsp.trajectory.fit_shared_trajectory`). Fitting it from the sources' own levels --
    rather than from what an encoder already stored -- is what lets each waveform be its recording divided
    by the curve, which is the whole of the carrier idea. What the shape leaves each source is
    :attr:`~optisample.dsp.trajectory.SharedTrajectory.dispersion_db`, the reading that says whether a set
    was well chosen to share one envelope.

    Returns ``NO_SHAPE`` where no source rings long enough to read one whole window of, which leaves the set
    at the level its own material carries.

    Raises:
        ValueError: when the sources were read at rates that differ, or when one is worth nothing to the
            instrument sharing the shape.
    """
    sample_rate = shared_rate(sources)
    window_s = reading_window_s(max(source.frames for source in sources), sample_rate)
    members = _members(sources, window_s=window_s, sample_rate=sample_rate)
    if not members:
        return NO_SHAPE

    return fit_shared_trajectory(members, nodes=nodes)


@dataclass(frozen=True)
class WrittenShape:
    """The one curve a set of recordings is written under, beside what sharing it costs them.

    ``dispersion_db`` is how far the furthest source stands from the shape, which is the reading that says
    whether the set was well chosen to be written together -- large where the recordings decline at rates
    of their own, near nothing where they decline alike.
    """

    envelope: Envelope | None
    dispersion_db: float


def written_shape(sources: Sequence[CarrierSource], *, target: ExportTarget, grid: EnvelopeGrid) -> WrittenShape:
    """The volume curve an instrument built from ``sources`` carries, beside what sharing it costs.

    The shape is fitted from the sources' own levels and written against its own loudest moment, since the
    level each source stands at is restored by the step beside its own waveform rather than by the curve.
    A set holding nothing long enough to read leaves ``NO_ENVELOPE``, which sounds every waveform as it
    stands and costs its sources nothing.
    """
    shape = carrier_shape(sources, nodes=shape_nodes(target.envelope_point_bound))
    if shape is NO_SHAPE:
        return WrittenShape(envelope=NO_ENVELOPE, dispersion_db=NO_DISPERSION)

    level = curve_level(shape.curve, Clock.PLAYED)
    return WrittenShape(
        envelope=volume_envelope(written_level(level, reference_db=level.peak_db), grid),
        dispersion_db=shape.dispersion_db,
    )
