from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

from optisample.config.loop import LoopConfig, QualityConfig
from optisample.dsp.decay import LinearDecay, fit_linear_decay
from optisample.dsp.envelope import level_reading
from optisample.dsp.loop import Loop, LoopQuality, loop_candidates, loop_quality
from optisample.dsp.surrogate import SettledLoop, SettledLoops
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.base import Signal


@unique
class Gate(StrEnum):
    """The measurement a candidate fell outside of, which is the reason the ladder climbed past it."""

    SEAM = "seam"
    LEVEL = "level"
    TIMBRE = "timbre"


@dataclass(frozen=True)
class StoredLoop:
    """The loop a recording is stored around, and what measuring it said about storing it.

    ``settled`` is what the encoder is handed -- the region and the decline past it -- and ``quality`` the
    readings that admitted the loop, kept beside it so a report states the case for the loop that was
    stored as well as against the ones passed over.
    """

    settled: SettledLoop
    quality: LoopQuality

    @property
    def loop(self) -> Loop:
        """Where the loop sits, in frames of the recording as the stage received it."""
        return self.settled.loop

    @property
    def decay(self) -> LinearDecay | None:
        """The ramp a note held past the stored span falls on."""
        return self.settled.decay


@dataclass(frozen=True)
class RejectedLoop:
    """A candidate the ladder climbed past: where it sat, what it measured, and the gate it fell outside."""

    loop: Loop
    quality: LoopQuality
    gate: Gate


@dataclass(frozen=True)
class Settlement:
    """What the stage found for one recording: the loops it offers, and the candidates it turned down.

    ``offered`` holds every candidate clearing every gate, cheapest stored span first, which is the
    frontier the sweep prices a loop's length along: a longer region carries more of the material's own
    movement for more bytes, and which of those trades is worth buying is settled against a budget rather
    than here. It is empty where the material offers no candidate clearing the gates, which stores the
    recording as it was played. ``rejected`` states each candidate that fell outside a gate, so the reason
    a recording ended up unlooped is readable rather than inferred. ``search_s`` is the stretch the
    candidates were measured over, which bounds where a loop was worth placing.
    """

    offered: tuple[StoredLoop, ...]
    rejected: tuple[RejectedLoop, ...]
    search_s: float

    @property
    def loops(self) -> bool:
        """Whether the recording has a loop to be stored around, which the sweep prices against trimming."""
        return bool(self.offered)

    @property
    def cheapest(self) -> StoredLoop | None:
        """The offer storing the least, which is the most aggressive loop the recording supports.

        The one an audition leads with and a report reads first, since it is the loop a budget under
        pressure reaches for and so the one whose sound decides whether the frontier's cheap end is usable.
        """
        return self.offered[0] if self.offered else None

    @property
    def settled(self) -> SettledLoops:
        """What the encoder is handed: each offer's region and decline, in the order params index them."""
        return tuple(stored.settled for stored in self.offered)


def _failed_gate(quality: LoopQuality, config: QualityConfig) -> Gate | None:
    """The gate ``quality`` falls outside of, read in the order the artefacts are heard in.

    Returns ``None`` for a candidate clearing every gate, which is a loop worth storing. The wrap comes
    first because a step there is a click once per round wherever the rest sits, the level next because
    flattening a steep region lifts its noise along with its tail, and the timbre last because a loop
    holding a sound the material has moved on from is the subtlest of the three.
    """
    if quality.seam_step > config.max_seam_step:
        return Gate.SEAM

    if quality.level_drift_db > config.max_level_drift_db:
        return Gate.LEVEL

    if quality.spectral_distance > config.max_spectral_distance_db:
        return Gate.TIMBRE

    return None


def _ladder(signal: Signal, sample_rate: int, config: LoopConfig, root_hz: float) -> tuple[Loop, ...]:
    """The candidates in the order the settlement measures and offers them: cheapest first.

    A candidate costs what storing ``[0, loop.end)`` costs, so ordering by ``loop.end`` puts the most
    aggressive loop -- the earliest and shortest the geometry allows -- at the front and leaves the offers
    running from cheap to dear, which is the order a rate-distortion frontier is read along. The start
    breaks ties so one recording climbs the same ladder on every run.
    """
    candidates = loop_candidates(signal, sample_rate, config.geometry, root_hz)
    return tuple(sorted(candidates, key=lambda loop: (loop.end, loop.start)))


def settle_loop(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
    *,
    root_hz: float,
    search_s: float,
) -> Settlement:
    """The loops ``signal`` may be stored around, taken from the ladder its own material offers.

    ``root_hz`` is the pitch the recording was played at, which the material's period is searched around
    (:func:`~optisample.dsp.loop.loop_candidates`) and its level read over two of
    (:func:`~optisample.dsp.envelope.level_reading`), so both readings are taken over the stretch this note
    repeats in. One reading serves the whole ladder, so every candidate is measured alike and each region
    that ends up stored is levelled by the same curve that admitted it.

    Candidates are measured over the first ``search_s`` of the recording -- the longest stretch the
    material asks of it -- because a loop ending past that stores more than keeping the played span would
    and so wins nothing. Every rung of the ladder is measured and each one clearing every quality gate is
    offered, cheapest first (:func:`_ladder`), so the gates say which loops a recording supports at all and
    a budget says which of them is worth its bytes.

    Each offer's decline is fitted over the whole recording rather than the searched stretch, so the ramp a
    held note falls on is read off every level the recording states, and it is fitted per loop because
    where a region starts is where the stored material stops following the recording's own envelope.
    """
    searched = signal[: seconds_to_frames(search_s, sample_rate)]
    reading = level_reading(sample_rate, config.envelope, root_hz)
    offered: list[StoredLoop] = []
    rejected: list[RejectedLoop] = []
    for loop in _ladder(searched, sample_rate, config, root_hz):
        quality = loop_quality(searched, loop, sample_rate, config, reading)
        gate = _failed_gate(quality, config.quality)
        if gate is None:
            decay = fit_linear_decay(signal, sample_rate, loop, reading)
            offered.append(StoredLoop(settled=SettledLoop(loop=loop, decay=decay), quality=quality))
        else:
            rejected.append(RejectedLoop(loop=loop, quality=quality, gate=gate))

    return Settlement(offered=tuple(offered), rejected=tuple(rejected), search_s=search_s)
