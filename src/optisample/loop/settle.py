from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

from optisample.config.loop import LoopConfig, QualityConfig
from optisample.dsp.decay import LinearDecay, fit_linear_decay
from optisample.dsp.loop import Loop, LoopQuality, loop_candidates, loop_quality
from optisample.dsp.surrogate import NO_LOOP, SettledLoop
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
    """What the stage decided for one recording: the loop it keeps, and the candidates it climbed past.

    ``stored`` is :data:`NO_LOOP` where the material offers no candidate clearing the gates, which stores
    the recording as it was played. ``rejected`` states each candidate that was measured and passed over,
    so the reason a recording ended up unlooped is readable rather than inferred. ``search_s`` is the stretch
    the candidates were measured over, which bounds where a loop was worth placing.
    """

    stored: StoredLoop | None
    rejected: tuple[RejectedLoop, ...]
    search_s: float

    @property
    def loops(self) -> bool:
        """Whether the recording is stored around a loop, which is what the sweep prices against trimming."""
        return self.stored is not None


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


def _ladder(signal: Signal, sample_rate: int, config: LoopConfig) -> tuple[Loop, ...]:
    """The candidates in the order the settlement tries them: cheapest first.

    A candidate costs what storing ``[0, loop.end)`` costs, so ordering by ``loop.end`` puts the most
    aggressive loop -- the earliest and shortest the geometry allows -- at the front. The start breaks ties
    so one recording climbs the same ladder on every run.
    """
    candidates = loop_candidates(signal, sample_rate, config.geometry)
    return tuple(sorted(candidates, key=lambda loop: (loop.end, loop.start)))


def settle_loop(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
    *,
    search_s: float,
) -> Settlement:
    """The loop ``signal`` is stored around, taken from the ladder its own material offers.

    Candidates are measured over the first ``search_s`` of the recording -- the longest stretch the
    material asks of it -- because a loop ending past that stores more than keeping the played span would
    and so wins nothing. The ladder is climbed cheapest first (:func:`_ladder`) and the first candidate
    clearing every quality gate is kept, so the loop stored is the most aggressive one the recording
    supports and the gates are what say how aggressive that is.

    The decline is fitted over the whole recording rather than the searched stretch, so the ramp a held
    note falls on is read off every level the recording states.
    """
    searched = signal[: seconds_to_frames(search_s, sample_rate)]
    rejected: list[RejectedLoop] = []
    for loop in _ladder(searched, sample_rate, config):
        quality = loop_quality(searched, loop, sample_rate, config)
        gate = _failed_gate(quality, config.quality)
        if gate is None:
            decay = fit_linear_decay(signal, sample_rate, loop, config.envelope)
            return Settlement(
                stored=StoredLoop(settled=SettledLoop(loop=loop, decay=decay), quality=quality),
                rejected=tuple(rejected),
                search_s=search_s,
            )

        rejected.append(RejectedLoop(loop=loop, quality=quality, gate=gate))

    return Settlement(stored=NO_LOOP, rejected=tuple(rejected), search_s=search_s)
