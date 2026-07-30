from __future__ import annotations

from dataclasses import dataclass

from optisample.config.loop import LoopConfig
from optisample.dsp.surrogate import NO_LOOP
from optisample.keys import SampleKey
from optisample.loop.settle import Settlement
from optisample.loop.stage import LoopRequest, settle_loops
from optisample.optimize.orchestrate.audio import LoadedInstrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.dedupe import NO_MATERIAL_S, longest_note_by_pitch
from optisample.optimize.tasks import LoopMap, StoredRecordings
from optisample.progress import ProgressSink

Settlements = dict[SampleKey, Settlement]


@dataclass(frozen=True)
class LoopedInstrument:
    """The recordings a run works from, with the loop each one is stored around.

    ``settlements`` states the whole decision per recording -- the loop kept and the candidates climbed past
    -- which is what a report reads; ``settled`` narrows it to the part every encode needs.
    """

    loaded: LoadedInstrument
    settlements: Settlements

    @property
    def settled(self) -> LoopMap:
        """The loop each recording is stored around, keyed the way the audio is."""
        return {
            key: settlement.stored.settled if settlement.stored is not None else NO_LOOP
            for key, settlement in self.settlements.items()
        }

    @property
    def recordings(self) -> StoredRecordings:
        """What every stage after this one encodes from: the audio, its loops, and the rate they share."""
        return StoredRecordings(
            audio=self.loaded.audio,
            settled=self.settled,
            sample_rate=self.loaded.sample_rate,
        )

    @property
    def looped_recordings(self) -> int:
        """How many survivors ended up with a loop, which is how many samples store less than they play."""
        return sum(1 for settlement in self.settlements.values() if settlement.loops)


def loop_requests(loaded: LoadedInstrument) -> tuple[LoopRequest, ...]:
    """One request per surviving recording: its audio, and the longest stretch the material asks of its pitch.

    A loop ending past that stretch stores more than keeping the played span would, so the search is bounded
    by it. Requests come back in key order, which is the order the settlements are reported in.
    """
    longest = longest_note_by_pitch(loaded.instrument.material)
    return tuple(
        LoopRequest(
            key=key,
            signal=loaded.audio[key],
            search_s=longest.get(key.pitch, NO_MATERIAL_S),
        )
        for key in sorted(loaded.audio)
    )


def settle_run_loops(
    loaded: LoadedInstrument,
    config: LoopConfig,
    *,
    workers: int,
    progress: ProgressSink,
) -> LoopedInstrument:
    """Settle the loop of every recording a run holds, at the rate the run analyses them at.

    Runs on the loaded audio, so the frames a loop names index into exactly the recordings the sweep goes on
    to encode and the stages after this one only ever shorten them.
    """
    requests = loop_requests(loaded)
    return LoopedInstrument(
        loaded=loaded,
        settlements=settle_loops(requests, config, loaded.sample_rate, workers=workers, progress=progress),
    )


def run_loops(loaded: LoadedInstrument, settings: OptimizeSettings) -> LoopedInstrument:
    """The loops one run stores: settled where it asks for them, and left off where it asks for none.

    A run storing no loops keeps every sample over the span its material plays, which is what the sweep
    prices when nothing is settled to price a loop against.
    """
    if not settings.loops:
        return LoopedInstrument(loaded=loaded, settlements={})

    return settle_run_loops(loaded, settings.loop, workers=settings.workers, progress=settings.progress)
