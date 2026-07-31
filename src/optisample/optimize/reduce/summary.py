from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from optisample.config.loop import LoopConfig
from optisample.config.reduce import ReduceConfig
from optisample.dsp.surrogate import EncodingParams
from optisample.keys import SampleKey
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec
from optisample.optimize.reduce.dedupe import (
    NO_MATERIAL_S,
    holds_material,
    longest_note_by_pitch,
    required_duration_s,
)
from optisample.optimize.reduce.grids import (
    GridContext,
    NarrowedGrid,
    StoredClip,
    narrow_grids,
)
from optisample.progress import ProgressSink


class ScoredClip(StoredClip, Protocol):
    """A stored clip plus how many note classes are scored against it, which is what merging left.

    The summary measures both sides of the reduction, so it reads one field more of a pitch task than
    narrowing that pitch's grid does.
    """

    @property
    def scored_classes(self) -> int: ...


@dataclass(frozen=True)
class KeptRecording:
    """One surviving recording measured against the longest note the material asks of its pitch.

    ``duration_s`` is the decoded length from the note onset, and ``required_duration_s`` what
    :func:`~optisample.optimize.reduce.dedupe.required_duration_s` says the pitch demands. A recording
    falling under that is scored over as much of the note as was recorded, which
    :attr:`covers_material` reports so the shortfall reaches the report and the artifacts.
    """

    key: SampleKey
    duration_s: float
    required_duration_s: float

    @property
    def covers_material(self) -> bool:
        """Whether the kept recording holds every note its key has to serve, in full."""
        return holds_material(self.duration_s, self.required_duration_s)


@dataclass(frozen=True)
class ReductionSummary:
    """How much smaller the pre-optimization stage made the problem the allocation then solves.

    Each pair states one axis of the reduction: the recorded grid down to one survivor per identity, the
    material down to the classes that reconstruct alike, and the whole rate ladder down to the one format
    each pitch is stored at, swept over the stored spans it offers. ``grids`` covers the samples the ungrouped
    strategy stores, one per played pitch; pitch-zone grouping settles a format again per zone, from what
    that zone's span asks of its representative.
    """

    listed_recordings: int
    played_notes: int
    scored_classes: int
    recordings: tuple[KeptRecording, ...]
    grids: tuple[NarrowedGrid, ...]

    @property
    def kept_recordings(self) -> int:
        """How many identities survived deduplication, which is how many samples the run may store."""
        return len(self.recordings)

    @property
    def shortfalls(self) -> tuple[KeptRecording, ...]:
        """Kept recordings running shorter than their pitch's longest note, in key order."""
        return tuple(recording for recording in self.recordings if not recording.covers_material)

    @property
    def swept(self) -> int:
        """Encodings the run sweeps in total, summed over the pitches the ungrouped strategy stores."""
        return sum(len(grid.encodings) for grid in self.grids)

    def encodings(self) -> dict[int, tuple[EncodingParams, ...]]:
        """The encodings to sweep, keyed by the pitch whose recording is stored under them."""
        return {grid.pitch: grid.encodings for grid in self.grids}


@dataclass(frozen=True)
class ReductionInputs:
    """The run-wide inputs the summary is measured against (bundled to stay under the argument limit).

    ``reduce`` and ``loop`` set how long a kept recording has to be; ``context`` is what settles a
    stored grid; ``workers`` is how many processes share the pitches out between them; and ``progress`` is
    where the pre-pass reports how many pitches it has narrowed so far.
    """

    reduce: ReduceConfig
    loop: LoopConfig
    context: GridContext
    workers: int
    progress: ProgressSink


def _kept_recordings(
    instrument: InstrumentSpec,
    audio: Mapping[SampleKey, Signal],
    inputs: ReductionInputs,
) -> tuple[KeptRecording, ...]:
    """Every survivor in ``audio``, measured against what the material asks of the pitch it sits at."""
    longest = longest_note_by_pitch(instrument.material)
    sample_rate = inputs.context.sample_rate
    return tuple(
        KeptRecording(
            key=key,
            duration_s=len(audio[key]) / sample_rate,
            required_duration_s=required_duration_s(
                longest.get(key.pitch, NO_MATERIAL_S),
                inputs.reduce,
                inputs.loop,
            ),
        )
        for key in sorted(audio)
    )


def summarize_reduction(
    instrument: InstrumentSpec,
    clips: Sequence[ScoredClip],
    audio: Mapping[SampleKey, Signal],
    inputs: ReductionInputs,
) -> ReductionSummary:
    """Run the bandwidth pre-pass over every played pitch and record what the whole stage reduced.

    This is where each pitch's stored format is settled, once per run, so the sweep that follows scores
    exactly the encodings the report and the artifacts state. ``clips`` are the pitch tasks the material
    plays, whose class counts say how far merging collapsed the notes.
    """
    return ReductionSummary(
        listed_recordings=len(instrument.samples),
        played_notes=len(instrument.material),
        scored_classes=sum(clip.scored_classes for clip in clips),
        recordings=_kept_recordings(instrument, audio, inputs),
        grids=narrow_grids(clips, inputs.context, workers=inputs.workers, progress=inputs.progress),
    )
