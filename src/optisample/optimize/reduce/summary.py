from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from optisample.config.dsp import LoopConfig
from optisample.config.reduce import DedupeConfig
from optisample.dsp.surrogate import EncodingParams
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec
from optisample.optimize.operating_points import sweep_param_grid
from optisample.optimize.reduce.bandwidth import (
    ClipDemand,
    SweepInputs,
    candidate_params,
    useful_rate_hz,
)
from optisample.optimize.reduce.dedupe import (
    NO_MATERIAL_S,
    holds_material,
    longest_note_by_pitch,
    required_duration_s,
)
from optisample.optimize.reduce.keys import SampleKey
from optisample.progress import ProgressSink

_OWN_KEY: Final = 0  # a key sounding its own recording plays it at the pitch it was recorded at
_ONE_KEY: Final = 1  # and that one stored sample answers for that one key alone
_NARROW_LABEL: Final = "Narrowing stored grids"


class StoredClip(Protocol):
    """What the pre-pass reads about the sample one pitch stores: its recording and how long it is held.

    Stated as a protocol so the summary measures the same pitch tasks the sweep scores, while the
    reduction subpackage stays a leaf the task layer builds on.
    """

    @property
    def pitch(self) -> int: ...

    @property
    def representative(self) -> Signal: ...

    @property
    def max_duration_s(self) -> float: ...

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
class NarrowedGrid:
    """The encodings the sweep runs for the sample one pitch stores, and the band that bounds them.

    ``useful_rate_hz`` is the stored rate carrying everything the recording still contributes at its own
    key (see :func:`~optisample.optimize.reduce.bandwidth.useful_rate_hz`); ``shortlist`` is what the
    frontier around the budget's per-key share leaves of the full grid.
    """

    pitch: int
    useful_rate_hz: float
    shortlist: tuple[EncodingParams, ...]


@dataclass(frozen=True)
class ReductionSummary:
    """How much smaller the pre-optimization stage made the problem the allocation then solves.

    Each pair states one axis of the reduction: the recorded grid down to one survivor per identity, the
    material down to the classes that reconstruct alike, and the ``(loop, depth, rate)`` grid down to the
    shortlist each pitch is swept over. ``grids`` covers the samples the ungrouped strategy stores, one
    per played pitch; pitch-zone grouping narrows again per zone, around what that zone's span and key
    count ask of its representative.
    """

    listed_recordings: int
    played_notes: int
    scored_classes: int
    grid_size: int
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
    def shortlisted(self) -> int:
        """Encodings the run sweeps in total, summed over the pitches the ungrouped strategy stores."""
        return sum(len(grid.shortlist) for grid in self.grids)

    def shortlists(self) -> dict[int, tuple[EncodingParams, ...]]:
        """The encodings to sweep, keyed by the pitch whose recording is stored under them."""
        return {grid.pitch: grid.shortlist for grid in self.grids}


@dataclass(frozen=True)
class ReductionInputs:
    """The run-wide inputs the summary is measured against (bundled to stay under the argument limit).

    ``dedupe`` and ``loop`` set how long a kept recording has to be; ``context`` is the run's scoring
    context, which is what prices and narrows a stored grid; ``progress`` is where the pre-pass reports
    how many pitches it has narrowed so far.
    """

    dedupe: DedupeConfig
    loop: LoopConfig
    context: SweepInputs
    progress: ProgressSink


def _grid_size(context: SweepInputs) -> int:
    """How many encodings the sweep enumerates per stored sample, a count the trim leaves unchanged."""
    return len(tuple(sweep_param_grid(context.sweep, context.sample_rate, trim_s=None)))


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
                inputs.dedupe,
                inputs.loop,
            ),
        )
        for key in sorted(audio)
    )


def _narrowed_grid(clip: StoredClip, inputs: ReductionInputs) -> NarrowedGrid:
    """Narrow one pitch's stored grid, at the demand a key sounding its own recording makes of it."""
    demand = ClipDemand(trim_s=clip.max_duration_s, delta_semitones=_OWN_KEY, key_count=_ONE_KEY)
    context = inputs.context
    return NarrowedGrid(
        pitch=clip.pitch,
        useful_rate_hz=useful_rate_hz(clip.representative, demand, context.sample_rate, context.bandwidth),
        shortlist=candidate_params(clip.representative, demand, context),
    )


def summarize_reduction(
    instrument: InstrumentSpec,
    clips: Sequence[StoredClip],
    audio: Mapping[SampleKey, Signal],
    inputs: ReductionInputs,
) -> ReductionSummary:
    """Run the bandwidth pre-pass over every played pitch and record what the whole stage reduced.

    This is where the shortlist is drawn, once per run, so the sweep that follows scores exactly the
    grid the report and the artifacts state. ``clips`` are the pitch tasks the material plays, whose
    class counts say how far merging collapsed the notes.
    """
    return ReductionSummary(
        listed_recordings=len(instrument.samples),
        played_notes=len(instrument.material),
        scored_classes=sum(clip.scored_classes for clip in clips),
        grid_size=_grid_size(inputs.context),
        recordings=_kept_recordings(instrument, audio, inputs),
        grids=tuple(
            _narrowed_grid(clip, inputs) for clip in inputs.progress.track(clips, label=_NARROW_LABEL, total=len(clips))
        ),
    )
