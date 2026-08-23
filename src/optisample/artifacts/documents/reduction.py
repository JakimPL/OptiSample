from __future__ import annotations

from optisample.artifacts.serialize import Frozen
from optisample.config.reduce import DedupeKey
from optisample.music import note_name
from optisample.optimize.reduce.summary import ReductionSummary
from optisample.optimize.reduce.trim import RecordingScreen


class KeptRecordingRecord(Frozen):
    """One recording that survived deduplication, measured against the material its pitch plays.

    ``covers_material`` is false when the survivor runs shorter than its pitch's longest note, in which
    case the objective scores it over as much of the note as was recorded.
    """

    key: str
    pitch: int
    velocity: int
    duration_s: float
    required_duration_s: float
    covers_material: bool


class StoredFormatRecord(Frozen):
    """The format the reduction settled for a pitch's stored sample, from the recording's own content.

    The rate follows the recording's own band, so it is settled here; ``depths`` are the grids the sweep
    then prices that rate at, and which of them a sample ends up stored on is read off the plan.
    """

    target_rate: int
    depths: list[int]


class NarrowedGridRecord(Frozen):
    """What the pre-pass settled for one pitch: the band it read and the format it stores at.

    ``useful_rate_hz`` is the rate the recording's own content asks for and ``stored`` the ladder rung
    reaching it, so a reader sees both the measurement and the format it named. ``swept`` counts the
    encodings the sweep then runs for this pitch, which is that one format over each stored span it offers.
    """

    pitch: int
    note: str
    useful_rate_hz: float
    stored: StoredFormatRecord
    swept: int


class ReductionDocument(Frozen):
    """The pre-optimization stage's decisions: what survived ingest and how small the search space got.

    The ``*_recordings``/``*_notes`` counts are the before side of each reduction axis; ``recordings`` and
    ``grids`` are the after side, per identity and per played pitch.
    """

    listed_recordings: int
    kept_recordings: int
    played_notes: int
    scored_classes: int
    recordings: list[KeptRecordingRecord]
    grids: list[NarrowedGridRecord]


class WrittenSampleRecord(Frozen):
    """One survivor as a reduced dataset holds it: the file written and the index its notes join on.

    ``index`` leads the filename, which is what a later ingest reads to route each note back to this
    recording. ``frames`` and ``duration_s`` measure what was written, so a dataset trimmed to what the
    material asks for states the length it kept.
    """

    index: int
    key: str
    file: str
    frames: int
    duration_s: float


class ScreenRecord(Frozen):
    """What admitting the recordings cost: the ones left out, and the material that left unplayable.

    ``silenced`` names each recording whose peak stayed under the configured silence floor, so a reader
    sees which slots the dataset holds no audio for. ``unplayable`` are the pitches those losses stripped
    of every recording, and ``dropped_notes`` how many played notes went with them.
    """

    silenced: list[str]
    unplayable: list[int]
    dropped_notes: int


class ReducedDocument(Frozen):
    """What one reduce run produced: the dataset it wrote and the decisions that shaped it.

    ``dedupe_key`` is the identity the survivors were kept under, so a run reading this dataset back
    states the projection it already stands at. ``sample_rate`` is the analysis rate every survivor was
    written at, which is the rate the bands in ``reduction`` were measured over. ``screen`` states
    what the dataset leaves out, beside the ``samples`` it holds.
    """

    instrument_id: str
    dedupe_key: DedupeKey
    sample_rate: int
    samples: list[WrittenSampleRecord]
    screen: ScreenRecord
    reduction: ReductionDocument


def screen_record(screen: RecordingScreen) -> ScreenRecord:
    """The load-time screen as a document, so a dataset states what it left behind as well as what it holds."""
    return ScreenRecord(
        silenced=[key.label for key in screen.silenced],
        unplayable=list(screen.unplayable),
        dropped_notes=screen.dropped_notes,
    )


def reduction_document(reduction: ReductionSummary) -> ReductionDocument:
    """The pre-optimization stage's own outcome as a document, for the plan tree and the reduced dataset.

    Both consumers state the same reduction, so a dataset written by ``reduce`` and a plan produced from
    it report their shared search space in one shape.
    """
    return ReductionDocument(
        listed_recordings=reduction.listed_recordings,
        kept_recordings=reduction.kept_recordings,
        played_notes=reduction.played_notes,
        scored_classes=reduction.scored_classes,
        recordings=[
            KeptRecordingRecord(
                key=recording.key.label,
                pitch=recording.key.pitch,
                velocity=recording.key.velocity,
                duration_s=recording.duration_s,
                required_duration_s=recording.required_duration_s,
                covers_material=recording.covers_material,
            )
            for recording in reduction.recordings
        ],
        grids=[
            NarrowedGridRecord(
                pitch=grid.pitch,
                note=note_name(grid.pitch),
                useful_rate_hz=grid.useful_rate_hz,
                stored=StoredFormatRecord(
                    target_rate=grid.stored.target_rate,
                    depths=list(grid.stored.depths),
                ),
                swept=len(grid.encodings),
            )
            for grid in reduction.grids
        ],
    )
