import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from optisample.io.audio import probe_wav
from optisample.model import (
    InstrumentSpec,
    Manifest,
    NoteEvent,
    ProjectSpec,
    SourceSample,
)

NOTES_SUFFIX: Final = ".notes.json"  # the manifest extension every NoteExtractor dataset is named by
NO_PADDING_S: Final = 0.0  # a recording cut to its note exactly, which is what this project's own writers produce
NO_TEMPO: Final = None  # what a dataset recorded away from a clock of its own states

_RENDER_FIELD: Final = "render"
_TEMPO_FIELD: Final = "tempo_bpm"


class RenderWindow(BaseModel):
    """Where one note sits in the render a manifest was extracted from, and which WAV holds it."""

    model_config = ConfigDict(extra="ignore")

    index: int
    start_seconds: float
    release_end_seconds: float


class ManifestNote(BaseModel):
    """One note as a ``.notes.json`` states it: what was played, and where it was recorded."""

    model_config = ConfigDict(extra="ignore")

    pitch: int
    velocity: int
    cc_averages: dict[int, float] = Field(default_factory=dict)
    render: RenderWindow

    @property
    def duration_s(self) -> float:
        """The audible span of the note: its onset through the end of its release."""
        return self.render.release_end_seconds - self.render.start_seconds


class RollSettings(BaseModel):
    """The audio the trimmer kept around each note, as the manifest of that extraction records it.

    ``pre_roll_seconds`` reaches back before the onset and ``post_roll_seconds`` carries on past the
    release end. The split run states them and the manifest carries them, so a dataset says for itself
    how its recordings were cut.
    """

    model_config = ConfigDict(extra="ignore")

    pre_roll_seconds: float = Field(ge=0.0)
    post_roll_seconds: float = Field(ge=0.0)


class ManifestSettings(BaseModel):
    """The extraction settings a manifest records, of which the ingest reads the rolls."""

    model_config = ConfigDict(extra="ignore")

    rolls: RollSettings


class ManifestRender(BaseModel):
    """What the render a manifest was extracted from was played at, of which the pipeline reads the clock.

    ``tempo_bpm`` is the tempo the material sounds at, which a volume envelope written for one of its
    recordings is counted in ticks of, since both tracker formats measure envelope time against the
    clock the module runs.
    """

    model_config = ConfigDict(extra="ignore")

    tempo_bpm: float = Field(gt=0.0)


class NotesManifest(BaseModel):
    """The fields a NoteExtractor ``.notes.json`` carries that the pipeline reads.

    ``settings`` is what the extraction was carried out under, and a manifest states it, so reading a
    dataset takes only where it lives. ``render`` states the clock the material was played at on the
    datasets extracted from a render that ran to one.
    """

    model_config = ConfigDict(extra="ignore")

    settings: ManifestSettings
    notes: list[ManifestNote]
    render: ManifestRender | None = NO_TEMPO

    @property
    def tempo_bpm(self) -> float | None:
        """The clock this dataset was played at, where it states one."""
        return None if self.render is None else self.render.tempo_bpm


def read_manifest(notes_json: Path | str) -> NotesManifest:
    """The fields of one ``.notes.json`` the pipeline reads, validated against the shape it expects."""
    raw: Any = json.loads(Path(notes_json).read_text(encoding="utf-8"))
    return NotesManifest.model_validate(raw)


def index_of_wav(path: Path | str) -> int:
    """Read a trimmed WAV's render index from its filename's leading token (before the first ``_``)."""
    token = Path(path).name.split("_", 1)[0]
    try:
        return int(token)
    except ValueError as error:
        raise ValueError(f"WAV filename {Path(path).name!r} has no leading integer render index") from error


@dataclass(frozen=True)
class IngestSettings:
    """The fields a source does not carry, supplied by the caller (CLI flags + config).

    ``instrument_id`` names the single instrument, ``budget_kb`` is its byte budget, and ``project`` holds
    the project-wide fidelity settings. ``keep_tail`` stores each recording through the padding past its
    release end, which is how a run listens to the material the default cut leaves out.

    ``pre_roll_s`` / ``post_roll_s`` state how a directory of recordings was padded, which is the one
    source shape saying nothing about itself; a ``.notes.json`` records its own rolls and is read by them.
    """

    instrument_id: str
    budget_kb: float
    project: ProjectSpec
    pre_roll_s: float
    post_roll_s: float
    keep_tail: bool


def _trail_out_s(recorded_s: float, sounding_s: float, post_roll_s: float) -> float:
    """The release padding one recording holds past its note, which the loader drops.

    ``recorded_s`` is what the file holds from the note onset onwards and ``sounding_s`` is how long the
    note itself sounds, so what is left over is the padding the trimmer added. Measuring it from the file
    is exact where the trimmer's cut ran into the end of the render and kept less padding than it was
    asked for. Bounding it by ``post_roll_s`` leaves the extra length in place where a recording was
    deliberately stored longer than its note, which is how this project's own reduced datasets are written.
    """
    return max(NO_PADDING_S, min(recorded_s - sounding_s, post_roll_s))


def _sample_of(note: ManifestNote, wav: Path, rolls: RollSettings, *, keep_tail: bool) -> SourceSample:
    """One manifest note as a recording the optimizer may store, with the padding at each end measured.

    The lead-in is bounded by the onset because the trimmer's cut starts at the beginning of the render
    where the pre-roll reaches back past it, and the trail is measured from the file for the matching
    reason at the other end.
    """
    lead_in = min(rolls.pre_roll_seconds, note.render.start_seconds)
    from_onset = probe_wav(wav).duration_s - lead_in
    trail_out = NO_PADDING_S if keep_tail else _trail_out_s(from_onset, note.duration_s, rolls.post_roll_seconds)
    return SourceSample(
        file=wav.resolve(),
        pitch=note.pitch,
        velocity=note.velocity,
        cc_averages=note.cc_averages,
        lead_in_s=lead_in,
        trail_out_s=trail_out,
    )


def load_notes(
    notes_json: Path | str,
    samples_dir: Path | str,
    settings: IngestSettings,
) -> Manifest:
    """Join a ``.notes.json`` to its samples directory into a single-instrument manifest.

    Each note becomes a :class:`~optisample.model.SourceSample` (matched to the WAV whose leading index
    token equals ``render.index``) and a :class:`~optisample.model.NoteEvent` whose ``duration_s`` is the
    audible span ``release_end_seconds - start_seconds``. The padding each recording was cut with comes
    from the manifest's own ``settings.rolls``, so a dataset is read by the rolls it was written with;
    every remaining field comes from ``settings`` (see :class:`IngestSettings`).

    Raises:
        ValueError: when a note names a render index no WAV of ``samples_dir`` carries.
    """
    notes_json = Path(notes_json)
    samples_dir = Path(samples_dir).resolve()
    parsed = read_manifest(notes_json)
    rolls = parsed.settings.rolls

    wavs = {index_of_wav(wav): wav for wav in sorted(samples_dir.glob("*.wav"))}

    samples: list[SourceSample] = []
    material: list[NoteEvent] = []
    for note in parsed.notes:
        wav = wavs.get(note.render.index)
        if wav is None:
            raise ValueError(f"note render index {note.render.index} has no WAV in {samples_dir}")
        samples.append(_sample_of(note, wav, rolls, keep_tail=settings.keep_tail))
        material.append(
            NoteEvent(
                pitch=note.pitch,
                velocity=note.velocity,
                cc_averages=note.cc_averages,
                duration_s=note.duration_s,
            )
        )

    instrument = InstrumentSpec(
        id=settings.instrument_id,
        budget_kb=settings.budget_kb,
        samples=samples,
        material=material,
        pre_roll_s=rolls.pre_roll_seconds,
        post_roll_s=rolls.post_roll_seconds,
        tempo_bpm=parsed.tempo_bpm,
    )
    return Manifest(project=settings.project, instruments=[instrument])


@dataclass(frozen=True)
class NoteRecord:
    """One note's consumed subset: the join index, its (pitch, velocity), duration, and CC averages."""

    index: int
    pitch: int
    velocity: int
    duration_s: float
    cc_averages: Mapping[int, float] = field(default_factory=dict)


def _render_block(tempo_bpm: float | None) -> dict[str, Any]:
    """The clock a written dataset states, carried on the datasets whose source stated one.

    A stage writes what it was given, so a dataset extracted from a render at a known tempo keeps saying
    so however many stages it passes through, and the instruments written from its recordings are counted
    in ticks of that clock.
    """
    if tempo_bpm is None:
        return {}

    return {_RENDER_FIELD: {_TEMPO_FIELD: tempo_bpm}}


def dump_notes(
    notes: Sequence[NoteRecord],
    path: Path | str,
    *,
    tracked_ccs: Sequence[int] = (),
    tempo_bpm: float | None = NO_TEMPO,
) -> None:
    """Write the consumed ``.notes.json`` subset for ``notes`` (render window ``[0, duration_s]``).

    The rolls are declared as none, because a written recording starts at its onset and is stored for as
    long as the pitch it serves asks for, so a later ingest keeps every frame of it.
    """
    data: dict[str, Any] = {
        **_render_block(tempo_bpm),
        "config": {"tracked_ccs": list(tracked_ccs)},
        "settings": {"rolls": {"pre_roll_seconds": NO_PADDING_S, "post_roll_seconds": NO_PADDING_S}},
        "notes": [
            {
                "pitch": note.pitch,
                "velocity": note.velocity,
                "cc_averages": {str(cc): value for cc, value in note.cc_averages.items()},
                "render": {
                    "index": note.index,
                    "start_seconds": 0.0,
                    "release_end_seconds": note.duration_s,
                },
            }
            for note in notes
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
