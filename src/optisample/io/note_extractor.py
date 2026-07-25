"""Read NoteExtractor output into the optimizer's in-memory model (and a minimal writer for the demo).

NoteExtractor emits a samples directory of per-note WAVs named ``{index}_p{pitch}_v{velocity}_...wav``
plus a ``.notes.json`` manifest ``{config, notes[]}``. Each note is one isolated performance of a
pitch at a velocity, so it maps to both a recorded :class:`~optisample.model.SourceSample` and the
:class:`~optisample.model.NoteEvent` that plays it; the join key is ``render.index`` matched against
each WAV filename's leading index token.

:func:`load_notes` parses that pair into a :class:`~optisample.model.Manifest`. :func:`dump_notes`
writes the consumed subset back out, used by the synth demo and tests to round-trip through the same
on-disk ingest path (it is not a full NoteExtractor emulation).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample


class _Render(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    start_seconds: float
    release_end_seconds: float


class _Note(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pitch: int
    velocity: int
    cc_averages: dict[int, float] = Field(default_factory=dict)
    render: _Render


class _NotesFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    notes: list[_Note]


def index_of_wav(path: Path | str) -> int:
    """Read a trimmed WAV's render index from its filename's leading token (before the first ``_``)."""
    token = Path(path).name.split("_", 1)[0]
    try:
        return int(token)
    except ValueError as error:
        raise ValueError(f"WAV filename {Path(path).name!r} has no leading integer render index") from error


def load_notes(
    notes_json: Path | str,
    samples_dir: Path | str,
    *,
    instrument_id: str,
    budget_kb: float,
    project: ProjectSpec,
    pre_roll_s: float = 0.0,
    post_roll_s: float = 0.0,
) -> Manifest:
    """Join a ``.notes.json`` to its samples directory into a single-instrument manifest.

    Each note becomes a :class:`~optisample.model.SourceSample` (matched to the WAV whose leading index
    token equals ``render.index``) and a :class:`~optisample.model.NoteEvent` whose ``duration_s`` is the
    audible span ``release_end_seconds - start_seconds``. ``pre_roll_s`` and ``post_roll_s`` record the
    trimmer padding; each sample's ``lead_in_s`` is the pre-roll clamped to the note's own start so the
    loader can align frame 0 with the onset.
    """
    notes_json = Path(notes_json)
    samples_dir = Path(samples_dir).resolve()
    raw: Any = json.loads(notes_json.read_text(encoding="utf-8"))
    parsed = _NotesFile.model_validate(raw)

    wavs = {index_of_wav(wav): wav for wav in sorted(samples_dir.glob("*.wav"))}

    samples: list[SourceSample] = []
    material: list[NoteEvent] = []
    for note in parsed.notes:
        wav = wavs.get(note.render.index)
        if wav is None:
            raise ValueError(f"note render index {note.render.index} has no WAV in {samples_dir}")
        samples.append(
            SourceSample(
                file=wav.resolve(),
                pitch=note.pitch,
                velocity=note.velocity,
                cc_averages=note.cc_averages,
                lead_in_s=min(pre_roll_s, note.render.start_seconds),
            )
        )
        material.append(
            NoteEvent(
                pitch=note.pitch,
                velocity=note.velocity,
                cc_averages=note.cc_averages,
                duration_s=note.render.release_end_seconds - note.render.start_seconds,
            )
        )

    instrument = InstrumentSpec(
        id=instrument_id,
        budget_kb=budget_kb,
        samples=samples,
        material=material,
        pre_roll_s=pre_roll_s,
        post_roll_s=post_roll_s,
    )
    return Manifest(project=project, instruments=[instrument])


@dataclass(frozen=True)
class NoteRecord:
    """One note's consumed subset: the join index, its (pitch, velocity), duration, and CC averages."""

    index: int
    pitch: int
    velocity: int
    duration_s: float
    cc_averages: Mapping[int, float] = field(default_factory=dict)


def dump_notes(notes: Sequence[NoteRecord], path: Path | str, *, tracked_ccs: Sequence[int] = ()) -> None:
    """Write the consumed ``.notes.json`` subset for ``notes`` (render window ``[0, duration_s]``)."""
    data = {
        "config": {"tracked_ccs": list(tracked_ccs)},
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
