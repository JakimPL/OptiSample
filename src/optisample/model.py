from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optisample.music import MIDI_MAX_VELOCITY

MidiNote = Annotated[int, Field(ge=0, le=MIDI_MAX_VELOCITY)]
MidiVelocity = Annotated[int, Field(ge=0, le=MIDI_MAX_VELOCITY)]
PositiveFloat = Annotated[float, Field(gt=0.0)]

Engine = Literal["openmpt"]
Interpolation = Literal["none", "linear", "cubic", "sinc"]


class OptiSampleModel(BaseModel):
    """Base model: forbid unknown keys so manifest typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class SourceSample(OptiSampleModel):
    """One recorded note of an instrument, grouped by (pitch, velocity, controller).

    ``controller`` is the CC0/C00 value averaged to a single scalar over the note.
    ``articulation`` is carried but ignored by the optimizer for now.
    """

    file: Path
    pitch: MidiNote
    velocity: MidiVelocity
    controller: float = 0.0
    articulation: str | None = None


class NoteEvent(OptiSampleModel):
    """A note the song actually plays for an instrument, with its held duration."""

    pitch: MidiNote
    velocity: MidiVelocity
    controller: float = 0.0
    duration_s: PositiveFloat
    count: Annotated[int, Field(ge=1)] = 1

    @property
    def weight(self) -> float:
        """Usage weight = total playing time this event accounts for."""
        return self.count * self.duration_s


class InstrumentSpec(OptiSampleModel):
    """One instrument: its recorded samples, the material that uses it, and a budget."""

    id: str
    budget_kb: PositiveFloat
    samples: Annotated[list[SourceSample], Field(min_length=1)]
    material: list[NoteEvent] | None = None
    material_midi: Path | None = None

    @model_validator(mode="after")
    def _exactly_one_material_source(self) -> Self:
        if (self.material is None) == (self.material_midi is None):
            raise ValueError("provide exactly one of 'material' or 'material_midi'")
        return self


class ProjectSpec(OptiSampleModel):
    """Project-wide settings that affect how fidelity is judged."""

    name: str
    target_engine: Engine = "openmpt"
    interpolation: Interpolation = "sinc"


class Manifest(OptiSampleModel):
    """The ingest contract: everything needed to run the optimizer on a project."""

    project: ProjectSpec
    instruments: Annotated[list[InstrumentSpec], Field(min_length=1)]
