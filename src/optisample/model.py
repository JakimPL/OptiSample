from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from optisample.music import MIDI_MAX_VELOCITY

MidiNote = Annotated[int, Field(ge=0, le=MIDI_MAX_VELOCITY)]
MidiVelocity = Annotated[int, Field(ge=0, le=MIDI_MAX_VELOCITY)]
PositiveFloat = Annotated[float, Field(gt=0.0)]

Engine = Literal["openmpt"]
Interpolation = Literal["none", "linear", "cubic", "sinc"]

_NO_LEAD_IN: Final = 0.0
_NO_ROLL: Final = 0.0


class OptiSampleModel(BaseModel):
    """Base model: forbid unknown keys so manifest typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class SourceSample(OptiSampleModel):
    """One recorded note of an instrument, grouped by (pitch, velocity).

    ``cc_averages`` maps each tracked MIDI controller number to its time-weighted average over the note.
    ``lead_in_s`` is the pre-roll padding before the note onset that the loader trims so frame 0 lands
    on the onset. ``articulation`` is carried through the model; the optimizer keys on (pitch, velocity).
    """

    file: Path
    pitch: MidiNote
    velocity: MidiVelocity
    cc_averages: dict[int, float] = Field(default_factory=dict)
    lead_in_s: float = _NO_LEAD_IN
    articulation: str | None = None


class NoteEvent(OptiSampleModel):
    """A note the song actually plays for an instrument, with its held duration.

    ``cc_averages`` maps each tracked MIDI controller number to its time-weighted average over the note.
    ``duration_s`` is the musical span (onset to release), bounding the sample against the decaying tail.
    """

    pitch: MidiNote
    velocity: MidiVelocity
    cc_averages: dict[int, float] = Field(default_factory=dict)
    duration_s: PositiveFloat
    count: Annotated[int, Field(ge=1)] = 1

    @property
    def weight(self) -> float:
        """Usage weight = total playing time this event accounts for."""
        return self.count * self.duration_s


class InstrumentSpec(OptiSampleModel):
    """One instrument: its recorded samples, the material that uses it, and a budget.

    ``pre_roll_s`` and ``post_roll_s`` record how the samples were trimmed (the padding kept before the
    onset and after the release), carried for provenance.
    """

    id: str
    budget_kb: PositiveFloat
    samples: Annotated[list[SourceSample], Field(min_length=1)]
    material: Annotated[list[NoteEvent], Field(min_length=1)]
    pre_roll_s: float = _NO_ROLL
    post_roll_s: float = _NO_ROLL


class ProjectSpec(OptiSampleModel):
    """Project-wide settings that affect how fidelity is judged."""

    name: str
    target_engine: Engine = "openmpt"
    interpolation: Interpolation = "sinc"


class Manifest(OptiSampleModel):
    """The ingest contract: everything needed to run the optimizer on a project."""

    project: ProjectSpec
    instruments: Annotated[list[InstrumentSpec], Field(min_length=1)]
