from dataclasses import dataclass
from pathlib import Path

from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.model import NoteEvent
from optisample.optimize.orchestrate import RunInputs
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.tasks import AudioMap, EvalContext, PitchTask
from optisample.progress import ProgressSink


@dataclass(frozen=True)
class DumpSettings:
    """What to dump and how (bundled to keep call sites small).

    ``optimize``/``render``/``playback`` carry the config the run needs; the CLI builds them from a
    loaded ``OptiConfig`` (see :func:`optisample.cli._dump_settings`). The remaining flags are
    behavioural toggles, so they keep ergonomic defaults.
    """

    optimize: OptimizeSettings
    render: RenderConfig
    playback: PlaybackConfig
    render_ground_truth: bool = True  # render module + per-note through openmpt123 if it is installed
    grouped: bool = True
    ungrouped: bool = True

    @property
    def progress(self) -> ProgressSink:
        """Where the dump reports its stages: the sink the optimization run already reports through."""
        return self.optimize.progress


@dataclass(frozen=True)
class DumpContext:
    """Inputs shared across both strategies for one instrument.

    ``inputs`` is the prepared run both strategies allocate from, so the velocity map, the pitch tasks,
    the scoring context and the bandwidth pre-pass are computed once per instrument and each strategy
    reads the same ones back.
    """

    audio: AudioMap
    sample_rate: int
    material: tuple[NoteEvent, ...]
    inputs: RunInputs
    settings: DumpSettings

    @property
    def eval_context(self) -> EvalContext:
        """The scoring context every strategy measures its reconstructions with."""
        return self.inputs.context

    @property
    def tasks_by_pitch(self) -> dict[int, PitchTask]:
        """The prepared pitch tasks, keyed by the pitch each one covers."""
        return {task.pitch: task for task in self.inputs.tasks}


@dataclass(frozen=True)
class PlanArtifacts:
    """What one strategy produced under its subdirectory (or why it could not).

    Its directory is ``DumpResult.directory / name``; only the per-strategy outcome is kept here.
    """

    name: str
    reason: str | None
    rendered: bool  # whether an openmpt123 ground-truth render was written
    objective: float | None
    used_bytes: int | None
    elapsed_s: float  # wall-clock for this strategy end to end (optimize + artifact dump)

    @property
    def feasible(self) -> bool:
        """Whether the strategy fit the budget; an infeasible run carries its ``reason`` instead of a plan."""
        return self.reason is None


@dataclass(frozen=True)
class DumpResult:
    """The full dump for one instrument: where it went and how each strategy fared."""

    instrument_id: str
    directory: Path
    plans: tuple[PlanArtifacts, ...]
