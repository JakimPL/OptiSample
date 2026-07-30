from dataclasses import dataclass
from pathlib import Path

from optisample.config.export import EnvelopeConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.layers.bands import VelocityLayers
from optisample.optimize.layers.tasks import layered_tasks
from optisample.optimize.orchestrate import RunInputs
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.tasks import EvalContext, PitchTask, StoredRecordings
from optisample.progress import ProgressSink


@dataclass(frozen=True)
class DumpSettings:
    """What to dump and how (bundled to keep call sites small).

    ``optimize``/``render``/``playback``/``envelope`` carry the config the run needs; the CLI builds them
    from a loaded ``OptiConfig`` (see :func:`optisample.cli._dump_settings`). The remaining flags are
    behavioural toggles, so they keep ergonomic defaults.
    """

    optimize: OptimizeSettings
    render: RenderConfig
    playback: PlaybackConfig
    envelope: EnvelopeConfig
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

    instrument: InstrumentSpec
    recordings: StoredRecordings
    inputs: RunInputs
    settings: DumpSettings

    @property
    def sample_rate(self) -> int:
        """The rate every recording was analysed at, which every stored sample is measured against."""
        return self.recordings.sample_rate

    @property
    def material(self) -> tuple[NoteEvent, ...]:
        """Every note the instrument plays, which is what the written module auditions."""
        return tuple(self.instrument.material or [])

    @property
    def eval_context(self) -> EvalContext:
        """The scoring context every strategy measures its reconstructions with."""
        return self.inputs.context

    def layer_tasks(self, layers: VelocityLayers) -> dict[tuple[int, int], PitchTask]:
        """The pitch tasks one plan's velocity split scores, keyed by the layer and pitch each covers.

        A key played softly and loudly holds one task per layer, each scoring only the notes its own band
        covers, so a plan's stored samples are measured against exactly the material they answer for. The
        single full-range split answers the tasks the run was prepared with, key for key.
        """
        return {
            (layer, task.pitch): task
            for layer, tasks in enumerate(layered_tasks(self.instrument, self.inputs.task_inputs, layers))
            for task in tasks
        }


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
