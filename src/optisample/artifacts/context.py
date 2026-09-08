from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.artifacts.instruments.dump import InstrumentSettings, WrittenInstruments
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
    from a loaded ``OptiConfig`` (see :func:`optisample.cli.settings._dump_settings`). The remaining flags are
    behavioral toggles, so they keep ergonomic defaults.

    ``post_loop`` says whether the standalone file written for each stored voice keeps the rest of its take
    behind the region it wraps on. It reaches those files alone: the module, the bank and the sample WAVs
    carry the waveform the plan was priced at, so the byte accounting stands whichever way it is set.
    """

    optimize: OptimizeSettings
    render: RenderConfig
    playback: PlaybackConfig
    envelope: EnvelopeConfig
    post_loop: bool
    render_ground_truth: bool = True  # render module + per-note through openmpt123 if it is installed
    grouped: bool = True
    ungrouped: bool = True

    @property
    def progress(self) -> ProgressSink:
        """Where the dump reports its stages: the sink the optimization run already reports through."""
        return self.optimize.progress

    @property
    def instruments(self) -> InstrumentSettings:
        """What the plan's own stored samples are written as standalone instruments with.

        The clock is the one the written module starts on, since the samples beside it are the very
        waveforms that module plays and an envelope written for them is counted in ticks of it.
        """
        return InstrumentSettings(
            encode=self.optimize.encode,
            target=self.optimize.target,
            release_s=self.envelope.release_s,
            configured_tempo_bpm=self.playback.tempo,
            progress=self.progress,
        )


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
        """The rate every recording was analyzed at, which every stored sample is measured against."""
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


NO_INSTRUMENTS: Final = None  # what a strategy the budget affords no allocation of leaves its samples dir


@dataclass(frozen=True)
class PlanArtifacts:
    """What one strategy produced under its subdirectory (or why it could not).

    Its directory is ``DumpResult.directory / name``; only the per-strategy outcome is kept here.
    ``instruments`` states where its stored samples landed as standalone instruments, which a strategy
    holding a plan writes one of per sample.
    """

    name: str
    reason: str | None
    rendered: bool  # whether an openmpt123 ground-truth render was written
    objective: float | None
    used_bytes: int | None
    instruments: WrittenInstruments | None
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
