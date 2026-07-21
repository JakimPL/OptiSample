"""Run configuration, the per-instrument context, and the result DTOs the artifact stages share.

:class:`DumpSettings` is what the CLI hands in -- which strategies to run, whether to render through
openmpt123, and the config the optimizer + exporter need. :class:`DumpContext` bundles the
once-per-instrument inputs (loaded audio, the eval context, the pitch->task lookup) so the unit builder
and the dumper take one object instead of a long argument list. :class:`PlanArtifacts` and
:class:`DumpResult` are what a dump reports back: what each strategy produced and where it landed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.model import NoteEvent
from optisample.optimize.orchestrate import OptimizeSettings
from optisample.optimize.tasks import AudioMap, EvalContext, PitchTask


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


@dataclass(frozen=True)
class DumpContext:
    """Inputs shared across both strategies for one instrument."""

    audio: AudioMap
    sample_rate: int
    material: tuple[NoteEvent, ...]
    eval_context: EvalContext
    tasks_by_pitch: dict[int, PitchTask]
    settings: DumpSettings


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
