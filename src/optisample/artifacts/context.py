"""Run configuration and the per-instrument context the artifact stages share.

:class:`DumpSettings` is what the CLI hands in -- which strategies to run, whether to render through
openmpt123, and the config the optimizer + exporter need. :class:`_DumpContext` bundles the
once-per-instrument inputs (loaded audio, the eval context, the pitch->task lookup) so the unit builder
and the dumper take one object instead of a long argument list.
"""

from __future__ import annotations

from dataclasses import dataclass

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
class _DumpContext:
    """Inputs shared across both strategies for one instrument."""

    audio: AudioMap
    sample_rate: int
    material: tuple[NoteEvent, ...]
    ctx: EvalContext
    tasks_by_pitch: dict[int, PitchTask]
    settings: DumpSettings
