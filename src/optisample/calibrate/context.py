from dataclasses import dataclass
from typing import Final

from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.composite import CompositeFidelity
from trackmod.spec.levels import MAX_VOLUME

_DEFAULT_PROBE_DURATION_S: Final = 0.5


@dataclass(frozen=True)
class NoteProbe:
    """One note to render for calibration: which key, how loud (0..64), and for how long."""

    pitch: int
    volume: int = MAX_VOLUME
    duration_s: float = _DEFAULT_PROBE_DURATION_S


@dataclass(frozen=True)
class RendererAgreement:
    """How closely the surrogate and openmpt123 agree on one rendered note.

    ``breakdown`` is the raw, unweighted per-metric distance, which makes this a *calibration*
    result: it shows *which* term drives any disagreement. Log-scale terms (``mcd``, ``logmel_l1``)
    can inflate on near-silent frames even when the waveforms are audibly identical, so read
    ``distance`` alongside the breakdown.
    """

    probe: NoteProbe
    distance: float
    loudness_delta_lu: float
    breakdown: dict[str, float]


@dataclass(frozen=True)
class CalibrationContext:
    """Everything the calibrator renders and scores a note with.

    ``render`` is how ``openmpt123`` renders the module, ``playback`` is the clock the probe module
    carries, ``target`` is the tracker format that module is written as, and ``composite`` is the
    loudness-matched metric the two engines are compared under.
    """

    render: RenderConfig
    playback: PlaybackConfig
    target: ExportTarget
    composite: CompositeFidelity
