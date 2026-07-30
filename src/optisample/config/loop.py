from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.stage import StageConfig


class GeometryConfig(ConfigModel):
    """Where a loop may sit in a note and how long it runs: the search band, the window, and the lengths.

    ``min_hz`` and ``max_hz`` bound the fundamental a period is searched for, and ``min_correlation`` is the
    autocorrelation peak a recording clears to count as periodic at all. ``attack_skip_s`` and
    ``tail_skip_s`` bound the steady window a loop is placed inside, so a loop begins past the onset
    transient and ends before the release. ``min_periods`` and ``min_loop_s`` together set the shortest loop
    that may be stored, which every candidate clears.

    ``placements`` is how many starts are spread through the steady window and ``length_multiples`` the
    lengths each start is offered as multiples of that floor, so the two say how wide a ladder
    :func:`~optisample.dsp.loop.loop_candidates` lays out. ``max_estimation_s`` caps the stretch the period
    is read off, which keeps the estimate on the part of the note that holds one pitch.
    """

    min_hz: Annotated[float, Field(gt=0.0)]
    max_hz: Annotated[float, Field(gt=0.0)]
    min_correlation: float
    min_periods: Annotated[int, Field(ge=1)]
    min_loop_s: Annotated[float, Field(gt=0.0)]
    placements: Annotated[int, Field(ge=1)]
    length_multiples: Annotated[tuple[Annotated[int, Field(ge=1)], ...], Field(min_length=1)]
    attack_skip_s: Annotated[float, Field(ge=0.0)]
    tail_skip_s: Annotated[float, Field(ge=0.0)]
    max_estimation_s: Annotated[float, Field(gt=0.0)]


class SeamConfig(ConfigModel):
    """How the wrap is blended, so a loop returns to its start on the motion the waveform already made.

    ``crossfade_s`` is the stretch before the loop end that is ramped toward the frames preceding the loop
    start, which is what carries ``signal[end - 1]`` onto ``signal[start - 1]`` and makes the wrap
    continuous (:func:`~optisample.dsp.loop.crossfade_loop`).
    """

    crossfade_s: Annotated[float, Field(ge=0.0)]


class QualityConfig(ConfigModel):
    """What a loop must measure to be worth storing, which is how far the ladder may be pushed for bytes.

    ``max_seam_step`` bounds the wrap in units of the loop region's own typical frame-to-frame motion, so
    1.0 admits a wrap as smooth as the waveform already moves and a larger value admits a step a listener
    starts to hear once per round. ``max_spectral_distance_db`` bounds the log-spectral distance between the
    loop region and the stretch it plays in place of, so a loop holds a timbre the material still has.

    Together they bound how aggressive a loop may be: candidates are offered cheapest first and the first
    one clearing both is stored, so tightening either gate buys a later, longer, more faithful loop and
    loosening it buys bytes.
    """

    max_seam_step: Annotated[float, Field(gt=0.0)]
    max_spectral_distance_db: Annotated[float, Field(gt=0.0)]


class LoopConfig(StageConfig):
    """How one recording is looped: where the loop sits, how its wrap is blended, and what it must measure."""

    geometry: GeometryConfig
    seam: SeamConfig
    quality: QualityConfig
