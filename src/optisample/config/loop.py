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
    that may be stored, which every candidate clears, alongside the analysis window the quality gates read a
    candidate over (:func:`~optisample.dsp.loop.shortest_loop_frames`) -- so however short a floor is asked
    for, the loop offered is one every gate measured.

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


class EnvelopeConfig(ConfigModel):
    """How the level a recording holds is read off it, which is the curve levelling and the decay ramp read.

    ``lowest_hz`` is the lowest frequency the reading treats as sound. The weighting spans two of its
    periods (:func:`~optisample.dsp.envelope.power_kernel`), so the power ripple every tone from half of it
    upward carries averages out and what the reading is left holding is the level. Raising it sharpens what
    the curve tracks at an onset and widens the band the curve itself occupies, so it takes proportionally
    more points to hold.

    ``floor_db`` places the quietest level the reading states, that far under the recording's own peak. It
    holds the curve strictly positive, leaves a silent stretch at the level it was recorded at, and scales
    with the material, so a recording captured hot and the same recording captured quiet read as one curve
    shifted and one carrier.
    """

    lowest_hz: Annotated[float, Field(gt=0.0)]
    floor_db: Annotated[float, Field(gt=0.0)]


class SeamConfig(ConfigModel):
    """How the wrap is blended, so a loop returns to its start on the motion the waveform already made.

    The stretch before the loop end is ramped toward the frames preceding the loop start, which carries
    ``signal[end - 1]`` onto ``signal[start - 1]`` and makes the wrap continuous
    (:func:`~optisample.dsp.loop.crossfade_loop`). ``fade_share`` states how much of that stretch is blended
    as a share of the loop's own length and ``min_fade_s`` floors it in seconds, so every loop is blended in
    proportion to the round it makes and a short one still gets a blend long enough to carry motion. The
    material preceding the loop start is what the blend reaches for, so that stretch bounds it
    (:func:`~optisample.dsp.loop.seam_frames`).

    The law the two sides are weighted by follows from how alike they measure
    (:func:`~optisample.dsp.loop.crossfade_loop`), which is what holds the level of a blend of material that
    stayed in phase and of material whose partials have drifted apart alike.
    """

    fade_share: Annotated[float, Field(ge=0.0, le=1.0)]
    min_fade_s: Annotated[float, Field(ge=0.0)]


class QualityConfig(ConfigModel):
    """What a loop must measure to be worth storing, which is how far the ladder may be pushed for bytes.

    ``max_seam_step`` bounds the wrap in units of the loop region's own typical frame-to-frame motion, so
    1.0 admits a wrap as smooth as the waveform already moves and a larger value admits a step a listener
    starts to hear once per round. ``max_level_drift_db`` bounds how far the region's own level falls across
    it, which is the gain holding it at one level asks of the material (:func:`~optisample.dsp.loop.level_loop`):
    a region falling faster than this is flattened only by fighting it, lifting its noise floor along with
    its tail. Levelling reaches to a ceiling of its own, so a gate set past that ceiling admits regions the
    line through their levels flattens in part. ``max_spectral_distance_db`` bounds the log-spectral distance
    between the loop region and the stretch it plays in place of, so a loop holds a timbre the material
    still has.

    Together they bound how aggressive a loop may be: candidates are offered cheapest first and the first
    one clearing all three is stored, so tightening any gate buys a later, longer, more faithful loop and
    loosening it buys bytes.
    """

    max_seam_step: Annotated[float, Field(gt=0.0)]
    max_level_drift_db: Annotated[float, Field(gt=0.0)]
    max_spectral_distance_db: Annotated[float, Field(gt=0.0)]


class LoopConfig(StageConfig):
    """How one recording is looped: where the loop sits, the level it is held at, its wrap, and its gates."""

    geometry: GeometryConfig
    envelope: EnvelopeConfig
    seam: SeamConfig
    quality: QualityConfig
