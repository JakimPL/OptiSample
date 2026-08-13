from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.stage import StageConfig


class GeometryConfig(ConfigModel):
    """Where a loop may sit in a note and how long it runs: the search band, the window, and the lengths.

    ``detune_semitones`` is how far either side of the pitch a recording was played at its period is
    searched for, which leaves room for the tuning the instrument was recorded at and for the stretch a
    piano's own strings carry; ``min_correlation`` is the autocorrelation peak a recording clears to count
    as periodic at all, and together the two say which material a loop has purchase on. ``max_attack_s`` and
    ``tail_skip_s`` bound the steady window the rounds of
    :class:`FrontierConfig` are looked for inside: the window opens where the recording's
    own material settles (:func:`~optisample.dsp.similarity.settling_frame`) and ``max_attack_s`` holds that
    to a stretch every note leaves room past, which is part of the room the reduce stage reserves for a loop
    search (:func:`~optisample.optimize.reduce.dedupe.required_duration_s`); ``tail_skip_s`` closes it
    before the release. ``min_periods`` and ``min_loop_s`` together set the shortest loop
    that may be stored, which every candidate clears, alongside the analysis window the quality gates read a
    candidate over (:func:`~optisample.dsp.loop.shortest_loop_frames`) -- so however short a floor is asked
    for, the loop offered is one every gate measured.

    ``max_estimation_s`` caps the stretch the period is read off, which keeps the estimate on the part of
    the note that holds one pitch.
    """

    detune_semitones: Annotated[float, Field(gt=0.0)]
    min_correlation: float
    min_periods: Annotated[int, Field(ge=1)]
    min_loop_s: Annotated[float, Field(gt=0.0)]
    max_attack_s: Annotated[float, Field(ge=0.0)]
    tail_skip_s: Annotated[float, Field(ge=0.0)]
    max_estimation_s: Annotated[float, Field(gt=0.0)]


class PhaseConfig(ConfigModel):
    """How a round the frontier names is landed on the waveform's own phase.

    A reach is read off timbre frames, each spanning several periods of the note, so both of its bounds
    arrive knowing the round they want and not the sample it starts on. Landing them is what leaves a wrap
    the crossfade has motion to blend across (:func:`~optisample.dsp.loop.crossfade_loop`), and it matters
    the more the material a sample holds is a carrier: a carrier states timbre alone, so what a wrap
    carries over is phase.

    ``snap_periods`` is how far either side of the named start an ascending zero crossing is looked for,
    in periods of the pitch the recording was played at, which lands the start mid-slope.
    ``match_periods`` is how far either side of the whole count of periods the end is searched, matching
    the phase the start is approached on (:func:`~optisample.dsp.loop._matched_end`). Half a period reaches
    every phase there is; asking for more lets the end move by whole periods as well, which trades the
    length the geometry laid out for a closer match.
    """

    snap_periods: Annotated[float, Field(gt=0.0)]
    match_periods: Annotated[float, Field(gt=0.0)]


class FeatureConfig(ConfigModel):
    """How a recording is read as a series of timbre frames, which is where its own onset is found.

    Each frame carries a log-mel spectrum with that frame's own mean level taken out
    (:func:`~optisample.dsp.similarity.frame_series`), so two moments of a recording compare on the sound
    they hold and a note ringing its way down reads the same shape however far it has decayed.

    ``window_periods`` states the stretch one frame spans as periods of the pitch the recording was played
    at, so a deep note is read over a window that resolves its own partials while a high note is read over
    a proportionally shorter one; ``min_window_s`` floors that stretch, which keeps the highest notes'
    frames long enough to carry a spectrum, and ``hop_share`` is the step between frames as a share of the
    window. ``bands`` is how many mel bands one shape is read over and ``dynamic_range_db`` the range each
    frame states its shape across, under that frame's own loudest band, which is what has a frame late in
    a decay read as fully as the attack was.

    ``change_span_s`` is the stretch the shape's travel is measured across
    (:func:`~optisample.dsp.similarity.change_rate`), which is what leaves the movement of the material in
    the reading and divides the wobble of two overlapping windows out of it; it is also how long the rate
    has to hold before the material counts as settled, so one reading dipping on its own noise is read as
    the noise it is. ``settle_db_per_s`` is that rate -- the threshold saying where a note's onset ends and
    the stretch a loop may be taken from begins. Set it just above the rate a note holds while it sustains,
    so what it excludes is the transient.
    """

    window_periods: Annotated[int, Field(ge=1)]
    min_window_s: Annotated[float, Field(gt=0.0)]
    hop_share: Annotated[float, Field(gt=0.0, le=1.0)]
    bands: Annotated[int, Field(ge=1)]
    dynamic_range_db: Annotated[float, Field(gt=0.0)]
    change_span_s: Annotated[float, Field(gt=0.0)]
    settle_db_per_s: Annotated[float, Field(gt=0.0)]


class FrontierConfig(ConfigModel):
    """Which rounds a recording is offered, read off how closely its own sound wraps onto itself.

    Every close a round may reach is priced at what the best placement ending there gives up
    (:func:`~optisample.dsp.loopability.loop_frontier`): the step across its wrap plus the movement it
    plays in place of. Storing a round keeps everything up to that close, so the readings make a
    cost-per-byte curve whose lower convex hull is the frontier a budget prices a loop's length along.

    ``max_reach_s`` is how far past the frame the material settles at a round is looked for, which sets the
    longest round offered and the work one recording's search does. ``max_wrap_distance_db`` is the step a
    wrap holds under for its round to be offered at all, which is what says a recording is loopable:
    material whose sound keeps moving offers no round and is stored over the span it plays. ``max_offers``
    holds how many points of the frontier the sweep is asked to price, spread along the bytes they store.
    """

    max_reach_s: Annotated[float, Field(gt=0.0)]
    max_wrap_distance_db: Annotated[float, Field(gt=0.0)]
    max_offers: Annotated[int, Field(ge=1)]


class EnvelopeConfig(ConfigModel):
    """How the level a recording holds is read off it, which is the curve levelling and the decay ramp read.

    The weighting spans two periods of the pitch the recording was played at
    (:func:`~optisample.dsp.envelope.power_kernel`), so the power ripple every tone from half of that pitch
    upward carries averages out and what the reading is left holding is the level. Reading each note over
    its own period is what has a note two octaves up followed as closely as the note below it.

    ``lowest_hz`` and ``highest_hz`` bound the frequency that weighting is formed at
    (:func:`~optisample.dsp.envelope.reading_frequency`). ``lowest_hz`` caps how long the weighting runs, so
    the deepest notes are read over a stretch a loop region has room for; ``highest_hz`` floors it, so the
    beating of two partials a few hertz apart stays in the carrier, where it is heard as timbre. Raising
    ``highest_hz`` sharpens what the curve tracks at an onset and widens the band the curve itself occupies,
    so it takes proportionally more points to hold.

    ``floor_db`` places the quietest level the reading states, that far under the recording's own peak. It
    holds the curve strictly positive, leaves a silent stretch at the level it was recorded at, and scales
    with the material, so a recording captured hot and the same recording captured quiet read as one curve
    shifted and one carrier.
    """

    lowest_hz: Annotated[float, Field(gt=0.0)]
    highest_hz: Annotated[float, Field(gt=0.0)]
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
    stayed in phase and of material whose partials have drifted apart alike. ``crossovers_hz`` is where that
    reading is taken band by band and ``crossover_octaves`` how far each band climbs into the next. A
    struck string rings on partials spaced a little wider than whole multiples of its pitch, so one round of
    a loop returns the fundamental to the phase it left while the partials above it arrive where their own
    spacing puts them: reading the blend per band weights each of them by how alike that band measures, so
    every band comes through the wrap at the level it had. Bands the blend is too short to read apart are
    weighed together (:func:`~optisample.dsp.spectral.band_masks`), which is the reading that stretch
    supports; listing no crossover at all weighs the whole spectrum as one band.
    """

    fade_share: Annotated[float, Field(ge=0.0, le=1.0)]
    min_fade_s: Annotated[float, Field(ge=0.0)]
    crossovers_hz: tuple[Annotated[float, Field(gt=0.0)], ...]
    crossover_octaves: Annotated[float, Field(gt=0.0)]


class QualityConfig(ConfigModel):
    """What a loop must measure to be worth storing, which is how far the frontier may be pushed for bytes.

    ``max_seam_step`` bounds the wrap in units of the loop region's own typical frame-to-frame motion, so
    1.0 admits a wrap as smooth as the waveform already moves and a larger value admits a step a listener
    starts to hear once per round. ``max_spectral_distance_db`` bounds the log-spectral distance between the
    loop region and the stretch it plays in place of, so a loop holds a timbre the material still has.

    Together they bound how aggressive a loop may be: every candidate clearing both is offered as an
    operating point, so tightening either leaves the frontier with the later, longer, more faithful loops
    alone and loosening it lets the cheap end of the frontier back on.
    """

    max_seam_step: Annotated[float, Field(gt=0.0)]
    max_spectral_distance_db: Annotated[float, Field(gt=0.0)]


class LoopConfig(StageConfig):
    """How one recording is looped: where the loop sits and lands, the level it is held at, its wrap and its gates."""

    geometry: GeometryConfig
    phase: PhaseConfig
    features: FeatureConfig
    frontier: FrontierConfig
    envelope: EnvelopeConfig
    seam: SeamConfig
    quality: QualityConfig
