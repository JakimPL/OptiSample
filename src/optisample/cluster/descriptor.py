from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct

from optisample.cluster.harmonics import harmonic_profile
from optisample.cluster.pooling import AnchorStack, Reached, anchor_stack, frames_spanning, pooled_rows, window_mean
from optisample.config.cluster import DescriptorConfig, FrequencyBasis
from optisample.config.loop import FeatureConfig
from optisample.config.spectral import StftParams
from optisample.dsp.levels import decay_trend, level_readings
from optisample.dsp.piecewise import fit_piecewise
from optisample.dsp.similarity import FrameSeries, change_rate, frame_series, settling_frame, shape_travel
from optisample.dsp.trajectory import reading_window_s

Signal = NDArray[np.float64]

_FIRST_COEFFICIENT: Final = 1  # the coefficient a reading opens at, the constant one reading zero past a mean
_TRAVEL_LAG: Final = 1  # frames a step of the walk spans, which is the finest the series states movement over
_STILL: Final = 0.0  # the movement material too short to read a step across is stated as
_STRAIGHT: Final = 0.0  # the decline material too short to draw a line through is stated as
_ONSET_FRAMES: Final = 1  # frames an attack is read over at the fewest, so a note settling at once still reads


@dataclass(frozen=True)
class MovementReading:
    """How much a recording's timbre moves while it sounds, read two ways over two stretches.

    ``change_db_per_s`` is the rate the shape travels at through the middle of the settled span, which
    states how steadily the note holds one sound once its onset is behind it. ``travel_db_per_s`` is the
    whole distance the shape walks from the first frame to the last, per second of the recording, so it
    carries the rush of the attack as well as the sustain. A struck string and a bowed one part company on
    the pair: one races and settles, the other keeps moving throughout.
    """

    change_db_per_s: float
    travel_db_per_s: float

    @property
    def values(self) -> Signal:
        """The pair as one block of the descriptor, in the order stated above."""
        return np.asarray([self.change_db_per_s, self.travel_db_per_s], dtype=np.float64)


@dataclass(frozen=True)
class EnvelopeReading:
    """The contour a recording plays, stated so that gain and length leave it untouched.

    Every term is a rate, a normalized time or a depth, and the level curve behind them is read against the
    recording's own peak, so a take captured hot and the same take captured quiet read one contour.
    ``attack_s`` is how long the note took to reach its peak and ``settle_s`` where its onset ends;
    ``decay_db_per_s`` is the straight line its settled level falls along; ``curvature_db`` is how far the
    fitted curve departs from that line at its widest, which is one number saying how far from a straight
    decline the note runs. ``anchor_log_s`` is how long the note took to reach each fall depth, in
    log-seconds, so the contour stands on the same grid the timbre blocks are pooled over: at each depth
    one says what the note sounded like there and the other how long it took to arrive.
    """

    attack_s: float
    settle_s: float
    decay_db_per_s: float
    curvature_db: float
    anchor_log_s: Signal

    @property
    def scalars(self) -> Signal:
        """The four readings the contour states ahead of its anchor times, in the order listed above."""
        return np.asarray([self.attack_s, self.settle_s, self.decay_db_per_s, self.curvature_db], dtype=np.float64)

    @property
    def values(self) -> Signal:
        """The whole contour as one block of the descriptor, the four scalars ahead of the anchor times."""
        return np.asarray(np.concatenate((self.scalars, self.anchor_log_s)), dtype=np.float64)

    def at(self, depths: Reached) -> Signal:
        """The contour read at the depths ``depths`` marks, the four scalars ahead of those anchor times.

        A corpus reads its recordings at the depths enough of them arrive at, and this states the contour
        over the same ones, so the timing of a fall and the sound held there stand on one grid.
        """
        return np.asarray(np.concatenate((self.scalars, self.anchor_log_s[depths])), dtype=np.float64)


@dataclass(frozen=True)
class SampleDescriptor:
    """One recording as the blocks a sample space is built from, each read past the level it was played at.

    ``onset`` is the shape held through the attack and ``sustain`` the shape held at each depth of the
    note's own decline, one row per fall depth; both are level-free, and on the relative basis pitch-free
    as well, so two takes of one sound at different velocities and in different registers sit close.
    ``movement`` says how far that shape travels while the note rings and ``envelope`` carries the contour
    as a supplementary distinction. ``reached`` states which depths the recording truly arrived at, so a
    caller reading a whole corpus builds its matrix on the part of a decline they share.
    """

    onset: Signal
    sustain: Signal
    movement: MovementReading
    envelope: EnvelopeReading
    reached: Reached

    @property
    def columns(self) -> int:
        """How many readings one row of shape holds -- partials, mel bands, or the coefficients kept of them."""
        return int(self.onset.size)

    @property
    def depths(self) -> int:
        """How many fall depths the recording was read at."""
        return int(self.sustain.shape[0])


def _shape_rows(
    signal: Signal,
    sample_rate: int,
    *,
    series: FrameSeries,
    root_hz: float,
    config: DescriptorConfig,
) -> Signal:
    """``signal`` read frame by frame on the frequency axis ``config`` names: ``(frames, columns)``.

    Both bases read one frame over the stretch the series reads it over, so the two answer on one grid and
    an anchor found on the level curve indexes either.
    """
    match config.frequency_basis:
        case FrequencyBasis.RELATIVE:
            params = StftParams(n_fft=series.window_length, hop_length=series.hop_length)
            return harmonic_profile(signal, sample_rate, params=params, root_hz=root_hz, config=config)

        case FrequencyBasis.ABSOLUTE:
            return series.shape


def _cepstra(rows: Signal, config: DescriptorConfig) -> Signal:
    """``rows`` truncated to the coefficients holding their smooth spectral envelope.

    The cosine transform orders a spectrum from its broadest tilt to its finest ripple, so keeping the
    leading coefficients keeps the envelope a listener hears as timbre and leaves the bin-by-bin detail
    that varies take to take. The constant coefficient reads zero on a reading already stated past its own
    mean, so the run opens at the one after it. Keeping none reads the bands as they stand.
    """
    if config.cepstral_coefficients == 0:
        return rows

    coefficients = np.asarray(dct(rows, type=2, norm="ortho", axis=1), dtype=np.float64)
    kept = slice(_FIRST_COEFFICIENT, _FIRST_COEFFICIENT + config.cepstral_coefficients)
    return np.asarray(coefficients[:, kept], dtype=np.float64)


def _movement(series: FrameSeries, features: FeatureConfig, settled: int) -> MovementReading:
    """How far ``series`` travels in shape, read across its settled span and across the whole of it.

    Material still on the move at its last frame leaves no settled span behind it, and the rate is then
    read across the whole curve -- which states what that material held throughout, and is what puts a
    sound still travelling beside one that arrived somewhere.
    """
    rate = change_rate(series, features)
    sustained = rate[min(settled, rate.size) :]
    holding = sustained if sustained.size else rate
    walked = shape_travel(series.shape, _TRAVEL_LAG) if series.frames > _TRAVEL_LAG else np.zeros(0, dtype=np.float64)
    seconds = series.frames * series.hop_s
    return MovementReading(
        change_db_per_s=float(np.median(holding)) if holding.size else _STILL,
        travel_db_per_s=float(np.sum(walked)) / seconds if seconds > 0.0 else _STILL,
    )


def _decline(signal: Signal, sample_rate: int, config: DescriptorConfig) -> tuple[float, float]:
    """The rate ``signal``'s level falls at and how far its fitted curve departs from that straight fall.

    Both are read in decibels, which is the domain a ringing note falls straight in and the domain a gain
    moves as one constant -- so the pair states the shape of a decline whatever level it was captured at.
    Material too short for a line to be drawn through reads a straight, level fall.
    """
    trend = decay_trend(signal, sample_rate)
    if trend is None:
        return _STRAIGHT, _STRAIGHT

    readings = level_readings(signal, sample_rate, window_s=reading_window_s(signal.size, sample_rate))
    curve = fit_piecewise(readings, nodes=config.envelope_nodes)
    line = trend.mean_db + trend.slope_db * (readings.seconds - trend.mean_s)
    return trend.slope_db, float(np.max(np.abs(curve.at(readings.seconds) - line)))


def _envelope(
    signal: Signal,
    sample_rate: int,
    *,
    stack: AnchorStack,
    settle: int,
    config: DescriptorConfig,
) -> EnvelopeReading:
    """The contour ``signal`` plays, with its decline read from the frame ``settle`` its onset ends at.

    Anchor times are stated in log-seconds against the finest moment the series grid resolves, one hop, so
    the early depths a note races through and the late ones it drifts to are read on one scale.
    """
    decay_db_per_s, curvature_db = _decline(np.asarray(signal[settle:], dtype=np.float64), sample_rate, config)
    return EnvelopeReading(
        attack_s=stack.attack_s,
        settle_s=settle / sample_rate,
        decay_db_per_s=decay_db_per_s,
        curvature_db=curvature_db,
        anchor_log_s=np.log10(np.maximum(stack.seconds, stack.hop_s)),
    )


def describe(
    signal: Signal,
    sample_rate: int,
    *,
    root_hz: float,
    features: FeatureConfig,
    config: DescriptorConfig,
) -> SampleDescriptor:
    """``signal``, played at ``root_hz``, read into the blocks a sample space is built from.

    The whole reading is anchored to the note's own decline and stated past its own level, so what comes
    back describes a sound rather than a take of one: two recordings of the same note at different
    velocities, at different lengths, or captured at different gains answer nearly the same blocks, while
    two notes that genuinely sound apart answer blocks that stand apart.
    """
    series = frame_series(signal, sample_rate, features, root_hz)
    stack = anchor_stack(series.level_db, config.anchor_depths_db, series.hop_s)
    rows = _cepstra(_shape_rows(signal, sample_rate, series=series, root_hz=root_hz, config=config), config)
    settle = settling_frame(series, features)
    settled = max(series.frame_after(settle), _ONSET_FRAMES)
    return SampleDescriptor(
        onset=window_mean(rows, start=0, end=settled),
        sustain=pooled_rows(rows, stack, frames_spanning(config.anchor_span_s, series.hop_s)),
        movement=_movement(series, features, settled),
        envelope=_envelope(signal, sample_rate, stack=stack, settle=settle, config=config),
        reached=stack.reached,
    )
