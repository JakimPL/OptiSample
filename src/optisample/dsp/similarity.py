from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from optisample.config.loop import FeatureConfig
from optisample.config.spectral import MelParams
from optisample.dsp.spectral import melspectrogram

Signal = NDArray[np.float64]

_POWER_DB: Final = 10.0  # decibels per decade of power
_QUIET_POWER: Final = 1e-20  # the mel power silence reads as, which leaves the shape of silence defined
_MEL_FMIN: Final = 0.0  # the mel bands open at zero, so the deepest partial a note carries lands in one
_MIN_HOP: Final = 1  # frames one step of the series covers, which is what has each frame its own reading
_MIN_SPAN: Final = 1  # frames a movement is read across, which is what states a rate


@dataclass(frozen=True)
class FrameSeries:
    """A recording read as the shape of its sound frame by frame, beside the level each frame sat at.

    ``shape`` holds one log-mel spectrum per frame in decibels with that frame's own mean taken out, so two
    moments of a recording compare on the sound they hold rather than on how loud they were: a note ringing
    its way down states the shape it holds at the end of a decay as clearly as at its start, which is the
    reading a loop is looked for under. ``level_db`` is what was taken out, the level curve the shape was
    read against, so the two together carry the whole of what each frame measured.

    ``window_length`` spans whole periods of the pitch the recording was played at, which resolves the
    partials of a deep note and a high one alike, and ``hop_length`` is the step from one frame to the next.
    """

    shape: Signal  # (frames, bands)
    level_db: Signal  # (frames,)
    hop_length: int
    window_length: int
    sample_rate: int

    @property
    def frames(self) -> int:
        """How many frames the recording was read as."""
        return int(self.shape.shape[0])

    @property
    def hop_s(self) -> float:
        """Seconds from one frame to the next, which is the step a rate read off the series spans."""
        return self.hop_length / self.sample_rate

    def frame_start(self, index: int) -> int:
        """The recording frame the series frame ``index`` opens at."""
        return index * self.hop_length

    def frame_index(self, frame: int) -> int:
        """The series frame covering the recording frame ``frame``, held inside the series.

        Rounding down keeps the reading inside a bound stated in recording frames, which is what a stretch
        the series is read over needs of its far end.
        """
        return min(max(frame // self.hop_length, 0), self.frames)

    def frame_after(self, frame: int) -> int:
        """The first series frame beginning at or past the recording frame ``frame``, held inside the series.

        Rounding up holds a reading at or past a bound stated in recording frames, which is what the near
        end of such a stretch needs: a window opening where the material settled starts no earlier.
        """
        return min(-(-frame // self.hop_length), self.frames)


def window_frames(sample_rate: int, config: FeatureConfig, root_hz: float) -> int:
    """The stretch one frame of the series is read over, in frames at ``sample_rate``.

    Spanning ``window_periods`` periods of the pitch a recording was played at places that note's partials
    the same number of bins apart whatever the note, so the bottom two octaves are resolved as clearly as
    the top -- a window fixed in seconds reads a deep note's partials as leakage between neighbouring mel
    bands. ``min_window_s`` floors the stretch, which keeps a high note's frame long enough to carry a
    spectrum, and the count is held even so the transform runs on a whole number of bins.
    """
    wanted = max(
        round(config.window_periods * sample_rate / root_hz),
        round(config.min_window_s * sample_rate),
    )
    return wanted + wanted % 2


def frame_series(signal: Signal, sample_rate: int, config: FeatureConfig, root_hz: float) -> FrameSeries:
    """``signal`` read as a series of timbre frames, each one stated past the level it was played at.

    Every frame is floored ``dynamic_range_db`` under its own loudest band before the log is taken, so each
    states its shape over a range of its own and a frame late in a decay is read as fully as the attack was.
    Scaling the recording moves ``level_db`` by that gain and leaves ``shape`` as it stands, which is what
    makes the shape a reading of the sound and the level a reading of the dynamics.
    """
    window = window_frames(sample_rate, config, root_hz)
    hop = max(_MIN_HOP, round(window * config.hop_share))
    params = MelParams(n_fft=window, hop_length=hop, n_mels=config.bands, fmin=_MEL_FMIN, fmax=None)
    mel = melspectrogram(signal, sample_rate, params)
    loudest = mel.max(axis=1, keepdims=True) * 10.0 ** (-config.dynamic_range_db / _POWER_DB)
    log_mel = _POWER_DB * np.log10(np.maximum(mel, np.maximum(loudest, _QUIET_POWER)))
    level_db = np.asarray(log_mel.mean(axis=1), dtype=np.float64)
    return FrameSeries(
        shape=np.asarray(log_mel - level_db[:, None], dtype=np.float64),
        level_db=level_db,
        hop_length=hop,
        window_length=window,
        sample_rate=sample_rate,
    )


def span_frames(series: FrameSeries, config: FeatureConfig) -> int:
    """Frames of ``series`` that ``change_span_s`` covers, which is the stretch one reading is held over."""
    return max(_MIN_SPAN, round(config.change_span_s / series.hop_s))


def shape_travel(shape: Signal, lag: int) -> Signal:
    """How far each frame's shape stands from the frame ``lag`` later, in decibels averaged over the bands.

    The mean absolute difference band by band is the ``logmel_l1`` distance the composite reads timbre
    with, so a travel measured here and a fidelity measured there state the same kind of number. Reading it
    at every frame at once is what turns one lag into a curve along the recording, which both the change
    rate and the loop frontier are read off.
    """
    return np.asarray(np.mean(np.abs(shape[lag:] - shape[:-lag]), axis=1), dtype=np.float64)


def change_rate(series: FrameSeries, config: FeatureConfig) -> Signal:
    """How fast the shape moves at each frame of ``series``, in decibels per second.

    The reading at one frame is how far the shape has travelled ``change_span_s`` later, stated per second
    of that span. Reading across a span leaves the movement of the material in the curve and divides the
    wobble of one reading out of it: a drift accumulates over the whole span while the jitter two
    overlapping windows make stays the size of one reading. Measured on real piano material that holds the
    rate a note sustains at within 7.3-14.3 dB/s across the keyboard, against 10.2-39.3 for the same
    contrast reached by smoothing and then reading neighbouring frames.

    Stating a rate rather than the depth of one step is what lets a single threshold answer for every
    recording, whatever hop its own pitch set. A series shorter than the span answers an empty curve,
    which is material with no travel to read.
    """
    lag = span_frames(series, config)
    if series.frames <= lag:
        return np.zeros(0, dtype=np.float64)

    return np.asarray(shape_travel(series.shape, lag) / (lag * series.hop_s), dtype=np.float64)


def _held_under(under: NDArray[np.bool_], span: int) -> int:
    """The first reading of ``under`` opening a run of ``span`` that all hold, or the count of readings.

    Asking the material to hold the rate for a whole span, rather than to touch it once, is what has a
    reading dipping under on its own noise counted as the noise it is. A series with fewer readings than
    the span is asked to hold every one of them, which is all the evidence it carries.
    """
    if under.size == 0:
        return 0

    runs = sliding_window_view(under, min(span, under.size)).all(axis=1)
    settled = np.nonzero(runs)[0]
    return int(settled[0]) if settled.size else int(under.size)


def settling_frame(series: FrameSeries, config: FeatureConfig) -> int:
    """The recording frame ``series`` settles at: where its shape slows to ``settle_db_per_s`` and holds.

    A struck note's spectrum races through its onset and slows as the partials that decay fastest die
    away, so the frame the change rate settles under the threshold and stays there is where the material
    begins holding a sound one loop can stand in for. Reading it off the recording states the onset each
    note makes for itself, which on real piano material runs from the first frame out to half a second.

    Material that never moves faster than the threshold answers frame 0, which is a note holding one sound
    from the moment it begins. Material whose shape keeps moving answers the frame its last reading opens
    at, which takes the whole recording as onset and leaves the geometry to say how much of that it will
    wait through.
    """
    rate = change_rate(series, config)
    return series.frame_start(_held_under(rate <= config.settle_db_per_s, span_frames(series, config)))
