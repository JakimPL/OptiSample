from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.cluster import DescriptorConfig
from optisample.config.spectral import StftParams
from optisample.dsp.levels import db_to_gain, gain_to_db
from optisample.dsp.spectral import stft_magnitude

Signal = NDArray[np.float64]
Bands = tuple[tuple[int, int], ...]

_LOUDEST_FLOOR: Final = 1e-30  # the loudest magnitude a silent frame is read against, leaving its ratio defined
_FIRST_HARMONIC: Final = 1  # the partial a reading opens at, which is the pitch the note was played at


def harmonic_bands(sample_rate: int, params: StftParams, root_hz: float, config: DescriptorConfig) -> Bands:
    """The transform bins each partial of ``root_hz`` is looked for in, ``config.harmonics`` of them upward.

    Each band reaches ``harmonic_band_share`` of the pitch either side of that partial's own multiple, so a
    share of 0.5 tiles the spectrum into harmonic-wide bands and every partial is read in a band of its
    own. Reading a band rather than one bin is what follows a struck string: its partials ring a little
    wider apart than whole multiples of the pitch, and the instrument itself is tuned where its own
    stretch put it, so the loudest bin near a multiple is the partial that multiple names.

    A band reaching past the transform's own range comes back empty, which reads as the quiet a recording
    holds above its Nyquist.
    """
    resolution = sample_rate / params.n_fft
    bins = params.n_fft // 2 + 1
    reach = config.harmonic_band_share * root_hz
    return tuple(
        (
            max(0, int(np.ceil((harmonic * root_hz - reach) / resolution))),
            min(bins, int(np.floor((harmonic * root_hz + reach) / resolution)) + 1),
        )
        for harmonic in range(_FIRST_HARMONIC, config.harmonics + _FIRST_HARMONIC)
    )


def harmonic_peaks(magnitude: Signal, bands: Bands) -> Signal:
    """The loudest bin each band of ``magnitude`` holds, frame by frame: ``(frames, harmonics)``.

    A band the transform reaches no bins for reads zero, which is the level of material there is no room
    to carry.
    """
    frames = magnitude.shape[0]
    return np.asarray(
        np.stack(
            [
                magnitude[:, low:high].max(axis=1) if high > low else np.zeros(frames, dtype=np.float64)
                for low, high in bands
            ],
            axis=1,
        ),
        dtype=np.float64,
    )


def harmonic_profile(
    signal: Signal,
    sample_rate: int,
    *,
    params: StftParams,
    root_hz: float,
    config: DescriptorConfig,
) -> Signal:
    """``signal`` read as the balance its own partials hold at each frame: ``(frames, harmonics)``.

    Every frame states its partials against its own loudest one and then past their own mean, which leaves
    the reading a statement of balance alone: scaling the recording scales every bin by the same amount, so
    the ratios the reading is built from stand exactly where they stood. Indexing by partial number rather
    than by frequency leaves the pitch out of it too, so a note and the note an octave up sharing a balance
    read alike -- which is what lets a group here mean a sound rather than a register.

    ``config.harmonic_range_db`` floors each frame that far under its own loudest partial, so a frame late
    in a decay states its balance across a range of its own and is read as fully as the attack was.
    """
    magnitude = stft_magnitude(signal, params)
    peaks = harmonic_peaks(magnitude, harmonic_bands(sample_rate, params, root_hz, config))
    loudest = np.maximum(peaks.max(axis=1, keepdims=True), _LOUDEST_FLOOR)
    decibels = gain_to_db(np.maximum(peaks / loudest, db_to_gain(-config.harmonic_range_db)))
    return np.asarray(decibels - decibels.mean(axis=1, keepdims=True), dtype=np.float64)
