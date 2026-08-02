from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct

from optisample.config.spectral import MelParams, StftParams

Signal = NDArray[np.float64]

_LOG_FLOOR: Final = 1e-10
_MEL_SCALE: Final = 2595.0  # HTK mel scale factor
_MEL_BREAK_HZ: Final = 700.0  # HTK mel break frequency (Hz)
_MIN_CROSSOVER_BINS: Final = 3  # bins a crossover spans its climb over for the split to separate anything


def frame(signal: Signal, frame_length: int, hop_length: int) -> Signal:
    """Slice ``signal`` into overlapping rectangular frames ``(n_frames, frame_length)`` (right-padded)."""
    data = np.asarray(signal, dtype=np.float64)
    if data.size < frame_length:
        data = np.pad(data, (0, frame_length - data.size))
    n_frames = 1 + (data.size - frame_length) // hop_length
    offsets = hop_length * np.arange(n_frames)[:, None]
    indices = np.arange(frame_length)[None, :] + offsets
    return np.asarray(np.take(data, indices), dtype=np.float64)


def stft_magnitude(signal: Signal, params: StftParams) -> Signal:
    """Magnitude STFT with a Hann window: ``(n_frames, n_fft // 2 + 1)``."""
    window = np.hanning(params.n_fft)
    frames = frame(signal, params.n_fft, params.hop_length) * window
    spectrum = np.fft.rfft(frames, n=params.n_fft, axis=1)
    return np.abs(spectrum).astype(np.float64)


def _hz_to_mel(hertz: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return np.asarray(
        _MEL_SCALE * np.log10(1.0 + np.asarray(hertz, dtype=np.float64) / _MEL_BREAK_HZ),
        dtype=np.float64,
    )


def _mel_to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        _MEL_BREAK_HZ * (10.0 ** (mel / _MEL_SCALE) - 1.0),
        dtype=np.float64,
    )


def mel_filterbank(sample_rate: int, params: MelParams) -> Signal:
    """Triangular mel filterbank ``(n_mels, n_fft // 2 + 1)`` on the FFT frequency grid."""
    fmax = params.fmax if params.fmax is not None else sample_rate / 2.0
    fft_freqs = np.fft.rfftfreq(params.n_fft, 1.0 / sample_rate)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(params.fmin), _hz_to_mel(fmax), params.n_mels + 2))
    filters = np.zeros((params.n_mels, fft_freqs.size), dtype=np.float64)
    for band in range(1, params.n_mels + 1):
        left, center, right = float(edges[band - 1]), float(edges[band]), float(edges[band + 1])
        rising = (fft_freqs - left) / max(center - left, _LOG_FLOOR)
        falling = (right - fft_freqs) / max(right - center, _LOG_FLOOR)
        filters[band - 1] = np.clip(np.minimum(rising, falling), 0.0, None)

    return filters


def melspectrogram(
    signal: Signal,
    sample_rate: int,
    params: MelParams,
) -> Signal:
    """Mel power spectrogram ``(n_frames, n_mels)``."""
    power = stft_magnitude(signal, params.stft()) ** 2
    filters = mel_filterbank(sample_rate, params)
    return np.asarray(power @ filters.T, dtype=np.float64)


def mfcc(
    signal: Signal,
    sample_rate: int,
    params: MelParams,
    n_mfcc: int,
    top_db: float,
) -> Signal:
    """Mel-frequency cepstral coefficients ``(n_frames, n_mfcc)`` (DCT-II of log-mel energies).

    The log-mel is floored ``top_db`` dB below its peak so near-silent bins (and any noise
    filling them) do not dominate the cepstrum. Flooring is done in the natural-log domain,
    keeping mel-cepstral-distortion's ``10 / ln 10`` dB constant valid downstream.
    """
    log_mel = np.log(np.maximum(melspectrogram(signal, sample_rate, params), _LOG_FLOOR))
    floor = float(np.max(log_mel)) - top_db * float(np.log(10.0)) / 10.0
    coeffs = np.asarray(dct(np.maximum(log_mel, floor), type=2, norm="ortho", axis=1), dtype=np.float64)
    return np.asarray(coeffs[:, :n_mfcc], dtype=np.float64)


def _magnitude_and_freqs(
    signal: Signal,
    sample_rate: int,
    params: StftParams,
) -> tuple[Signal, Signal]:
    magnitude = stft_magnitude(signal, params)
    freqs = np.fft.rfftfreq(params.n_fft, 1.0 / sample_rate)
    return magnitude, np.asarray(freqs, dtype=np.float64)


def spectral_centroid(
    signal: Signal,
    sample_rate: int,
    params: StftParams,
) -> float:
    """Energy-weighted mean frequency (Hz), averaged over frames — a brightness proxy."""
    magnitude, freqs = _magnitude_and_freqs(signal, sample_rate, params)
    total = np.sum(magnitude, axis=1)
    centroid = np.where(total > 0.0, (magnitude @ freqs) / np.maximum(total, _LOG_FLOOR), 0.0)
    return float(np.mean(centroid))


def spectral_rolloff(
    signal: Signal,
    sample_rate: int,
    params: StftParams,
    roll_percent: float,
) -> float:
    """Frequency (Hz) below which ``roll_percent`` of the energy lies, averaged over frames."""
    magnitude, freqs = _magnitude_and_freqs(signal, sample_rate, params)
    cumulative = np.cumsum(magnitude, axis=1)
    total = cumulative[:, -1]
    threshold = roll_percent * total[:, None]
    index = np.argmax(cumulative >= threshold, axis=1)
    rolloff = np.where(total > 0.0, freqs[index], 0.0)
    return float(np.mean(rolloff))


def spectral_flatness(signal: Signal, params: StftParams) -> float:
    """Geometric/arithmetic mean-power ratio, averaged over frames (1.0 ≈ noise, ~0 ≈ tonal)."""
    power = stft_magnitude(signal, params) ** 2 + _LOG_FLOOR
    geometric = np.exp(np.mean(np.log(power), axis=1))
    arithmetic = np.mean(power, axis=1)
    return float(np.mean(geometric / np.maximum(arithmetic, _LOG_FLOOR)))


def spectral_flux(signal: Signal, params: StftParams) -> Signal:
    """Per-frame L2 magnitude change ``(n_frames - 1,)`` — how fast the spectrum evolves."""
    magnitude = stft_magnitude(signal, params)
    if magnitude.shape[0] < 2:
        return np.zeros(1, dtype=np.float64)

    return np.sqrt(np.sum(np.diff(magnitude, axis=0) ** 2, axis=1)).astype(np.float64)


def band_energy(
    signal: Signal,
    sample_rate: int,
    f_low: float,
    f_high: float,
) -> float:
    """Total spectral energy in ``[f_low, f_high)`` (whole-signal FFT, Parseval-proportional)."""
    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def content_edge_hz(
    signal: Signal,
    sample_rate: int,
    floor_db: float,
    band_hz: float,
) -> float:
    """Highest frequency still carrying power within ``floor_db`` of the signal's loudest band.

    This is where a recording's spectrum genuinely ends, which is the band a stored copy has to hold.
    Reading it against the peak keeps it a question of dynamic range: content this far down is masked by
    what sits above it whatever the absolute level. Power is averaged into ``band_hz``-wide bands first,
    so the answer tracks the spectral envelope and one loud bin stands only for its own band; a band
    wider than the spectrum collapses to one, reading the whole signal at once. Silence reports 0 Hz.

    A share-of-energy reading answers a different question, and a misleading one here: a harmonic tone
    keeps almost all its energy in the first few partials, so any fraction short of the whole reports a
    frequency far below where the tone still sounds.
    """
    power = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64))) ** 2
    if float(np.max(power)) <= 0.0:
        return 0.0

    width = min(max(1, round(band_hz * signal.size / sample_rate)), power.size)
    banded_bins = (power.size // width) * width
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    banded = power[:banded_bins].reshape(-1, width).mean(axis=1)
    centers = freqs[:banded_bins].reshape(-1, width).mean(axis=1)
    audible = np.nonzero(banded >= float(np.max(banded)) * 10.0 ** (-floor_db / 10.0))[0]
    return float(centers[audible[-1]])


def bandlimit(
    signal: Signal,
    sample_rate: int,
    f_low: float,
    f_high: float,
) -> Signal:
    """Zero every frequency outside ``[f_low, f_high)`` and return the time-domain signal."""
    length = signal.size
    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    freqs = np.fft.rfftfreq(length, 1.0 / sample_rate)
    spectrum = spectrum * ((freqs >= f_low) & (freqs < f_high))
    return np.fft.irfft(spectrum, n=length).astype(np.float64)


def _resolved_crossovers(
    crossovers_hz: Sequence[float],
    resolution_hz: float,
    nyquist_hz: float,
    transition_octaves: float,
) -> list[float]:
    """The crossovers a spectrum read at ``resolution_hz`` per bin separates, lowest first.

    A crossover climbs from one band to the next across ``transition_octaves``, so the span it occupies
    grows with the frequency it sits at while the resolution of the spectrum stays fixed. Three conditions
    keep the ones that separate material: the climb spans at least ``_MIN_CROSSOVER_BINS`` bins, so the two
    bands it divides each hold a reading of their own; it starts an octave span above the crossover below
    it, so each climb finishes before the next begins and every band stays positive; and it finishes under
    Nyquist, so the band above it holds spectrum. A crossover the spectrum reads too coarsely to place
    leaves its two bands joined as one, which is the reading that stretch supports.
    """
    span = 2.0 ** (transition_octaves / 2.0)
    width = span - 1.0 / span
    kept: list[float] = []
    for crossover in sorted(set(crossovers_hz)):
        spans_enough_bins = crossover * width >= _MIN_CROSSOVER_BINS * resolution_hz
        clears_the_one_below = not kept or crossover >= kept[-1] * 2.0**transition_octaves
        finishes_under_nyquist = crossover * span < nyquist_hz
        if spans_enough_bins and clears_the_one_below and finishes_under_nyquist:
            kept.append(crossover)

    return kept


def _rising_step(freqs: Signal, crossover_hz: float, transition_octaves: float) -> Signal:
    """A raised-cosine climb from 0 to 1 spanning ``transition_octaves`` centred on ``crossover_hz``."""
    progress = np.log2(np.maximum(freqs, _LOG_FLOOR) / crossover_hz) / transition_octaves + 0.5
    return np.asarray(np.sin(0.5 * np.pi * np.clip(progress, 0.0, 1.0)) ** 2, dtype=np.float64)


def band_masks(
    sample_rate: int,
    length: int,
    crossovers_hz: Sequence[float],
    transition_octaves: float,
) -> Signal:
    """Masks over the rfft grid of a ``length``-frame signal that sum to one at every bin.

    The masks are the gaps between a ladder of raised-cosine climbs, one per crossover the spectrum
    resolves (:func:`_resolved_crossovers`), so they sum to one exactly and a signal split by them adds
    back up to itself. Splitting the amplitude this way hands material sitting on a crossover to the two
    bands in the proportion each mask names, so the two carry it back whole however they are then weighted.

    Returns one mask per band, lowest band first: ``(kept crossovers + 1, length // 2 + 1)``.
    """
    freqs = np.asarray(np.fft.rfftfreq(length, 1.0 / sample_rate), dtype=np.float64)
    kept = _resolved_crossovers(crossovers_hz, sample_rate / length, sample_rate / 2.0, transition_octaves)
    ladder = (
        [np.ones(freqs.size, dtype=np.float64)]
        + [_rising_step(freqs, crossover, transition_octaves) for crossover in kept]
        + [np.zeros(freqs.size, dtype=np.float64)]
    )
    return np.stack([above - below for above, below in zip(ladder, ladder[1:])])


def split_bands(
    signal: Signal,
    sample_rate: int,
    crossovers_hz: Sequence[float],
    transition_octaves: float,
) -> Signal:
    """``signal`` separated into the frequency bands ``crossovers_hz`` names, ``(n_bands, signal.size)``.

    The bands sum back to ``signal`` frame by frame (:func:`band_masks`), so a caller weighting them apart
    is holding the whole of the material and nothing else.
    """
    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    masks = band_masks(sample_rate, signal.size, crossovers_hz, transition_octaves)
    return np.fft.irfft(masks * spectrum, n=signal.size).astype(np.float64)
