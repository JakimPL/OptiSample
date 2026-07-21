"""Fast numpy "surrogate" codec: encode a recording into an IT-style stored sample and render notes
back from it, standing in for the real tracker in the optimizer's inner loop.

Encoding composes the byte-reducing transforms -- peak-normalize (store hot), resample to a lower
rate, trim to only the duration the material needs, and requantize to 8/16-bit with dither/noise
shaping -- into a :class:`~optisample.dsp.surrogate.sample.StoredSample`. Rendering reads that sample
back at a target pitch (a resample by ``2**(semitones / 12)``, the way a tracker repitches), scales by
the note volume (``0..64``, linear per IT), and fits the note to a requested duration.

Looping lets a short stored sample sustain a long note: with ``params.loop`` set, storage is trimmed
to the attack plus a looped region and :func:`render` repeats that region to fill the duration;
without a loop, a note held longer than the sample is zero-padded (it simply ends). The surrogate is
the objective the rate-distortion search minimizes.

This package exposes the codec's public API; the implementation is split across
:mod:`~optisample.dsp.surrogate.params` (the decision variables and encode context),
:mod:`~optisample.dsp.surrogate.sample` (the stored sample), :mod:`~optisample.dsp.surrogate.encode`,
and :mod:`~optisample.dsp.surrogate.render`.
"""

from optisample.dsp.surrogate.encode import encode
from optisample.dsp.surrogate.params import EncodeContext, EncodingParams
from optisample.dsp.surrogate.render import render
from optisample.dsp.surrogate.sample import MAX_VOLUME, Signal, StoredSample

__all__ = [
    "MAX_VOLUME",
    "Signal",
    "EncodeContext",
    "EncodingParams",
    "StoredSample",
    "encode",
    "render",
]
