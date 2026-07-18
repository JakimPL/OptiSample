"""Helpers backing the interactive notebooks.

Split so the marimo notebook can stay a thin presentation layer:

* :mod:`~notebooks.utils.loading` — find/generate a manifest and load samples.
* :mod:`~notebooks.utils.audioio` — turn a signal into WAV bytes for ``mo.audio``.
* :mod:`~notebooks.utils.degrade` — encoding-preview degradations (quantize/resample/lowpass/gain).
* :mod:`~notebooks.utils.views` — per-sample and per-instrument tables + budget roll-up + comparisons.
* :mod:`~notebooks.utils.viz` — matplotlib figures (waveform, spectrogram, metric breakdown).
"""
