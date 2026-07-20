"""Fidelity measurement: the composite scorer, its component metrics, and the byte-size model.

Importing this package registers the four built-in metric factories (``mrstft``, ``logmel_l1``,
``mcd``, ``spectral_shape``): :mod:`optisample.metrics.spectral` and :mod:`optisample.metrics.timbre`
call :func:`~optisample.metrics.base.register_metric` at import time, so :func:`build_composite` and
:func:`~optisample.metrics.base.build_metric` can resolve them by name. Diagnostics and the low-level
registry helpers are imported from their owning module, not through this package.
"""

from optisample.metrics.base import MetricContext, available_metrics, build_metric, register_metric, unregister_metric
from optisample.metrics.composite import CompositeFidelity, build_composite, evaluate
from optisample.metrics.preprocess import integrated_loudness, loudness_normalize, prepare
from optisample.metrics.size import SampleSize, bytes_to_kib, kib_to_bytes
from optisample.metrics.spectral import LogMelL1, MultiResolutionStft
from optisample.metrics.timbre import MelCepstralDistortion, SpectralShape

__all__ = [
    "MetricContext",
    "available_metrics",
    "build_metric",
    "register_metric",
    "unregister_metric",
    "CompositeFidelity",
    "build_composite",
    "evaluate",
    "integrated_loudness",
    "loudness_normalize",
    "prepare",
    "SampleSize",
    "bytes_to_kib",
    "kib_to_bytes",
    "LogMelL1",
    "MultiResolutionStft",
    "MelCepstralDistortion",
    "SpectralShape",
]
