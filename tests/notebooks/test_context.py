from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from notebooks.utils import context, degrade
from optisample.metrics import evaluate

SR = 44_100


def test_every_part_is_read_off_the_config_the_context_holds() -> None:
    notebook = context.notebook_context()
    assert notebook.spectrogram.params == notebook.config.analysis.spectral.stft
    assert notebook.spectrogram.dynamic_range_db == notebook.config.analysis.metrics.preprocess.dynamic_range_db
    assert notebook.target.format == notebook.config.export.tracker.format
    assert notebook.storage is notebook.target.storage


def test_the_composite_scores_a_degraded_signal_above_an_untouched_one(
    tone: Callable[..., NDArray[np.float64]],
) -> None:
    notebook = context.notebook_context()
    signal = tone()
    quantized = degrade.quantize(signal, bits=4)
    assert evaluate(signal, signal, SR, notebook.composite).fidelity == pytest.approx(0.0, abs=1e-9)
    assert evaluate(signal, quantized, SR, notebook.composite).fidelity > 0.0
