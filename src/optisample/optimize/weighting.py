from optisample.dsp.level import mean_energy
from optisample.metrics.base import Signal


def energy_weight(reference: Signal, exponent: float) -> float:
    """How much a note's own energy scales what its distortion costs the objective.

    A reconstruction is scored on loudness-matched signals
    (:func:`~optisample.metrics.composite.evaluate`), so the fidelity it earns states a *relative* error:
    how wrong the note sounds measured against itself. What that error is worth in the mix is the note's
    own energy, read here off the stretch of recording the note is scored over and raised to
    ``exponent``.

    Full scale weighs 1.0, so the exponent alone says how steeply a quiet note counts for less: ``0``
    prices every note alike, ``0.5`` follows its amplitude, ``1`` its energy, and ``0.3`` the loudness an
    ear reports for that energy. A note 30 dB under full scale therefore weighs 1.0, 0.032, 0.001 and
    0.126 respectively.
    """
    return float(mean_energy(reference) ** exponent)
