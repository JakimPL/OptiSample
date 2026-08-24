from dataclasses import dataclass, replace

import numpy as np

from optisample.config.spectral import MelParams
from optisample.dsp.onset import ATTACK_READING, onset_window
from optisample.dsp.spectral import melspectrogram
from optisample.metrics.base import MetricContext, Signal, register_metric
from optisample.metrics.spectral import floored_log, min_frames


@dataclass(frozen=True)
class OnsetShape:
    """Log-mel L1 over the attack alone, which is the stretch a mean over a whole note has no say about.

    Every other term of the composite averages over the frames of the note it scores, so what a listener
    uses to recognise a sound -- the few milliseconds it starts with -- reaches the objective in proportion
    to how long it lasts. A hit whose attack runs ten milliseconds inside a note lasting a second reaches
    it as a hundredth of the score, whatever it does to the sound. This term reads the same distance the
    composite already trusts over the attack by itself, so an attack a stored copy smeared, softened or
    played into noise states its own cost rather than being averaged into the note behind it.

    Each signal is windowed from its own onset (:func:`~optisample.dsp.onset.onset_window`), so a delay the
    codec introduces leaves the reading alone and what the two windows differ by is the shape of the attack.

    ``pre_s`` and ``span_s`` say how much of the attack is read; every other threshold is the convention
    :data:`~optisample.dsp.onset.ATTACK_READING` states.
    """

    params: MelParams
    dynamic_range_db: float
    pre_s: float
    span_s: float
    name: str = "onset"

    def distance(
        self,
        reference: Signal,
        candidate: Signal,
        context: MetricContext,
    ) -> float:
        """``dynamic_range_db`` floors mel *power*, hence the ``/10`` conversion (amplitude would use ``/20``)."""
        reading = replace(ATTACK_READING, pre_s=self.pre_s, span_s=self.span_s)
        rel_floor = 10.0 ** (-self.dynamic_range_db / 10.0)
        ref_mel = melspectrogram(
            onset_window(reference, context.sample_rate, reading), context.sample_rate, self.params
        )
        cand_mel = melspectrogram(
            onset_window(candidate, context.sample_rate, reading), context.sample_rate, self.params
        )
        frames = min_frames(ref_mel, cand_mel)
        ref_mel, cand_mel = ref_mel[:frames], cand_mel[:frames]
        peak = float(np.max(ref_mel)) if ref_mel.size else 0.0
        return float(np.mean(np.abs(floored_log(ref_mel, peak, rel_floor) - floored_log(cand_mel, peak, rel_floor))))


register_metric(
    "onset",
    lambda config: OnsetShape(
        params=config.onset.mel,
        dynamic_range_db=config.preprocess.dynamic_range_db,
        pre_s=config.onset.pre_s,
        span_s=config.onset.span_s,
    ),
)
