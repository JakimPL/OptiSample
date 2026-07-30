from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.codec import EncodeConfig
from optisample.dsp.decay import LinearDecay
from optisample.dsp.loop import Loop

NO_LOOP: Final = None  # the settled loop of a clip whose material no loop stands in for


@dataclass(frozen=True)
class SettledLoop:
    """The loop a clip is stored around and the decline it goes on making past it.

    Settled before any encoding runs, on the recording at the rate it was analysed at, so every copy of the
    clip reaches the same region of the material: :func:`~optisample.dsp.loop.loop_at_rate` scales the
    bounds onto whatever rate a copy is stored at. ``decay`` is the ramp a note held past the stored span
    falls on, which is how the level a loop repeats comes down outside the PCM.
    """

    loop: Loop
    decay: LinearDecay | None


@dataclass(frozen=True)
class EncodingParams:
    """One encoding configuration for a stored sample: the per-sample decision variables.

    ``compress`` asks for the dynamics stage ahead of the quantizer, which narrows the crest factor so
    more of the depth's grid carries material. ``looped`` stores the clip around the loop settled for it,
    keeping the attack plus one loop region; left off, the clip keeps ``trim_s`` of the recording as it was
    played. Those two are what the sweep prices against each other.
    """

    target_rate: int
    depth_bits: int
    trim_s: float | None = None
    dither: bool = True
    noise_shaping: bool = False
    looped: bool = False
    compress: bool = False


@dataclass(frozen=True)
class EncodeContext:
    """The fixed surroundings of one :func:`~optisample.dsp.surrogate.encode.encode` call.

    Bundles the root pitch stamped on the stored sample, the encode config (the seam blend, compression and
    the stored headroom), the loop the clip was settled around, and the dither RNG. ``settled`` is
    :data:`NO_LOOP` for a clip the loop stage found nothing to loop, which stores the trimmed span whatever
    the params ask for.
    """

    root_pitch: int
    config: EncodeConfig
    settled: SettledLoop | None = NO_LOOP
    rng: np.random.Generator | None = None
