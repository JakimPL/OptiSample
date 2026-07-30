from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.codec import EncodeConfig
from optisample.dsp.decay import LinearDecay
from optisample.dsp.loop import Loop

NO_LOOP: Final = None  # the loop of a stored sample that plays out rather than wrapping
UNLOOPED: Final = None  # the loop an encoding names when it stores the played span instead of a region


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


SettledLoops = tuple[SettledLoop, ...]  # the loops one clip offers, cheapest stored span first
NO_LOOPS: Final[SettledLoops] = ()  # the offer of a clip whose material no loop stands in for


@dataclass(frozen=True)
class EncodingParams:
    """One encoding configuration for a stored sample: the per-sample decision variables.

    ``compress`` asks for the dynamics stage ahead of the quantizer, which narrows the crest factor so
    more of the depth's grid carries material. ``loop_index`` names one of the loops the stage settled for
    the clip (:attr:`EncodeContext.settled`), storing the attack plus that region; left :data:`UNLOOPED`,
    the clip keeps ``trim_s`` of the recording as it was played. A longer loop carries more of the
    material's own movement for more bytes, so naming which one is what turns loop length into an axis the
    sweep prices alongside the trimmed span.
    """

    target_rate: int
    depth_bits: int
    trim_s: float | None = None
    dither: bool = True
    noise_shaping: bool = False
    loop_index: int | None = UNLOOPED
    compress: bool = False


@dataclass(frozen=True)
class EncodeContext:
    """The fixed surroundings of one :func:`~optisample.dsp.surrogate.encode.encode` call.

    Bundles the root pitch stamped on the stored sample, the encode config (the seam blend, compression and
    the stored headroom), the loops the clip was settled around, and the dither RNG. ``settled`` holds them
    in the order :attr:`EncodingParams.loop_index` counts, and is :data:`NO_LOOPS` for a clip the loop stage
    found nothing to loop, which stores the trimmed span whatever the params ask for.
    """

    root_pitch: int
    config: EncodeConfig
    settled: SettledLoops = NO_LOOPS
    rng: np.random.Generator | None = None
