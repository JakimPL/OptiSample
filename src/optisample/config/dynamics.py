from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel


class HoldConfig(ConfigModel):
    """The curve a level is held back along once it rises past the threshold.

    ``threshold_db`` sits below whatever peak the level is read against, so material is shaped by how far
    it spreads under its own loudest moment rather than by how hot it happens to stand. Levels reaching
    past the threshold keep ``1 / ratio`` of their excess, arriving at that slope over a bend ``knee_db``
    wide.
    """

    threshold_db: float
    ratio: Annotated[float, Field(ge=1.0)]
    knee_db: Annotated[float, Field(ge=0.0)]


class LimitConfig(HoldConfig):
    """The hold curve, plus how quickly it acts and the ceiling it holds absolutely.

    The curve itself is :class:`HoldConfig`. ``attack_share`` and ``release_share`` state how long the
    reduction takes to arrive and how long it stays, each as a share of the reach the level detector already
    spans (:attr:`~optisample.dsp.envelope.LevelReading.reach`), so both follow the pitch the recording was
    played at rather than a span fixed for every note alike. The detector reads a frame from the material
    centred on it, which is the lookahead an attack shorter than that reach acts inside: a share under 1.0
    brings the reduction in ahead of the peak that asked for it, which is what makes the pass hold a
    transient rather than follow it. A release longer than the attack keeps the reduction through the decay
    behind a peak, so the level settles once instead of moving with every cycle. Shares of zero hold the
    reduction to exactly the moment the curve asks for it.

    ``ceiling_db`` is the absolute limit, stated over the same body the threshold is read against: whatever
    the curve leaves, the result stays under it. Because the level it acts on is already a smooth curve, the
    ceiling moves smoothly too, which is what lets it be a brick wall and still leave the waveform inside a
    cycle as it stands. A ceiling far above the threshold leaves the ratio alone to shape the material.
    """

    attack_share: Annotated[float, Field(ge=0.0)]
    release_share: Annotated[float, Field(ge=0.0)]
    ceiling_db: float


class DynamicsConfig(HoldConfig):
    """Soft-knee compression ahead of the quantizer: where the curve acts, how hard, and how quickly.

    The curve itself is :class:`HoldConfig`, read against the clip's own peak. ``rms_window_s`` is how long
    the level detector averages over and ``gain_smoothing_s`` how long the gain takes to follow it, which
    together keep the reduction tracking the material rather than each sample.
    """

    rms_window_s: Annotated[float, Field(gt=0.0)]
    gain_smoothing_s: Annotated[float, Field(gt=0.0)]
