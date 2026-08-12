from typing import Annotated, Self

from pydantic import Field, model_validator

from optisample.config.base import ConfigModel


class SubsonicConfig(ConfigModel):
    """The depth every recording arrives without: the band under hearing, stated as two points of a curve.

    ``cutoff_hz`` is where the roll-off takes hold, set under the lowest note a keyboard sounds so every
    partial the material carries stands in the band that is passed. ``rejection_hz`` and ``rejection_db``
    name how far down the curve stands at a given depth, which is what settles how steeply it falls
    (:func:`~optisample.dsp.subsonic.subsonic_order`). Naming the shape as a pair of points keeps the
    response as flat as one reaching that depth can be, so what a recording holds above the cutoff arrives
    at the level it was captured at.
    """

    cutoff_hz: Annotated[float, Field(gt=0.0)]
    rejection_hz: Annotated[float, Field(gt=0.0)]
    rejection_db: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def _rejection_under_cutoff(self) -> Self:
        """Hold the rejection point under the cutoff, which is the side of the curve it measures.

        Raises:
            ValueError: when ``rejection_hz`` stands at or above ``cutoff_hz``, where the curve is already
                passing what it was asked to reject.
        """
        if self.rejection_hz >= self.cutoff_hz:
            raise ValueError(f"rejection_hz {self.rejection_hz} must stand under cutoff_hz {self.cutoff_hz}")

        return self
