from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.dynamic_axis import DynamicAxisConfig
from optisample.config.subsonic import SubsonicConfig


class SubsetConfig(ConfigModel):
    """What a slice admits: how long a note has to sound to be worth carrying through the pipeline.

    ``min_duration_s`` is the span a note sounds for at the least, read from its onset through the end of
    its release, which is the stretch every stage after the slice decodes. A note sounding for less than a
    loop may run (:attr:`~optisample.config.loop.GeometryConfig.min_loop_s`, opening past
    :attr:`~optisample.config.loop.GeometryConfig.max_attack_s`) offers a loop stage nothing to settle and
    a group nothing to stand behind, so holding it out at the way in leaves every later stage reading
    material it can work with. A floor of zero admits whatever the source recorded.
    """

    min_duration_s: Annotated[float, Field(ge=0.0)]


class IntakeConfig(ConfigModel):
    """What the way into a run does to its source: what it admits, the band it passes, and how it spreads.

    All three are settled at the first stage and nowhere later, so they travel as one bundle: a note
    sounding for less than ``min_duration_s`` stays behind, every take that lands is written past
    ``subsonic``, and the share that is kept is spread along ``dynamic_axis`` -- the reading this
    instrument's dynamics travel on, so a slice spans the dynamics its material was played across rather
    than an axis that happens to be recorded beside them. That leaves every stage afterwards reading a
    dataset already holding material it can work with, at the content a listener has.
    """

    min_duration_s: float
    subsonic: SubsonicConfig
    dynamic_axis: DynamicAxisConfig
