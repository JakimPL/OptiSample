from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.stage import StageConfig
from optisample.config.tracker import TrackerConfig


class EnvelopeConfig(ConfigModel):
    """The volume envelope written beside an instrument: how long a released note takes to fall silent.

    The decline itself is read off the recording, so the one thing left to state is what happens once a
    note is let go, which the material never states because a recording is a note held to its end.
    """

    release_s: Annotated[float, Field(gt=0.0)]


class ExportConfig(StageConfig):
    """What leaves the run: the module written, the clock it plays on, and how it is rendered to audio.

    ``tracker`` states the format and the levels stored in it, ``playback`` the speed and tempo both
    formats share, ``envelope`` how a released note is let go, and ``render`` what openmpt123 is asked for
    when the written module is turned back into audio.
    """

    tracker: TrackerConfig
    render: RenderConfig
    playback: PlaybackConfig
    envelope: EnvelopeConfig
