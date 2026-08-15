from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.dynamics import HoldConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.stage import StageConfig
from optisample.config.tracker import TrackerConfig


class EnvelopeConfig(ConfigModel):
    """The volume envelope written beside an instrument: how long a released note takes to fall silent.

    The decline itself is read off the recording, so the one thing left to state is what happens once a
    note is let go, which the material never states because a recording is a note held to its end.
    """

    release_s: Annotated[float, Field(gt=0.0)]


class InstrumentsConfig(ConfigModel):
    """How a set of recordings is written as the standalone instruments a player loads on their own.

    One volume envelope serves a whole set, so it states the level they agree on and each recording keeps
    whatever it holds beyond that. What is left over stays in the waveform, where peak normalization reads
    it as the crest and the depth is spent on it. ``compression`` is the curve that remainder is held back
    along before any of it is stored (:func:`~optisample.dsp.dynamics.held_back`), which is what has a
    shallow grid spend itself on timbre.

    ``post_loop`` keeps what a recording goes on making past the region it wraps on, stored behind it. A
    file written to be loaded and edited is worth carrying it in: a player sounds the loop, so the tail
    costs a listener nothing until they delete the loop to reach it.
    """

    compression: HoldConfig
    post_loop: bool


class ExportConfig(StageConfig):
    """What leaves the run: the module written, the clock it plays on, and how it is rendered to audio.

    ``tracker`` states the format and the levels stored in it, ``playback`` the speed and tempo both
    formats share, ``envelope`` how a released note is let go, ``instruments`` how a set of recordings is
    written as standalone files, and ``render`` what openmpt123 is asked for when the written module is
    turned back into audio.
    """

    tracker: TrackerConfig
    render: RenderConfig
    playback: PlaybackConfig
    envelope: EnvelopeConfig
    instruments: InstrumentsConfig
