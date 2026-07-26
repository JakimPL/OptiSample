from enum import StrEnum, unique

from optisample.config.base import ConfigModel
from trackmod.limits.compliance import Compliance


@unique
class TrackerFormat(StrEnum):
    """A tracker file format a module can be written as."""

    IT = "it"
    XM = "xm"


class ITTrackerConfig(ConfigModel):
    """The song-wide levels Impulse Tracker stores beyond the clock every tracker format shares."""

    global_volume: int
    mix_volume: int


class XMTrackerConfig(ConfigModel):
    """What FastTracker 2 records about the module beyond the song itself.

    ``tracker`` is the name the file credits as its writer, which a tracker shows when the module is
    opened.
    """

    tracker: str


class TrackerConfig(ConfigModel):
    """Which format a module is written as, how strictly it holds to it, and its per-format settings.

    ``compliance`` decides which bounds the module is graded against: ``canonical`` keeps within what the
    original tracker itself reads, ``extended`` allows everything the record layout can hold.
    """

    format: TrackerFormat
    compliance: Compliance
    it: ITTrackerConfig
    xm: XMTrackerConfig
