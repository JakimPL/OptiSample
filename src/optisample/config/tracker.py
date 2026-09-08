from enum import StrEnum, unique

from trackmod import Compliance

from optisample.config.base import ConfigModel


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

    ``compliance`` decides which of a format's three ceilings the module is graded against, in the terms
    :class:`~trackmod.limits.compliance.Compliance` states them: ``canonical`` keeps within what the
    tracker the format names allowed in its own editor, ``extended`` within what the players descended
    from it read, and ``structural`` within what the record layout physically holds. Writing at
    ``canonical`` is what makes a module the original tracker opens, and it is the tightest of the three
    -- Impulse Tracker numbers ninety-nine instruments and ninety-nine samples there, where the layout
    itself has room for two hundred and fifty-five.
    """

    format: TrackerFormat
    compliance: Compliance
    it: ITTrackerConfig
    xm: XMTrackerConfig
