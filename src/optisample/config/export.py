from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.stage import StageConfig
from optisample.config.tracker import TrackerConfig


class ExportConfig(StageConfig):
    """What leaves the run: the module written, the clock it plays on, and how it is rendered to audio.

    ``tracker`` states the format and the levels stored in it, ``playback`` the speed and tempo both
    formats share, and ``render`` what openmpt123 is asked for when the written module is turned back
    into audio.
    """

    tracker: TrackerConfig
    render: RenderConfig
    playback: PlaybackConfig
