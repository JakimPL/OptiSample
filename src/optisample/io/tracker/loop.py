from trackmod import Loop as TrackerLoop

from optisample.dsp.loop import Loop


def stored_loop(loop: Loop | None) -> TrackerLoop | None:
    """``loop`` as the half-open frame range a player wraps on, where a region was settled to wrap.

    Both sides count frames from the first frame of the stored waveform, so the region a stage settled is
    the region a tracker repeats. A recording carrying no region is sounded end to end.
    """
    if loop is None:
        return None

    return TrackerLoop(begin=loop.start, end=loop.end)
