"""Timebase conversion between seconds and sample frames.

A single home for the seconds-to-frames rounding used wherever a duration is turned into a slice
length, so trimming a signal to a note's duration reads the same everywhere.
"""

from __future__ import annotations


def seconds_to_frames(duration_s: float, sample_rate: int) -> int:
    """Frame count spanning ``duration_s`` at ``sample_rate`` (rounded to nearest, floored at zero)."""
    return max(0, int(round(duration_s * sample_rate)))
