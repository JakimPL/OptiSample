from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

Readings = NDArray[np.float64]
Frames = NDArray[np.intp]
Reached = NDArray[np.bool_]

_MIN_SPAN: Final = 1  # frames one reading covers, which is what has every anchor a reading of its own


@dataclass(frozen=True)
class AnchorStack:
    """Where a recording stands at each fall depth below its own peak, and which of those depths it reached.

    Anchoring to the note's own decline is what leaves its length out of every reading taken here: anchor
    *k* is the moment the recording had fallen *k* decibels from its loudest, so a two-second take and an
    eight-second take of the same note are read at the same points of the same sound. A note falling into
    its own floor before the deepest anchor holds the frame it last reached, which reads the shape it had
    when it got there, and ``reached`` states which depths it truly arrived at so a caller reading a whole
    corpus can build on the part of a decline its recordings share.

    ``frames`` and ``peak_frame`` count in frames of the series the level curve was read on, and ``hop_s``
    is what turns either into seconds on the played clock.
    """

    frames: Frames
    reached: Reached
    peak_frame: int
    hop_s: float

    @property
    def depths(self) -> int:
        """How many fall depths the recording was read at."""
        return int(self.frames.size)

    @property
    def seconds(self) -> Readings:
        """The moment each anchor sits at, in seconds from the recording's first frame."""
        return np.asarray(self.frames * self.hop_s, dtype=np.float64)

    @property
    def attack_s(self) -> float:
        """How long the recording took to reach its own peak, which is the length of its attack."""
        return self.peak_frame * self.hop_s


def frames_spanning(seconds: float, hop_s: float) -> int:
    """How many frames of a series stepping ``hop_s`` a stretch of ``seconds`` covers, one of them at least."""
    return max(_MIN_SPAN, round(seconds / hop_s))


def window_mean(readings: Readings, *, start: int, end: int) -> Readings:
    """The mean of ``readings`` over the rows ``[start, end)``, held inside the rows there are.

    The bounds are clipped to the material, so a window reaching past either end reads what is there. A
    window landing on no rows at all reads zeros, which is the shape a stretch holding nothing states.
    """
    rows = np.asarray(readings, dtype=np.float64)
    low, high = max(0, min(start, rows.shape[0])), min(rows.shape[0], max(end, 0))
    if high <= low:
        return np.zeros(rows.shape[1], dtype=np.float64)

    return np.asarray(rows[low:high].mean(axis=0), dtype=np.float64)


def anchor_stack(level_db: Readings, depths_db: Sequence[float], hop_s: float) -> AnchorStack:
    """The frame ``level_db`` has fallen each depth of ``depths_db`` below its own peak at.

    The search opens at the peak and runs forward, so each anchor is a moment on the note's decline rather
    than on its way up, and a depth is reached the first frame the level stands that far under the peak.
    Depths ascend, so a recording short of one is read at the deepest frame it did reach and every deeper
    anchor holds that same frame -- which is what has a truncated take and the whole take of one note
    agree at every depth both of them arrive at.

    Raises:
        ValueError: when ``level_db`` holds no readings, which is material with no decline to anchor to.
    """
    readings = np.asarray(level_db, dtype=np.float64)
    if readings.size == 0:
        raise ValueError("a fall depth is read off at least one level reading")

    peak = int(np.argmax(readings))
    fallen = float(readings[peak]) - readings[peak:]
    depths = np.asarray(depths_db, dtype=np.float64)
    below = fallen[np.newaxis, :] >= depths[:, np.newaxis]  # (depths, frames past the peak)
    reached = np.asarray(below.any(axis=1), dtype=np.bool_)
    arrived = np.where(reached, peak + np.argmax(below, axis=1), peak)
    return AnchorStack(
        frames=np.asarray(np.maximum.accumulate(arrived), dtype=np.intp),
        reached=reached,
        peak_frame=peak,
        hop_s=hop_s,
    )


def pooled_rows(readings: Readings, stack: AnchorStack, span: int) -> Readings:
    """The mean of ``readings`` over the ``span`` frames centred on each anchor: ``(depths, columns)``.

    Averaging a stretch around an anchor rather than reading the single frame at it states what the note
    held while it passed that depth, so the reading carries the material's own sound and leaves the wobble
    two overlapping windows make out of it.
    """
    reach = span // 2
    return np.asarray(
        np.stack([window_mean(readings, start=frame - reach, end=frame - reach + span) for frame in stack.frames]),
        dtype=np.float64,
    )
