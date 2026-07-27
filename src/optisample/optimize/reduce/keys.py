from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor
from typing import Protocol

from optisample.config.reduce import DedupeConfig, DedupeKey
from optisample.music import pitch_label

CcBuckets = tuple[tuple[int, int], ...]


class Identified(Protocol):
    """What a recorded sample and a played note both state about which sound they are.

    Reading the identity through a protocol lets one extractor serve both sides, so the recordings that
    survive deduplication and the material that decides what they must cover are keyed the same way.
    """

    @property
    def pitch(self) -> int: ...

    @property
    def velocity(self) -> int: ...

    @property
    def cc_averages(self) -> Mapping[int, float]: ...


def cc_buckets(cc_averages: Mapping[int, float], quantum: float) -> CcBuckets:
    """Controller averages as a discrete identity: each value floored onto a ``quantum``-wide bucket.

    A time-weighted average is continuous, so flooring it onto a fixed grid is what gives every note
    played under the same controller position one identity. Entries are ordered by controller number so
    the tuple hashes and compares deterministically.
    """
    return tuple(sorted((controller, floor(value / quantum)) for controller, value in cc_averages.items()))


@dataclass(frozen=True, order=True)
class SampleKey:
    """The identity of one recording: the pitch and velocity it was played at, and its controller variant.

    ``pitch`` and ``velocity`` are always the recording's own. A survivor kept under a pitch-only key
    still reports the velocity it was recorded at, which is what the velocity map and the
    nearest-velocity lookup read. ``cc`` carries a variant only when the configured key distinguishes
    controller positions.
    """

    pitch: int
    velocity: int
    cc: CcBuckets = ()

    @property
    def label(self) -> str:
        """Stable display name: the note, the velocity, and any controller variant that sets it apart."""
        base = f"{pitch_label(self.pitch)}_v{self.velocity:03d}"
        if not self.cc:
            return base

        variant = "_".join(f"cc{controller}-{bucket}" for controller, bucket in self.cc)
        return f"{base}_{variant}"


@dataclass(frozen=True, order=True)
class DedupeGroup:
    """The projection of a :class:`SampleKey` deciding which recordings compete for one storage slot.

    Recordings sharing a group are renditions of the same thing under the configured key, so exactly one
    of them survives. ``velocity`` is unset under a pitch-only key and ``cc`` under any key that reads no
    controller variant, which is what collapses those axes.
    """

    pitch: int
    velocity: int | None
    cc: CcBuckets


def sample_key(identified: Identified, config: DedupeConfig) -> SampleKey:
    """The full identity of ``identified``, carrying a controller variant when the key reads one."""
    cc = cc_buckets(identified.cc_averages, config.cc_quantum) if config.key.includes_cc else ()
    return SampleKey(pitch=identified.pitch, velocity=identified.velocity, cc=cc)


def dedupe_group(key: SampleKey, dedupe_key: DedupeKey) -> DedupeGroup:
    """Project ``key`` onto the axes ``dedupe_key`` treats as identity."""
    return DedupeGroup(
        pitch=key.pitch,
        velocity=key.velocity if dedupe_key.includes_velocity else None,
        cc=key.cc if dedupe_key.includes_cc else (),
    )


def nearest_key(available: Sequence[SampleKey], velocity: int) -> SampleKey:
    """Recorded key whose velocity is closest to ``velocity``.

    This is the many-to-one lookup that routes a played note to the recording it is scored against: a
    whole span of velocities resolves to one key, which is what lets those notes be scored once. Ties
    favour the louder recording, then the lowest CC bucket, so a pitch with several timbral variants at
    one velocity still resolves to the same reference on every run.
    """
    return min(available, key=lambda key: (abs(key.velocity - velocity), -key.velocity, key.cc))
