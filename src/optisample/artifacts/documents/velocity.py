from __future__ import annotations

from optisample.artifacts.serialize import Frozen
from optisample.optimize.velocity_map import VelocityVolumeMap


class AnchorRecord(Frozen):
    """One measured velocity anchor of the loudness-matched velocity->volume map."""

    velocity: int
    loudness_lufs: float
    volume: int


class VelocityMapDocument(Frozen):
    """The velocity->volume map: its anchors and the full 0..127 lookup table."""

    reference_volume: int
    anchors: list[AnchorRecord]
    volumes: list[int]


def velocity_map_document(velocity_map: VelocityVolumeMap) -> VelocityMapDocument:
    """The measured map as a document, which the plan states and every bank layer carries a copy of."""
    reference_volume = max((anchor.volume for anchor in velocity_map.anchors), default=0)
    return VelocityMapDocument(
        reference_volume=reference_volume,
        anchors=[
            AnchorRecord(velocity=anchor.velocity, loudness_lufs=anchor.loudness_lufs, volume=anchor.volume)
            for anchor in velocity_map.anchors
        ],
        volumes=list(velocity_map.volumes),
    )
