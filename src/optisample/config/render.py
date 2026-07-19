"""Rendering / playback configuration: openmpt123 render settings and IT global playback.

``RenderConfig`` replaces ``io.render.RenderSettings``' value fields (the ``interpolation`` name is
mapped to openmpt123 ``--filter`` taps by a free function in ``io/render.py``). ``PlaybackConfig`` holds
the IT global playback the exporter/calibrator used to hardcode as ``ITPlayback()`` -- ``speed``/``tempo``
set the row duration, so they affect rendered note timing. ``Interpolation`` is redeclared here to keep
the config package a dependency leaf.
"""

from __future__ import annotations

from typing import Literal

from optisample.config.base import ConfigModel

Interpolation = Literal["none", "linear", "cubic", "sinc"]


class RenderConfig(ConfigModel):
    """How openmpt123 renders an ``.IT`` file: output rate, interpolation filter, output gain (dB)."""

    sample_rate: int
    interpolation: Interpolation
    gain_db: float


class PlaybackConfig(ConfigModel):
    """IT global playback: ticks/row via ``speed``+``tempo``, plus global and mix volume."""

    speed: int
    tempo: int
    global_volume: int
    mix_volume: int
