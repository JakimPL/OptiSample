from enum import StrEnum, unique
from typing import Annotated, Final

from pydantic import Field

from optisample.config.base import ConfigModel

MAX_CONTROLLER: Final = 127  # MIDI numbers its controllers over one data byte, as it does its velocities
MOD_WHEEL: Final = 1  # the controller a sustaining library gives its dynamics to, by long convention


@unique
class DynamicAxis(StrEnum):
    """Which reading of a note says how loudly it was played.

    A keyboard states a note's force as its velocity, and a library built around a sustaining instrument
    states it as a controller the player holds -- the mod wheel, most often -- leaving velocity to shape
    the attack or to pick an articulation. Both readings arrive on every note, so which of them carries
    the dynamics is a fact about the library that recorded them.
    """

    VELOCITY = "velocity"
    CONTROLLER = "controller"


class DynamicAxisConfig(ConfigModel):
    """Where an instrument's dynamics are read from, which is what every stage keys its recordings on.

    ``axis`` names the reading, and ``controller`` the controller number that reading takes where the axis
    is a controller. The two travel together so a run reading velocities still states the controller it
    would read, and moving one instrument onto the wheel is one word.

    A tracker pattern cell carries one dynamic column, so whichever reading an instrument's dynamics
    travel on is the one written there: the reading is taken once, at the way in
    (:func:`~optisample.dynamic_axis.on_axis`), and every stage afterwards works on an axis that varies
    with how loudly the material was played.
    """

    axis: DynamicAxis
    controller: Annotated[int, Field(ge=0, le=MAX_CONTROLLER)]


AS_WRITTEN: Final = DynamicAxisConfig(axis=DynamicAxis.VELOCITY, controller=MOD_WHEEL)
"""The axis a dataset this project wrote is read back on.

Every stage writes the dynamic it was keyed on as each note's velocity, so a stage reading another
stage's dataset finds the dynamics already on the velocity axis and takes them from there. The
controller stands unread, and names the wheel because that is what a run reading a controller reads.
"""
