import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from trackmod import Compliance, Sample
from trackmod.module.size import SizeReport

from optisample.config.tracker import TrackerFormat
from optisample.io.tracker.target import ExportTarget
from optisample.io.tracker.written import BEYOND_STORAGE, WrittenModule, written_module
from tests.io.conftest import ProbeModule

_RATE = 22_050
_SIZE = SizeReport(patterns=120, pcm=9000, headers=800, largest_pattern=90)


@pytest.mark.parametrize("tracker_format", tuple(TrackerFormat))
def test_a_written_module_states_its_size_its_reach_and_its_writer(
    tracker_format: TrackerFormat,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
    tone: Callable[..., np.ndarray],
) -> None:
    """A module holding nothing unusual opens in the tracker its format names, and says who wrote it."""
    module = probe_module([Sample(name="tone", pcm=tone(), rate=_RATE)], retarget(tracker_format))
    stated = written_module(module)

    assert stated.size.total == len(module.to_bytes())
    assert stated.reach is Compliance.CANONICAL
    assert stated.exceeded == ()
    assert stated.provenance  # both formats name their writer in the header


def test_a_module_no_layout_holds_reaches_none_of_the_levels() -> None:
    stated = WrittenModule(size=_SIZE, reach=None, exceeded=(), provenance=None)
    assert stated.reach_label == BEYOND_STORAGE


def test_the_reach_a_module_states_is_named_by_the_level_itself() -> None:
    stated = WrittenModule(size=_SIZE, reach=Compliance.EXTENDED, exceeded=(), provenance="TrackMod")
    assert stated.reach_label == Compliance.EXTENDED.value
    assert dataclasses.replace(stated, reach=Compliance.CANONICAL).reach_label == Compliance.CANONICAL.value
