import pytest
from pydantic import ValidationError
from trackmod import BitDepth

from optisample.config.optimize import SweepConfig


def _sweep(depth: int) -> SweepConfig:
    return SweepConfig(
        rates=(22_050, 11_025),
        rate_headroom=1,
        depth=depth,  # type: ignore[arg-type]
        dither=True,
        noise_shaping=False,
        compress=True,
        carrier=True,
    )


@pytest.mark.parametrize("depth", tuple(BitDepth))
def test_every_depth_a_tracker_format_stores_is_a_depth_a_run_may_sweep_at(depth: BitDepth) -> None:
    assert _sweep(depth).depth is depth


@pytest.mark.parametrize("depth", [0, 12, 24, 32])
def test_a_depth_no_tracker_format_stores_is_refused_where_it_enters(depth: int) -> None:
    """A depth is read from config and carried as the tracker's own vocabulary, so an unstorable one
    is refused at the boundary rather than reaching the encoder that would have to answer for it."""
    with pytest.raises(ValidationError):
        _sweep(depth)
