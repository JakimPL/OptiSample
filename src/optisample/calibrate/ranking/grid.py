from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from trackmod import BitDepth

from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import EncodingParams
from optisample.optimize.operating_points import compresses, sweep_rates


@dataclass(frozen=True)
class RankingGrid:
    """How far a listening set widens the grid the sweep offers, along the axes the sweep settles first.

    One depth is read off a clip's own content before the sweep starts
    (:func:`~optisample.optimize.reduce.bandwidth.stored_format`), and the rungs the allocation chooses
    among reach as far up as
    :attr:`~optisample.config.optimize.SweepConfig.rate_headroom` opens them, so a run leaving that at
    zero offers encodings differing in how much of the recording they store and in nothing else. A metric
    judged on those alone is judged on one axis. ``depths`` are the depths a clip is also stored at, which
    is what puts a quantization step in the set; ``rate_steps`` is how many rungs of the sweep's own ladder
    below the settled rate it is also read at, which puts a bandwidth step there beside it.

    Each widened rung is stored both as the pipeline would store it -- the depth carrying the compression
    it earns (:func:`~optisample.optimize.operating_points.compresses`) -- and with that compression
    turned the other way. Offering both is what keeps a depth question about depth: the pipeline ties the
    two together, so a pair drawn from the pipeline's points alone would move the grid and the dynamics at
    once and a label on it would speak for neither.
    """

    depths: tuple[BitDepth, ...]
    rate_steps: int


def _rate_ladder(settled_rate: int, sweep: SweepConfig, sample_rate: int, steps: int) -> tuple[int, ...]:
    """``settled_rate`` and the ``steps`` rungs of the sweep's own ladder nearest below it."""
    below = [rate for rate in sweep_rates(sweep, sample_rate) if rate < settled_rate]
    return (settled_rate, *below[:steps])


def widened_encodings(
    swept: Sequence[EncodingParams],
    grid: RankingGrid,
    sweep: SweepConfig,
    *,
    sample_rate: int,
) -> tuple[EncodingParams, ...]:
    """Every encoding a listening set renders for one clip: the sweep's own, widened along rate and depth.

    The sweep's encodings lead in the order it offers them, so every operating point the allocation
    chooses among is in the set and a label about them speaks to the plan as it stands. Beside them the
    stored span the sweep leads with -- the played length, which every clip offers whatever its loops --
    is read at each rung of ``grid``, at both settings of the compressor the pipeline ties to a depth, so
    a listener meets the point the pipeline would store and the one holding the dynamics where they were.

    Halving the rate and halving the depth each halve the bytes, so widening both puts pairs of nearly
    equal size in the set: the trade a byte budget actually makes, offered to a listener as one question.
    """
    widened = list(swept)
    span = swept[0]
    for rate in _rate_ladder(span.target_rate, sweep, sample_rate, grid.rate_steps):
        for depth in grid.depths:
            for compress in dict.fromkeys((compresses(sweep, depth), span.compress)):
                member = replace(span, target_rate=rate, depth=depth, compress=compress)
                if member not in widened:
                    widened.append(member)

    return tuple(widened)
