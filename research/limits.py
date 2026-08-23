from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Final

from optisample.config.loader import load_config

from research.bench import BenchRecord, read

_NAME: Final = 18
_WIDTH: Final = 10
_MISSING: Final = "."
_CONCORDANT: Final = 1
_HALVED: Final = {"rate": "half rate", "depth": "half depth", "carrier": "half depth, carried"}


def _by_class(records: Sequence[BenchRecord]) -> dict[str, list[BenchRecord]]:
    """The takes of each acoustic class, which is the grain every claim is stated at."""
    gathered: dict[str, list[BenchRecord]] = {}
    for record in records:
        gathered.setdefault(record.material_class, []).append(record)

    return gathered


def _mean_by_label(records: Sequence[BenchRecord], axis: str) -> dict[str, float]:
    """What each setting of ``axis`` measured on average, over the takes of one class."""
    gathered: dict[str, list[float]] = {}
    for record in records:
        if record.axis == axis:
            gathered.setdefault(record.label, []).append(record.fidelity)

    return {label: fmean(values) for label, values in gathered.items()}


def ladder(records: Sequence[BenchRecord], axis: str) -> str:
    """One class per row, one setting of ``axis`` per column, holding what the composite measured."""
    classes = _by_class(records)
    scored = {name: _mean_by_label(takes, axis) for name, takes in classes.items()}
    labels = sorted({label for measured in scored.values() for label in measured}, key=float)
    lines = [f"=== {axis}", f"{'class':{_NAME}} " + " ".join(f"{label:>{_WIDTH}}" for label in labels)]
    for name in sorted(scored):
        cells = [
            f"{scored[name][label]:{_WIDTH}.3f}" if label in scored[name] else f"{_MISSING:>{_WIDTH}}"
            for label in labels
        ]
        lines.append(f"{name:{_NAME}} " + " ".join(cells))

    return "\n".join(lines)


def halving(records: Sequence[BenchRecord], top_rate: str, half_rate: str) -> str:
    """What each class makes of the two ways of storing half the bytes: half the rate, or half the grid.

    Both spend the same bytes, so which of them a class prefers is what says whether its sound lives in
    the band it occupies or in the resolution each frame is held at.
    """
    lines = [f"=== halving the bytes  (rate {half_rate} against a shallower grid at {top_rate})"]
    lines.append(f"{'class':{_NAME}} " + " ".join(f"{name:>{_WIDTH}}" for name in ("as stored", *_HALVED.values())))
    for name, takes in sorted(_by_class(records).items()):
        stored = _mean_by_label(takes, "rate").get(top_rate)
        halved = [
            _mean_by_label(takes, "rate").get(half_rate),
            _mean_by_label(takes, "depth").get("8"),
            _mean_by_label(takes, "carrier").get("8"),
        ]
        cells = [stored, *halved]
        rendered = " ".join(f"{value:{_WIDTH}.3f}" if value is not None else f"{_MISSING:>{_WIDTH}}" for value in cells)
        lines.append(f"{name:{_NAME}} " + rendered)

    return "\n".join(lines)


def term_shares(records: Sequence[BenchRecord], weights: Mapping[str, float]) -> str:
    """What share of the composite each term carries, once the weight it is summed under is applied.

    The bench stores each term as its own metric measured it and the composite sums those under
    ``weights``, so a term's share is its weighted contribution against the whole -- which is what says
    how far the number written in the config indicates the influence the term actually has.
    """
    gathered: dict[str, dict[str, list[float]]] = {}
    for record in records:
        weighted = {term: weights.get(term, 0.0) * value for term, value in record.breakdown.items()}
        total = sum(weighted.values())
        if total <= 0.0:
            continue

        shares = gathered.setdefault(record.material_class, {})
        for term, value in weighted.items():
            shares.setdefault(term, []).append(value / total)

    terms = sorted({term for shares in gathered.values() for term in shares})
    header = f"{'class':{_NAME}} " + " ".join(f'{term:>{_WIDTH}}' for term in terms)
    lines = ["=== share of the composite each term carries, weighted as the objective sums them", header]
    for name in sorted(gathered):
        cells = " ".join(f'{fmean(gathered[name].get(term, [0.0])):{_WIDTH}.3f}' for term in terms)
        lines.append(f"{name:{_NAME}} " + cells)

    return "\n".join(lines)


def _agreement(pairs: Sequence[tuple[float, float]]) -> float:
    """Kendall's tau-b between two rankings of the same degradations, over the pairs both of them order."""
    concordant = discordant = tied_first = tied_second = 0
    for (first_a, second_a), (first_b, second_b) in combinations(pairs, 2):
        first = (first_a > first_b) - (first_a < first_b)
        second = (second_a > second_b) - (second_a < second_b)
        if first == 0:
            tied_first += _CONCORDANT
        if second == 0:
            tied_second += _CONCORDANT
        if first and second:
            concordant += _CONCORDANT if first == second else 0
            discordant += 0 if first == second else _CONCORDANT

    total = concordant + discordant
    scale = ((total + tied_first) * (total + tied_second)) ** 0.5
    return (concordant - discordant) / scale if scale else 0.0


def diagnostics_agreement(records: Sequence[BenchRecord]) -> str:
    """How closely the composite's ordering of the degradations follows a reading outside it.

    Segmental SNR sits outside the objective, so where the two orderings part company is where a claim
    made on the composite alone carries the least weight.
    """
    lines = ["=== composite against segmental SNR (Kendall tau-b over one class's degradations)"]
    for name, takes in sorted(_by_class(records).items()):
        paired = [
            (record.fidelity, -record.segmental_snr_db) for record in takes if record.segmental_snr_db is not None
        ]
        lines.append(f"{name:{_NAME}} {_agreement(paired):{_WIDTH}.3f}  over {len(paired):4} readings")

    return "\n".join(lines)


def report(
    records: Sequence[BenchRecord],
    top_rate: str,
    half_rate: str,
    weights: Mapping[str, float],
) -> str:
    """Every table the metric-limits bench answers."""
    axes = sorted({record.axis for record in records})
    blocks = [ladder(records, axis) for axis in axes]
    blocks.append(halving(records, top_rate, half_rate))
    blocks.append(term_shares(records, weights))
    blocks.append(diagnostics_agreement(records))
    return "\n\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> None:
    """Print what the composite made of every way of spending fewer bytes."""
    parser = argparse.ArgumentParser(prog="research.limits", description="Tabulate the metric-limits bench.")
    parser.add_argument("records", type=Path, help="The bench's JSONL readings")
    parser.add_argument("--top-rate", default="44100", help="The rate a recording is stored at as captured")
    parser.add_argument("--half-rate", required=True, help="The rate that halves those bytes")
    parser.add_argument("--config", type=Path, default=None, help="Config the term weights are read from")
    args = parser.parse_args(argv)
    weights = load_config(args.config).analysis.metrics.weights
    print(report(read(args.records), args.top_rate, args.half_rate, weights))


if __name__ == "__main__":
    main()
