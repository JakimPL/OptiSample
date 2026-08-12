from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.cluster.space import Coordinates

_EMPTY: Final = 0.0  # the spread a corpus standing at one point holds
_PRODUCT_SCALE: Final = -0.5  # what turns double-centred squared distances into the inner products behind them


@dataclass(frozen=True)
class Embedding:
    """A sample space laid out in few enough dimensions to be looked at, and what each of them carries.

    ``coordinates`` holds one row per recording and ``explained`` the share of the space's whole spread each
    component accounts for, widest first, so a reader knows how much of the geometry a picture is showing.
    """

    coordinates: Coordinates
    explained: Coordinates

    @property
    def components(self) -> int:
        """How many dimensions the layout holds."""
        return int(self.coordinates.shape[1])

    @property
    def covered(self) -> float:
        """The share of the space's spread the components shown carry between them."""
        return float(self.explained.sum())


def principal_components(coordinates: Coordinates, components: int) -> Embedding:
    """``coordinates`` turned onto the axes carrying most of their spread, the widest first.

    The axes come from the singular value decomposition of the centred matrix, so the layout is settled by
    the material alone and one space read twice draws one picture. Distances along the kept axes are the
    space's own distances as far as those axes reach, which has a picture stand for the geometry a
    representative was chosen in.
    """
    centred = np.asarray(coordinates - coordinates.mean(axis=0), dtype=np.float64)
    _, strengths, axes = np.linalg.svd(centred, full_matrices=False)
    kept = min(components, int(strengths.size))
    return Embedding(
        coordinates=np.asarray(centred @ axes[:kept].T, dtype=np.float64),
        explained=_shares(np.square(strengths), kept),
    )


def classical_scaling(distances: Coordinates, components: int) -> Embedding:
    """Points placed so their distances match ``distances`` as closely as ``components`` dimensions allow.

    Double-centring the squared distances gives the inner products a set of points standing that far apart
    would have, and the leading eigenvectors of those products place them. Distances that came from a
    Euclidean space are placed exactly where its principal components put them, up to a turn of the axes,
    so the two layouts read one geometry; distances from anywhere else are placed as near as this space
    reaches.
    """
    samples = int(distances.shape[0])
    centring = np.eye(samples) - np.full((samples, samples), 1.0 / samples)
    products = _PRODUCT_SCALE * centring @ np.square(distances) @ centring
    strengths, axes = np.linalg.eigh(products)
    order = np.argsort(strengths)[::-1]
    carried = np.maximum(strengths[order], _EMPTY)
    kept = min(components, samples)
    return Embedding(
        coordinates=np.asarray(axes[:, order[:kept]] * np.sqrt(carried[:kept]), dtype=np.float64),
        explained=_shares(carried, kept),
    )


def _shares(strengths: Coordinates, kept: int) -> Coordinates:
    """The share of the whole spread each of the leading ``kept`` axes carries, and zeros where it is nil."""
    whole = float(strengths.sum())
    if whole <= _EMPTY:
        return np.zeros(kept, dtype=np.float64)

    return np.asarray(strengths[:kept] / whole, dtype=np.float64)
