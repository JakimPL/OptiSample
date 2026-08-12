from __future__ import annotations

from enum import StrEnum, unique
from importlib.util import find_spec
from typing import Final

import numpy as np

from optisample.cluster.embed import Embedding, classical_scaling, principal_components
from optisample.cluster.space import Coordinates, pairwise_distances

_SKLEARN: Final = "sklearn"  # what t-SNE is read from, installed with the dev group
_UMAP: Final = "umap"  # the module umap-learn installs itself as

_NEIGHBOUR_SHARES: Final = 0  # shares a layout placing points by their company states, its axes carrying none
_TSNE_PERPLEXITY: Final = 30.0  # neighbours t-SNE weighs a point against where the corpus affords that many
_PERPLEXITY_SHARE: Final = 3.0  # recordings per neighbour a smaller corpus holds its perplexity to
_MIN_PERPLEXITY: Final = 2.0  # neighbours t-SNE weighs a point against at the fewest
_UMAP_NEIGHBOURS: Final = 15  # company UMAP reads a point's neighbourhood over where the corpus affords it
_MIN_NEIGHBOURS: Final = 2  # company UMAP reads a neighbourhood over at the fewest
_TSNE_INIT: Final = "pca"  # where t-SNE opens, which is the layout the space's own widest axes give


@unique
class Layout(StrEnum):
    """Which rule places the recordings for the eye, once the space itself is settled.

    ``PCA`` turns the space onto the axes carrying most of its spread and ``MDS`` places points to match
    their distances, so both draw the geometry a group was cut in and what is read off the picture holds in
    the space. ``TSNE`` and ``UMAP`` place each recording beside the company it keeps, which draws a group
    as a cluster while the room between clusters follows the neighbourhoods rather than the distances --
    they are variants for the eye, and the numbers every panel reports stay the space's own.
    """

    PCA = "pca"
    MDS = "mds"
    TSNE = "t-sne"
    UMAP = "umap"

    @property
    def preserves_distance(self) -> bool:
        """Whether the placement stands on the space's distances, which says what a picture may be read for."""
        return self in (Layout.PCA, Layout.MDS)


def available_layouts() -> tuple[Layout, ...]:
    """The layouts this installation offers, the distance-preserving pair first.

    PCA and MDS come from the project's own numerical stack and are always there; t-SNE arrives with
    scikit-learn and UMAP with umap-learn, each probed as it is asked for, so a notebook opened where the
    dev group is absent draws the pair that stands on the space's own geometry.
    """
    offered = [Layout.PCA, Layout.MDS]
    if find_spec(_SKLEARN) is not None:
        offered.append(Layout.TSNE)

    if find_spec(_UMAP) is not None:
        offered.append(Layout.UMAP)

    return tuple(offered)


def _placed(coordinates: Coordinates) -> Embedding:
    """A neighbour layout's placement as an embedding, stating the shares its axes carry as none.

    The axes of such a layout carry the company each point keeps rather than a share of the space's spread,
    so what it has to state is the placement alone.
    """
    return Embedding(
        coordinates=np.asarray(coordinates, dtype=np.float64),
        explained=np.zeros(_NEIGHBOUR_SHARES, dtype=np.float64),
    )


def _tsne(coordinates: Coordinates, *, components: int, seed: int) -> Embedding:
    """``coordinates`` placed by t-SNE, its perplexity held inside what the corpus has neighbours for."""
    from sklearn.manifold import TSNE  # pylint: disable=import-outside-toplevel

    samples = int(coordinates.shape[0])
    perplexity = max(_MIN_PERPLEXITY, min(_TSNE_PERPLEXITY, (samples - 1) / _PERPLEXITY_SHARE))
    model = TSNE(n_components=components, perplexity=perplexity, init=_TSNE_INIT, random_state=seed)
    return _placed(np.asarray(model.fit_transform(coordinates), dtype=np.float64))


def _umap(coordinates: Coordinates, *, components: int, seed: int) -> Embedding:
    """``coordinates`` placed by UMAP, its neighbourhood held inside the company the corpus offers."""
    from umap import UMAP  # pylint: disable=import-outside-toplevel

    samples = int(coordinates.shape[0])
    neighbours = max(_MIN_NEIGHBOURS, min(_UMAP_NEIGHBOURS, samples - 1))
    model = UMAP(n_components=components, n_neighbors=neighbours, random_state=seed)
    return _placed(np.asarray(model.fit_transform(coordinates), dtype=np.float64))


def draw(coordinates: Coordinates, *, layout: Layout, components: int, seed: int) -> Embedding:
    """``coordinates`` laid out in ``components`` dimensions by the rule ``layout`` names.

    Every rule is settled from ``seed`` where it draws at all, so one space read twice draws one picture and
    a finding on screen is there again next time the notebook opens.
    """
    match layout:
        case Layout.PCA:
            return principal_components(coordinates, components)

        case Layout.MDS:
            return classical_scaling(pairwise_distances(coordinates), components)

        case Layout.TSNE:
            return _tsne(coordinates, components=components, seed=seed)

        case Layout.UMAP:
            return _umap(coordinates, components=components, seed=seed)
