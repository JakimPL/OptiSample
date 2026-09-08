from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from importlib.util import find_spec
from typing import Final

import numpy as np

from notebooks.utils.views import Row
from optisample.cluster.embed import Embedding, classical_scaling, principal_components
from optisample.cluster.space import Coordinates, pairwise_distances

_SKLEARN: Final = "sklearn"  # what t-SNE is read from, installed with the dev group
_UMAP: Final = "umap"  # the module umap-learn installs itself as

_PLANE: Final = 2  # axes a picture drawn on a plane holds, a third one turning it into a box
_PITCH_AXIS: Final = "pitch"  # the note a take plays, which the keys are read across
_VELOCITY_AXIS: Final = "velocity"  # how hard it was struck, which the keys are read up
_DURATION_AXIS: Final = "dur_s"  # how long it rings, the third axis the keys are given in a box
_DURATION_TITLE: Final = "duration (s)"  # what that axis is called, the column holding it going by seconds
_NEIGHBOR_SHARES: Final = 0  # shares a layout placing points by their company states, its axes carrying none
_TSNE_PERPLEXITY: Final = 30.0  # neighbors t-SNE weighs a point against where the corpus affords that many
_PERPLEXITY_SHARE: Final = 3.0  # recordings per neighbor a smaller corpus holds its perplexity to
_MIN_PERPLEXITY: Final = 2.0  # neighbors t-SNE weighs a point against at the fewest
_UMAP_NEIGHBORS: Final = 15  # company UMAP reads a point's neighborhood over where the corpus affords it
_MIN_NEIGHBORS: Final = 2  # company UMAP reads a neighborhood over at the fewest
_TSNE_INIT: Final = "pca"  # where t-SNE opens, which is the layout the space's own widest axes give


@unique
class Layout(StrEnum):
    """Which rule places the recordings for the eye, once the space itself is settled.

    ``PCA`` turns the space onto the axes carrying most of its spread and ``MDS`` places points to match
    their distances, so both draw the geometry a group was cut in and what is read off the picture holds in
    the space. ``TSNE`` and ``UMAP`` place each recording beside the company it keeps, which draws a group
    as a cluster while the room between clusters follows the neighborhoods rather than the distances --
    they are variants for the eye, and the numbers every panel reports stay the space's own. ``KEYS``
    leaves the space aside and lays the corpus out as the keyboard holds it, which reads a grouping against
    the keys it came from.
    """

    PCA = "pca"
    MDS = "mds"
    KEYS = "keys"
    TSNE = "t-sne"
    UMAP = "umap"

    @property
    def reads_the_space(self) -> bool:
        """Whether the placement comes out of the space's own coordinates rather than off what a take plays."""
        return self in (Layout.PCA, Layout.MDS, Layout.TSNE, Layout.UMAP)

    @property
    def preserves_distance(self) -> bool:
        """Whether the placement stands on the space's distances, which says what a picture may be read for."""
        return self in (Layout.PCA, Layout.MDS)


def available_layouts() -> tuple[Layout, ...]:
    """The layouts this installation offers, the always-available ones first.

    PCA, MDS and the keys come from the project's own numerical stack and are always there; t-SNE arrives
    with scikit-learn and UMAP with umap-learn, each probed as it is asked for, so a notebook opened where
    the dev group is absent draws the pair that stands on the space's own geometry beside the keyboard.
    """
    offered = [Layout.PCA, Layout.MDS, Layout.KEYS]
    if find_spec(_SKLEARN) is not None:
        offered.append(Layout.TSNE)

    if find_spec(_UMAP) is not None:
        offered.append(Layout.UMAP)

    return tuple(offered)


@dataclass(frozen=True)
class Placement:
    """Where every recording sits on one drawn picture, beside what the axes it is drawn on are called.

    A picture drawn from a layout that reads the space carries the components an embedding placed each
    recording on; one drawn from the keys carries the note it plays, the velocity it was struck at and, in
    a box, how long it rings. Both stand here in the one shape, so a field is colored, hovered and clicked
    the same way whichever of the two it draws.
    """

    axes: tuple[tuple[float, ...], ...]
    titles: tuple[str, ...]

    @property
    def components(self) -> int:
        """How many axes the picture is drawn on, which is what tells a plane from a box."""
        return len(self.axes)

    @property
    def is_plane(self) -> bool:
        """Whether the picture is drawn flat, which sets how a trace is built and how its axes are named."""
        return self.components <= _PLANE


def _placed(coordinates: Coordinates) -> Embedding:
    """A neighbor layout's placement as an embedding, stating the shares its axes carry as none.

    The axes of such a layout carry the company each point keeps rather than a share of the space's spread,
    so what it has to state is the placement alone.
    """
    return Embedding(
        coordinates=np.asarray(coordinates, dtype=np.float64),
        explained=np.zeros(_NEIGHBOR_SHARES, dtype=np.float64),
    )


def _tsne(coordinates: Coordinates, *, components: int, seed: int) -> Embedding:
    """``coordinates`` placed by t-SNE, its perplexity held inside what the corpus has neighbors for."""
    from sklearn.manifold import TSNE  # pylint: disable=import-outside-toplevel

    samples = int(coordinates.shape[0])
    perplexity = max(_MIN_PERPLEXITY, min(_TSNE_PERPLEXITY, (samples - 1) / _PERPLEXITY_SHARE))
    model = TSNE(n_components=components, perplexity=perplexity, init=_TSNE_INIT, random_state=seed)
    return _placed(np.asarray(model.fit_transform(coordinates), dtype=np.float64))


def _umap(coordinates: Coordinates, *, components: int, seed: int) -> Embedding:
    """``coordinates`` placed by UMAP, its neighborhood held inside the company the corpus offers."""
    from umap import UMAP  # pylint: disable=import-outside-toplevel

    samples = int(coordinates.shape[0])
    neighbors = max(_MIN_NEIGHBORS, min(_UMAP_NEIGHBORS, samples - 1))
    model = UMAP(n_components=components, n_neighbors=neighbors, random_state=seed)
    return _placed(np.asarray(model.fit_transform(coordinates), dtype=np.float64))


def draw(coordinates: Coordinates, *, layout: Layout, components: int, seed: int) -> Embedding:
    """``coordinates`` laid out in ``components`` dimensions by the rule ``layout`` names.

    Every rule is settled from ``seed`` where it draws at all, so one space read twice draws one picture and
    a finding on screen is there again next time the notebook opens.

    Raises:
        ValueError: when ``layout`` places recordings by what they play, which ``place`` reads off the rows.
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

        case Layout.KEYS:
            raise ValueError(f"{layout.value} places a recording at the key it plays, which the rows carry")


def _axis_title(embedding: Embedding, axis: int) -> str:
    """What one drawn axis is called, carrying the share of the spread it holds where the layout states one."""
    if axis < embedding.explained.size:
        return f"component {axis + 1} — {embedding.explained[axis]:.1%}"

    return f"component {axis + 1}"


def _laid_out(embedding: Embedding) -> Placement:
    """The recordings where the layout put them, on axes named for the share of the spread each carries."""
    return Placement(
        axes=tuple(
            tuple(float(value) for value in embedding.coordinates[:, axis]) for axis in range(embedding.components)
        ),
        titles=tuple(_axis_title(embedding, axis) for axis in range(embedding.components)),
    )


def _keyed(rows: Sequence[Row], components: int) -> Placement:
    """The recordings at the keys they play: the note across, the velocity up, and in a box the length away."""
    axes = [
        tuple(float(row[_PITCH_AXIS]) for row in rows),
        tuple(float(row[_VELOCITY_AXIS]) for row in rows),
    ]
    titles = [_PITCH_AXIS, _VELOCITY_AXIS]
    if components > _PLANE:
        axes.append(tuple(float(row[_DURATION_AXIS]) for row in rows))
        titles.append(_DURATION_TITLE)

    return Placement(axes=tuple(axes), titles=tuple(titles))


def place(coordinates: Coordinates, rows: Sequence[Row], *, layout: Layout, components: int, seed: int) -> Placement:
    """Where every recording sits on the picture ``layout`` draws, and what the axes it sits on are called.

    A layout reading the space places each recording by the geometry its groups were cut in; the keys place
    it at what it plays, which reads a grouping against the keyboard it came from. Both answer in the one
    shape, so a field is colored, hovered and clicked the same way whichever a reader picked.
    """
    if layout.reads_the_space:
        return _laid_out(draw(coordinates, layout=layout, components=components, seed=seed))

    return _keyed(rows, components)
