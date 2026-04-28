"""AABB overlap helpers used by the packing constraint.

Resumo em português:
    Cálculo da área de interseção entre retângulos alinhados aos
    eixos cartesianos (AABB). Usado pela restrição de não-sobreposição
    entre sapatas vizinhas. A versão escalar (oito vértices) é mantida
    por compatibilidade; a versão matricial faz o cálculo N×N
    inteiramente em numpy e é o caminho quente do `_avaliar_projeto`
    desde a Sprint 3.8.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sobreposicao_sapatas(
    x1_i: float, y1_i: float, x2_i: float, y2_i: float,
    x3_i: float, y3_i: float, x4_i: float, y4_i: float,
    x1_j: float, y1_j: float, x2_j: float, y2_j: float,
    x3_j: float, y3_j: float, x4_j: float, y4_j: float,
) -> float:
    """This function returns the AABB overlap area between two axis-aligned rectangles.

    Each rectangle is described by its four vertices (eight scalars in
    the order x1, y1, ..., x4, y4). The function reduces both shapes to
    their AABB by taking the per-axis min/max of the four supplied
    coordinates and then computes the overlap on each axis. The total
    overlap area is the product of the two per-axis overlaps and is
    zero when the rectangles do not intersect or only touch on an edge.

    :param x1_i: Vertex 1, x coordinate of rectangle i [m]
    :param y1_i: Vertex 1, y coordinate of rectangle i [m]
    :param x2_i: Vertex 2, x coordinate of rectangle i [m]
    :param y2_i: Vertex 2, y coordinate of rectangle i [m]
    :param x3_i: Vertex 3, x coordinate of rectangle i [m]
    :param y3_i: Vertex 3, y coordinate of rectangle i [m]
    :param x4_i: Vertex 4, x coordinate of rectangle i [m]
    :param y4_i: Vertex 4, y coordinate of rectangle i [m]
    :param x1_j: Vertex 1, x coordinate of rectangle j [m]
    :param y1_j: Vertex 1, y coordinate of rectangle j [m]
    :param x2_j: Vertex 2, x coordinate of rectangle j [m]
    :param y2_j: Vertex 2, y coordinate of rectangle j [m]
    :param x3_j: Vertex 3, x coordinate of rectangle j [m]
    :param y3_j: Vertex 3, y coordinate of rectangle j [m]
    :param x4_j: Vertex 4, x coordinate of rectangle j [m]
    :param y4_j: Vertex 4, y coordinate of rectangle j [m]

    :return: Overlap area between the two rectangles [m^2]
    """
    xi_min = min(x1_i, x2_i, x3_i, x4_i)
    xi_max = max(x1_i, x2_i, x3_i, x4_i)
    yi_min = min(y1_i, y2_i, y3_i, y4_i)
    yi_max = max(y1_i, y2_i, y3_i, y4_i)

    xj_min = min(x1_j, x2_j, x3_j, x4_j)
    xj_max = max(x1_j, x2_j, x3_j, x4_j)
    yj_min = min(y1_j, y2_j, y3_j, y4_j)
    yj_max = max(y1_j, y2_j, y3_j, y4_j)

    overlap_x = max(0, min(xi_max, xj_max) - max(xi_min, xj_min))
    overlap_y = max(0, min(yi_max, yj_max) - max(yi_min, yj_min))
    return overlap_x * overlap_y


def sobreposicao_matrix(
    xmin: NDArray[np.float64],
    xmax: NDArray[np.float64],
    ymin: NDArray[np.float64],
    ymax: NDArray[np.float64],
) -> NDArray[np.float64]:
    """This function returns the N×N matrix of pairwise AABB overlap areas.

    The four input arrays describe the axis-aligned bounding box of
    each rectangle (one entry per rectangle). For every ordered pair
    (i, j) the overlap on each axis is taken as
    ``max(0, min(max_i, max_j) - max(min_i, min_j))`` and the matrix
    cell is the product of the two per-axis overlaps. The diagonal is
    forced to zero so that summing rows skips the self-pair, matching
    the historical ``j != i`` guard of the loop-based implementation.

    Resumo em português:
        Versão vetorizada de :func:`sobreposicao_sapatas`. Recebe os
        bounds AABB já reduzidos (xmin, xmax, ymin, ymax) — um valor
        por sapata — e devolve a matriz N×N das áreas de sobreposição
        entre cada par de retângulos. Substitui o laço duplo
        ``df.iterrows()`` em ``fundacao._avaliar_projeto``.

    :param xmin: Lower x bound of each rectangle, shape (N,) [m]
    :param xmax: Upper x bound of each rectangle, shape (N,) [m]
    :param ymin: Lower y bound of each rectangle, shape (N,) [m]
    :param ymax: Upper y bound of each rectangle, shape (N,) [m]

    :return: Matrix of pairwise overlap areas, shape (N, N), with
             zeroed diagonal [m^2]
    """
    xmin = np.asarray(xmin, dtype=np.float64)
    xmax = np.asarray(xmax, dtype=np.float64)
    ymin = np.asarray(ymin, dtype=np.float64)
    ymax = np.asarray(ymax, dtype=np.float64)

    overlap_x = np.maximum(
        0.0,
        np.minimum(xmax[:, None], xmax[None, :])
        - np.maximum(xmin[:, None], xmin[None, :]),
    )
    overlap_y = np.maximum(
        0.0,
        np.minimum(ymax[:, None], ymax[None, :])
        - np.maximum(ymin[:, None], ymin[None, :]),
    )
    overlap = overlap_x * overlap_y
    np.fill_diagonal(overlap, 0.0)
    return overlap
