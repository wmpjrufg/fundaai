"""Sapata entity (footing — design variables).

Resumo em português:
    Sapata isolada com as três dimensões (`h_x`, `h_y`, `h_z`) que são
    as variáveis de projeto. Diferente das demais entidades, é
    **mutável**: o otimizador altera as dimensões durante a busca.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.domain.pilar import Pilar


@dataclass(slots=True)
class Sapata:
    """This class represents a single isolated footing as the optimisation variable.

    Mutable on purpose: ``h_x``, ``h_y`` and ``h_z`` are the design
    variables tweaked by the optimiser. The entity keeps a reference to
    the pillar above to derive the in-plane vertices (centred on the
    pillar centroid).

    :param pilar: Pillar above this footing
    :param h_x: Footing dimension on the X axis [m]
    :param h_y: Footing dimension on the Y axis [m]
    :param h_z: Footing height [m]

    :raises ValueError: When any dimension is non-positive
    """

    pilar: Pilar
    h_x: float
    h_y: float
    h_z: float

    def __post_init__(self) -> None:
        """This hook validates strictly positive dimensions.

        :return: None
        """
        if self.h_x <= 0 or self.h_y <= 0 or self.h_z <= 0:
            raise ValueError(
                "footing dimensions must be positive; "
                f"got h_x={self.h_x}, h_y={self.h_y}, h_z={self.h_z}."
            )

    @property
    def volume(self) -> float:
        """This property returns the concrete volume of the footing.

        :return: Volume [m^3]
        """
        return self.h_x * self.h_y * self.h_z

    @property
    def vertices(self) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """This property returns the four AABB vertices in (SW, SE, NE, NW) order.

        Vertices are centred on the column centroid ``(pilar.xg, pilar.yg)``,
        which mirrors the layout assumption of
        ``core.engineering.packing.sobreposicao_sapatas``.

        :return: Tuple of four (x, y) pairs in counter-clockwise order
                 starting at the south-west corner
        """
        xg, yg = self.pilar.xg, self.pilar.yg
        hx2, hy2 = self.h_x / 2, self.h_y / 2
        return (
            (xg - hx2, yg - hy2),   # SW
            (xg + hx2, yg - hy2),   # SE
            (xg + hx2, yg + hy2),   # NE
            (xg - hx2, yg + hy2),   # NW
        )
