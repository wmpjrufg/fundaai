"""Pilar entity (column data passed to the foundation).

Resumo em português:
    Pilar de concreto armado posicionado em planta. Contém o rótulo do
    elemento, as dimensões `(a_p, b_p)` da seção e as coordenadas
    `(xg, yg)` do centróide.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pilar:
    """This class represents a reinforced-concrete column above a footing.

    Immutable record bundling the geometric data that the foundation
    design needs from the structural project: section dimensions and
    centroid position.

    :param rotulo: Element label as it appears in the input spreadsheet (e.g. "P04")
    :param a_p: Column dimension along the X axis [m]
    :param b_p: Column dimension along the Y axis [m]
    :param xg: X coordinate of the column centroid [m]
    :param yg: Y coordinate of the column centroid [m]

    :raises ValueError: When `a_p` or `b_p` are non-positive
    """

    rotulo: str
    a_p: float
    b_p: float
    xg: float
    yg: float

    def __post_init__(self) -> None:
        """This hook validates positive section dimensions.

        :return: None
        """
        if self.a_p <= 0 or self.b_p <= 0:
            raise ValueError(
                f"column dimensions must be positive; got a_p={self.a_p}, b_p={self.b_p}."
            )
