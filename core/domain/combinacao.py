"""Combinacao entity (load combination delivered by the superstructure).

Resumo em português:
    Combinação de carregamento (rótulo `c1`, `c2`, ...) com a carga
    axial e os momentos característicos transmitidos pelo pilar à
    sapata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Combinacao:
    """This class represents a single load combination acting on a footing.

    Immutable record bundling the label and the three characteristic
    actions (vertical load and two bending moments) coming from the
    superstructure analysis for a given combination index.

    :param rotulo: Combination label (e.g. "c1", "c2", "c3")
    :param f_z: Characteristic vertical load on the column [kN]
    :param m_x: Characteristic bending moment about the X axis [kN m]
    :param m_y: Characteristic bending moment about the Y axis [kN m]
    """

    rotulo: str
    f_z: float
    m_x: float
    m_y: float
