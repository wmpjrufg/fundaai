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
    :param f_z: Characteristic vertical compressive load on the column [kN];
                must be strictly positive because null load or uplift is
                outside the current soil-contact pre-design model
    :param m_x: Characteristic moment component producing eccentricity
                along X in the FundaIA convention [kN m]
    :param m_y: Characteristic moment component producing eccentricity
                along Y in the FundaIA convention [kN m]

    :raises ValueError: When ``f_z`` is not strictly positive
    """

    rotulo: str
    f_z: float
    m_x: float
    m_y: float

    def __post_init__(self) -> None:
        """This hook validates the strictly positive vertical load precondition.

        :return: None
        """
        if self.f_z <= 0:
            raise ValueError(
                f"combination {self.rotulo!r}: f_z must be strictly positive "
                f"for the current soil-contact pre-design model; got "
                f"f_z={self.f_z}. Null or uplift loads are not supported."
            )
