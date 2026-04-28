"""Solo entity (geotechnical input).

Resumo em português:
    Representação imutável das características geotécnicas associadas a
    uma fundação: tipo de solo (qualitativo) e índice SPT (quantitativo).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.engineering.solo import tensao_adm_solo


SoilType = str   # alias semantico; aceita "pedregulho", "areia", "silte", "argila"


@dataclass(frozen=True, slots=True)
class Solo:
    """This class represents the soil characteristics under a single foundation.

    Immutable record bundling the qualitative soil type and the SPT
    index obtained from a borehole. The admissible bearing pressure is
    computed lazily through ``sigma_adm_kpa`` so that the entity stays
    a pure data container.

    :param tipo: Soil type identifier ("pedregulho", "areia", "silte" or "argila")
    :param spt: SPT index (Nspt), dimensionless

    :raises ValueError: When `spt` is negative
    """

    tipo: SoilType
    spt: float

    def __post_init__(self) -> None:
        """This hook validates non-negative SPT after dataclass init.

        :return: None
        """
        if self.spt < 0:
            raise ValueError(f"spt must be non-negative; got {self.spt}.")

    @property
    def sigma_adm_kpa(self) -> float:
        """This property returns the admissible soil pressure in kPa.

        Delegates to the engineering layer to keep the empirical
        correlation in a single place.

        :return: Admissible soil pressure [kPa]
        """
        return tensao_adm_solo(self.tipo, self.spt)
