"""FundacaoProjeto entity (root aggregator).

Resumo em português:
    Raiz do agregado que reúne todos os pilares de um projeto, o solo
    sob cada um e as combinações de carregamento por elemento, mais os
    parâmetros globais de projeto (`f_ck` e cobrimento).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from core.domain.combinacao import Combinacao
from core.domain.pilar import Pilar
from core.domain.solo import Solo


@dataclass(frozen=True, slots=True)
class FundacaoProjeto:
    """This class represents the full foundation design problem (root aggregator).

    Bundles all pillars of a building, the soil profile under each one
    and the load combinations applied per element, plus the two global
    design parameters required by every code check (``f_ck`` and
    ``cobrimento``).

    Mappings are keyed by ``Pilar.rotulo`` so that the per-element data
    can be retrieved unambiguously even when iteration order changes.

    :param pilares: Sequence of Pilar entities (project order preserved)
    :param solo_por_pilar: Mapping from pillar label to Solo entity
    :param combinacoes_por_pilar: Mapping from pillar label to a sequence
                                  of Combinacao entities (one per load
                                  combination index, in label order)
    :param f_ck_kpa: Characteristic concrete compressive strength [kPa]
    :param cobrimento_m: Concrete cover [m]

    :raises ValueError: When per-pillar maps are missing entries for any
                        declared pillar, when ``f_ck_kpa`` is non-positive
                        or when ``cobrimento_m`` is negative
    """

    pilares: Sequence[Pilar]
    solo_por_pilar: Mapping[str, Solo]
    combinacoes_por_pilar: Mapping[str, Sequence[Combinacao]]
    f_ck_kpa: float
    cobrimento_m: float

    def __post_init__(self) -> None:
        """This hook validates the consistency of the aggregate and global parameters.

        :return: None
        """
        if self.f_ck_kpa <= 0:
            raise ValueError(f"f_ck_kpa must be positive; got {self.f_ck_kpa}.")
        if self.cobrimento_m < 0:
            raise ValueError(
                f"cobrimento_m must be non-negative; got {self.cobrimento_m}."
            )
        rotulos = {p.rotulo for p in self.pilares}
        missing_solo = rotulos - set(self.solo_por_pilar)
        missing_comb = rotulos - set(self.combinacoes_por_pilar)
        if missing_solo:
            raise ValueError(
                f"solo_por_pilar missing entries for: {sorted(missing_solo)}"
            )
        if missing_comb:
            raise ValueError(
                f"combinacoes_por_pilar missing entries for: {sorted(missing_comb)}"
            )

    @property
    def n_fund(self) -> int:
        """This property returns the number of foundation elements in the project.

        :return: Number of pillars (and therefore footings) [int]
        """
        return len(self.pilares)

    @property
    def n_comb(self) -> int:
        """This property returns the number of load combinations declared per element.

        Assumes that every pillar carries the same combination count
        (typical convention of the input spreadsheet). Returns 0 when
        the project has no pillars.

        :return: Number of load combinations [int]
        """
        if not self.pilares:
            return 0
        first = self.pilares[0].rotulo
        return len(self.combinacoes_por_pilar[first])
