"""Solo entity (geotechnical input).

Pure data container — no project imports. The admissible bearing
pressure is **not** a method on this class on purpose: it is an
empirical engineering correlation that lives in
``core.engineering.solo.tensao_adm_solo`` and must be called from
the engineering or API layer (``core.api.evaluate``,
``core.api.optimize``). Keeping the correlation outside the domain
preserves the architectural rule "core.domain depends on nothing
inside the project".

Resumo em português:
    Representação imutável das características geotécnicas associadas
    a uma fundação: tipo de solo (qualitativo) e índice SPT
    (quantitativo). Para obter ``sigma_adm`` use
    ``from core.engineering import tensao_adm_solo`` — a entidade
    propositalmente **não** importa engenharia para preservar a
    regra de dependência declarada em ``ARCHITECTURE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


SoilType = str   # alias semantico; aceita "pedregulho", "areia", "silte", "argila"


@dataclass(frozen=True, slots=True)
class Solo:
    """This class represents the soil characteristics under a single foundation.

    Immutable record bundling the qualitative soil type and the SPT
    index obtained from a borehole. The admissible bearing pressure
    derivation is delegated to
    :func:`core.engineering.tensao_adm_solo` — the entity itself is
    a pure data container and does not import the engineering layer.

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
