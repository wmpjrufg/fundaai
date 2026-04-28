"""Domain layer — business entities of the foundation design problem.

This subpackage hosts the immutable, framework-free dataclasses that
represent the project entities: ``Solo``, ``Pilar``, ``Combinacao``,
``Sapata`` and the root aggregator ``FundacaoProjeto``.

Resumo em português:
    Camada de domínio. Define as entidades de negócio (Solo, Pilar,
    Combinação, Sapata, FundacaoProjeto) como dataclasses puras, sem
    dependência de pandas, sklearn, mealpy ou Streamlit.
"""

from .combinacao import Combinacao
from .pilar import Pilar
from .projeto import FundacaoProjeto
from .sapata import Sapata
from .solo import Solo, SoilType

__all__ = [
    "Combinacao",
    "FundacaoProjeto",
    "Pilar",
    "Sapata",
    "Solo",
    "SoilType",
]
