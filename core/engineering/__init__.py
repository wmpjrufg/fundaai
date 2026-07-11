"""Engineering layer — pure analytical checks (NBR 6118 / NBR 6122).

This subpackage hosts stateless functions that compute admissible soil
pressure, applied stresses (composite bending), punching shear at the
C critical section, geometric overhang and AABB overlap. Each function
takes plain numerical arguments and returns plain numbers — no
DataFrames in the signature. DataFrame adapters live in higher layers
(``core.api`` or ``core.io``).

Resumo em português:
    Camada de engenharia. Verificações analíticas puras para sapatas
    isoladas (NBR 6118/6122): tensão admissível do solo, σ_max e σ_min,
    restrição de tensão, geometria mínima, punção (seção C) e
    sobreposição AABB.
"""

from .solo import tensao_adm_solo
from .tensao import calcular_sigma_max_min, checagem_tensao_max_min
from .geometria import checagem_geometria
from .puncao import (
    k_tabela_19_2,
    rho_minimo_flexao,
    verificacao_puncao_sapata,
    verificacao_puncao_sapata_c_linha,
)
from .packing import sobreposicao_matrix, sobreposicao_sapatas

__all__ = [
    "tensao_adm_solo",
    "calcular_sigma_max_min",
    "checagem_tensao_max_min",
    "checagem_geometria",
    "k_tabela_19_2",
    "rho_minimo_flexao",
    "verificacao_puncao_sapata",
    "verificacao_puncao_sapata_c_linha",
    "sobreposicao_sapatas",
    "sobreposicao_matrix",
]
