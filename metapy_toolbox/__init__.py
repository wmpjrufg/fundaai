"""metapy_toolbox — núcleo de otimização do FundaIA.

Re-exporta utilitários (funcs), funções de benchmark, Algoritmo Genético,
Grey Wolf Optimizer e a arquitetura EGO híbrida.

Histórico: o módulo `methods.py` foi removido em refactor/code-base-v2 por
estar 100% comentado; sua única função pública (`initial_population_01`)
permanece em `funcs.py`, com versão atualizada.
"""

from .funcs import *
from .benchmark import *
from .genetic_algorithm import *
from .grey_wolf import *
from .ego import *
