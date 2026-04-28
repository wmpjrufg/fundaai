"""Optimization layer — surrogate-assisted global optimisation.

This subpackage hosts the algorithms that historically lived under
``metapy_toolbox``:

    * Common utilities (Latin Hypercube sampling, fitness, evaluation,
      bounds-checking, mutation operators) in ``funcs``.
    * Classical benchmark functions (sphere, rosenbrock, rastrigin,
      ackley, griewank, powell, ...) in ``benchmark``.
    * Genetic Algorithm with eight crossover operators in
      ``genetic_algorithm``.
    * Grey Wolf Optimizer in ``grey_wolf``.
    * Hybrid Efficient Global Optimization (EGO) architecture with GPR
      surrogate in ``ego``.

The legacy package ``metapy_toolbox`` is now a backwards-compatibility
shim that re-exports from this subpackage, so existing imports such as
``from metapy_toolbox import ego_01_architecture`` keep working.

Resumo em português:
    Camada de otimização. Hospeda EGO+GPR+AG, GWO, funções benchmark e
    utilitários comuns, todos migrados de ``metapy_toolbox``. O pacote
    antigo permanece como camada de compatibilidade.
"""

from .funcs import *   # noqa: F401, F403  (intentional re-export)
from .benchmark import *   # noqa: F401, F403
from .genetic_algorithm import *   # noqa: F401, F403
from .grey_wolf import *   # noqa: F401, F403
from .ego import *   # noqa: F401, F403
from .cache import (  # noqa: F401  (explicit re-export)
    SurrogateCache,
    fingerprint,
    fit_or_get_cached,
    pipeline_signature,
)
