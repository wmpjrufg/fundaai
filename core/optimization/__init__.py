"""Optimization layer — surrogate-assisted global optimisation.

This subpackage hosts the algorithms that originally lived under
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
    * Content-addressed surrogate cache in ``cache``.

The legacy package ``metapy_toolbox`` was retired in **Sprint 4.3**;
all imports must use ``from core.optimization import ...`` directly.

Resumo em português:
    Camada de otimização. Hospeda EGO+GPR+AG, GWO, funções benchmark,
    utilitários comuns e o cache do surrogate. Todo código novo deve
    importar diretamente de ``core.optimization``; o pacote
    ``metapy_toolbox`` foi removido na Sprint 4.3.
"""

from .funcs import *   # noqa: F401, F403  (intentional re-export)
from .benchmark import *   # noqa: F401, F403
from .genetic_algorithm import *   # noqa: F401, F403
from .grey_wolf import *   # noqa: F401, F403
from .ego import *   # noqa: F401, F403
from .cbo import cbo_01_architecture   # noqa: F401  (explicit re-export)
from .cache import (  # noqa: F401  (explicit re-export)
    SurrogateCache,
    fingerprint,
    fit_or_get_cached,
    pipeline_signature,
)
