"""API layer — high-level entry points consumed by UI, CLI and notebooks.

Exposes two stable, framework-free functions:

    * ``evaluate(projeto, sapatas)`` — runs the pseudo-objective for a
      fixed design and returns the per-element constraint table.
    * ``optimize(projeto, config)`` — runs EGO+GPR+GA with best-of-N
      independent repetitions and returns the winning design.

Plus the typed structures (``OptimisationConfig``, ``OptimisationResult``,
``EvaluationResult``) that travel between this layer and its callers.

Resumo em português:
    Camada de API. ``optimize`` e ``evaluate`` são as portas de saída
    da lógica do projeto; tudo acima (UI, notebooks, CLI) deve passar
    por aqui.
"""

from .evaluate import evaluate
from .optimize import OptimisationCancelled, optimize
from .types import EvaluationResult, OptimisationConfig, OptimisationResult

__all__ = [
    "EvaluationResult",
    "OptimisationCancelled",
    "OptimisationConfig",
    "OptimisationResult",
    "evaluate",
    "optimize",
]
