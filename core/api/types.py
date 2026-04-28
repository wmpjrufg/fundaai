"""Public dataclasses exposed by the API layer.

Two dataclasses live here:

    * ``OptimisationConfig`` bundles every knob that the user (or a
      Streamlit page, or a notebook) exposes to the optimisation
      pipeline: bounds on the design variables, generations, population
      size, number of independent repetitions, base seed and the
      kernel/optimiser strings consumed by ``ego_01_architecture``.

    * ``OptimisationResult`` is the structured answer of ``optimize``:
      the best objective value found, the corresponding list of
      ``Sapata`` entities and the seed that produced the winning
      repetition.

Both are deliberately framework-free (pure dataclasses) so that the
result can travel from a notebook to a CSV report or to the Streamlit
session state without any glue code.

Resumo em português:
    Dataclasses públicas da camada API. Encapsulam a configuração da
    otimização e o resultado final, mantendo as fronteiras tipadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from core.domain import Sapata


@dataclass(frozen=True, slots=True)
class OptimisationConfig:
    """This class bundles every parameter consumed by the API ``optimize`` function.

    All fields have safe defaults that mirror the historical Streamlit
    page (``n_rep = 5``, ``n_gen = 2``, ``n_pop = 250``, ``base_seed = 42``).
    The penalty factor stays optional and falls back to the engineering
    default ``_PENALTY_DEFAULT = 10`` when ``None``.

    :param h_min_m: Lower bound for h_x, h_y, h_z [m]
    :param h_max_m: Upper bound for h_x, h_y, h_z [m]
    :param n_gen: Number of EGO generations per repetition
    :param n_pop: Initial Latin Hypercube population size
    :param n_rep: Number of independent EGO repetitions (best-of-N selection)
    :param base_seed: Seed used to derive ``rep_seed = base_seed + rep``
    :param kernel_index: Index in ``constroi_kernel()`` that selects the
                         GPR covariance function. ``-1`` means "the last
                         kernel", which is the production default
    :param ga_epoch: ``epoch`` parameter passed to ``mealpy.GA.BaseGA``
    :param ga_pop_size: ``pop_size`` parameter passed to ``mealpy.GA.BaseGA``
    :param penalty: Penalty factor applied to constraint violations.
                    ``None`` falls back to the engineering default

    :raises ValueError: When the configuration is internally inconsistent
                        (non-positive bounds, h_min >= h_max, non-positive
                        counts, ...)
    """

    h_min_m: float = 0.60
    h_max_m: float = 1.50
    n_gen: int = 2
    n_pop: int = 250
    n_rep: int = 5
    base_seed: int = 42
    kernel_index: int = -1
    ga_epoch: int = 50
    ga_pop_size: int = 150
    penalty: float | None = None

    def __post_init__(self) -> None:
        """This hook validates the cross-field invariants.

        :return: None
        """
        if self.h_min_m <= 0 or self.h_max_m <= 0:
            raise ValueError(
                f"h_min_m and h_max_m must be positive; "
                f"got h_min_m={self.h_min_m}, h_max_m={self.h_max_m}."
            )
        if self.h_min_m >= self.h_max_m:
            raise ValueError(
                f"h_min_m must be strictly less than h_max_m; "
                f"got h_min_m={self.h_min_m}, h_max_m={self.h_max_m}."
            )
        if self.n_gen < 1:
            raise ValueError(f"n_gen must be >= 1; got {self.n_gen}.")
        if self.n_pop < 2:
            raise ValueError(f"n_pop must be >= 2; got {self.n_pop}.")
        if self.n_rep < 1:
            raise ValueError(f"n_rep must be >= 1; got {self.n_rep}.")
        if self.ga_epoch < 1 or self.ga_pop_size < 2:
            raise ValueError(
                f"ga_epoch must be >= 1 and ga_pop_size must be >= 2; "
                f"got ga_epoch={self.ga_epoch}, ga_pop_size={self.ga_pop_size}."
            )
        if self.penalty is not None and self.penalty <= 0:
            raise ValueError(f"penalty (when set) must be positive; got {self.penalty}.")


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    """This class holds the structured answer produced by ``optimize``.

    The repetition that achieved the lowest pseudo-objective is
    selected; its design vector is converted back to the list of
    ``Sapata`` entities. ``per_rep_of`` keeps the best objective from
    each repetition for downstream reporting (e.g. ``mean ± std``).

    :param sapatas: List of optimised Sapata entities, in pillar order
    :param best_of: Lowest pseudo-objective value (volume + penalised
                    violations) across all repetitions [m^3]
    :param best_seed: Seed of the repetition that produced ``best_of``
    :param per_rep_of: Best objective achieved on each repetition
                       (length equals ``config.n_rep``)
    """

    sapatas: Sequence[Sapata]
    best_of: float
    best_seed: int
    per_rep_of: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """This class holds the per-element verification table for a fixed design.

    Returned by ``evaluate`` for diagnosis and for unit tests that need
    to inspect every constraint individually. ``constraints`` mirrors
    the column names produced historically by ``obj_teste``.

    :param of_total: Pseudo-objective value (volume + penalised violations) [m^3]
    :param sapatas: Sapata entities exactly as supplied to ``evaluate``
    :param constraints: Per-element mapping ``rotulo -> {constraint_name: value}``.
                        Constraints follow the convention ``g <= 0`` is feasible
    """

    of_total: float
    sapatas: Sequence[Sapata]
    constraints: dict[str, dict[str, float]] = field(default_factory=dict)
