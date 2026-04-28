"""Public types exposed by the API layer.

Three structures live here:

    * ``OptimisationConfig`` — Pydantic v2 ``BaseModel`` that bundles
      every knob exposed to the optimisation pipeline. Field-level
      constraints (``ge``, ``gt``) and a ``model_validator`` enforce
      the cross-field invariants up front, with rich error messages,
      JSON schema generation and round-trip serialisation built in.

    * ``OptimisationResult`` — frozen dataclass returned by
      ``optimize``: the best objective, the corresponding ``Sapata``
      list, the winning seed and the per-rep trajectory.

    * ``EvaluationResult`` — frozen dataclass returned by ``evaluate``
      for diagnostics: pseudo-objective and the per-element constraint
      table.

The result types stay as dataclasses on purpose — they are produced by
the API itself, not received from the outside world, so Pydantic's
input validation buys us nothing for them. The configuration, on the
other hand, is the natural place for strict validation because it
flows in from Streamlit, CLI or notebooks.

Resumo em português:
    Tipos públicos da camada API. ``OptimisationConfig`` é Pydantic
    (validação rigorosa de entrada vinda da UI/CLI). ``OptimisationResult``
    e ``EvaluationResult`` permanecem como dataclasses imutáveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.domain import Sapata


# =============================================================================
# OptimisationConfig — Pydantic input validation
# =============================================================================
class OptimisationConfig(BaseModel):
    """This class bundles every parameter consumed by the API ``optimize`` function.

    Built on Pydantic v2 to provide rich field-level validation,
    generated JSON schema and frozen instances. All defaults mirror the
    historical Streamlit page (``n_rep = 5``, ``base_seed = 42``,
    ``kernel_index = -1``). The optional ``penalty`` falls back to the
    engineering default ``_PENALTY_DEFAULT = 10`` when ``None``.

    :param h_min_m: Lower bound for h_x, h_y, h_z [m]; must be > 0
    :param h_max_m: Upper bound for h_x, h_y, h_z [m]; must be > 0 and > h_min_m
    :param n_gen: Number of EGO generations per repetition (>= 1)
    :param n_pop: Initial Latin Hypercube population size (>= 2)
    :param n_rep: Number of independent EGO repetitions (>= 1)
    :param base_seed: Seed used to derive ``rep_seed = base_seed + rep``
    :param kernel_index: Index in ``constroi_kernel()`` selecting the GPR
                         covariance function. ``-1`` means "the last kernel"
                         and is the production default
    :param ga_epoch: ``epoch`` parameter for ``mealpy.GA.BaseGA`` (>= 1)
    :param ga_pop_size: ``pop_size`` parameter for ``mealpy.GA.BaseGA`` (>= 2)
    :param penalty: Penalty factor applied to constraint violations
                    (positive when set; ``None`` falls back to the engineering default)

    :raises pydantic.ValidationError: When any single field violates its
                                      constraints. ``ValidationError``
                                      derives from ``ValueError``, so any
                                      ``except ValueError`` block keeps
                                      working as before
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    h_min_m: float = Field(default=0.60, gt=0.0, description="Lower bound for h_x, h_y, h_z [m]")
    h_max_m: float = Field(default=1.50, gt=0.0, description="Upper bound for h_x, h_y, h_z [m]")
    n_gen: int = Field(default=2, ge=1, description="Number of EGO generations per repetition")
    n_pop: int = Field(default=250, ge=2, description="Initial Latin Hypercube population size")
    n_rep: int = Field(default=5, ge=1, description="Number of independent EGO repetitions")
    base_seed: int = Field(default=42, description="Seed used to derive rep_seed = base_seed + rep")
    kernel_index: int = Field(default=-1, description="Index in constroi_kernel(); -1 = last kernel")
    ga_epoch: int = Field(default=50, ge=1, description="mealpy GA epoch")
    ga_pop_size: int = Field(default=150, ge=2, description="mealpy GA population size")
    penalty: float | None = Field(
        default=None,
        gt=0.0,
        description="Penalty factor for constraint violations; None falls back to engineering default",
    )

    @model_validator(mode="after")
    def _check_bounds_order(self) -> "OptimisationConfig":
        """This validator ensures h_min_m < h_max_m after both fields are set.

        :return: The validated model (Pydantic v2 contract)

        :raises ValueError: When h_min_m >= h_max_m
        """
        if self.h_min_m >= self.h_max_m:
            raise ValueError(
                f"h_min_m must be strictly less than h_max_m; "
                f"got h_min_m={self.h_min_m}, h_max_m={self.h_max_m}."
            )
        return self


# =============================================================================
# Result types — pure dataclasses (produced by the API, not received from outside)
# =============================================================================
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
