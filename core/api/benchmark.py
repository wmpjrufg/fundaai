"""Benchmark entry point — head-to-head comparison between EGO and pure metaheuristics.

This is the experiment bench consumed by the Streamlit ``Experimentos``
page (and by future notebook/CLI clients). It encapsulates the
orchestration required to compare ``EGO+GPR`` against pure ``GA``,
``PSO`` and ``GWO`` on the *same* objective function, under a *common
evaluation budget*, with seeds controlled per repetition.

Design notes
------------
The metric of merit is the **best objective per number of real
objective evaluations**, not wall-clock time. This is the honest
argument for EGO when the objective is cheap (as it is today): a
surrogate-assisted method should converge in fewer real evaluations
than its pure-metaheuristic counterparts, even if each surrogate
iteration costs more than one objective evaluation. Wall-clock time
is reported as a secondary metric for completeness.

Every algorithm sees the same wrapped objective ``TracedObjective``
that (a) counts evaluations, (b) records the per-evaluation trace
``(eval_idx, of_value, of_best_so_far, time_eval_s, time_total_s)``
and (c) raises ``_BudgetExhausted`` (a ``BaseException`` subclass, so
it bypasses generic ``except Exception`` blocks inside ``mealpy``/
``scipy``) the moment the budget is hit.

Resumo em português:
    ``run_benchmark`` executa um conjunto de algoritmos (EGO, GA, PSO,
    GWO) sobre o mesmo projeto, com o **mesmo orçamento de avaliações
    reais** e ``n_rep`` repetições com seeds controladas. Devolve um
    ``BenchmarkResult`` tipado com histórico unificado (linha por
    avaliação real) e tabela-resumo pronta para o artigo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from mealpy import FloatVar, GA, GWO, PSO
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.api._adapter import projeto_to_dataframe
from core.api.types import OptimisationConfig
from core.domain import FundacaoProjeto
from core.observability import get_logger, run_context
from core.optimization import (
    cbo_01_architecture,
    ego_01_architecture,
    initial_population_01,
)
from core.api.objective import (
    avaliar_projeto_componentes,
    avaliar_projeto_fast,
    avaliar_projeto_legacy,
)
from core.api._adapter import design_vector_to_sapatas
from fundacao import constroi_kernel  # moved to core.optimization in Sprint 5.x

_log = get_logger("benchmark")


Algorithm = Literal["ego", "cbo", "ga", "pso", "gwo", "random"]
ALL_ALGORITHMS: tuple[Algorithm, ...] = ("ego", "cbo", "ga", "pso", "gwo", "random")

ALGORITHM_LABELS: dict[str, str] = {
    "ego":    "EGO + GPR",
    "cbo":    "CBO (ECI)",
    "ga":     "GA puro",
    "pso":    "PSO puro",
    "gwo":    "GWO puro",
    "random": "Busca aleatória",
}


# =============================================================================
# Internal sentinels
# =============================================================================
class _BudgetExhausted(BaseException):
    """Sentinel raised by :class:`TracedObjective` when the evaluation budget is hit.

    Inherits from ``BaseException`` (not ``Exception``) so it bypasses
    blanket ``except Exception`` clauses inside the inner optimisers
    (mealpy ``solve``, scipy minimisers, scikit-learn GPR fitting),
    exactly like ``core.optimization.ego._CancelSentinel`` does for
    cooperative cancellation.
    """


# =============================================================================
# Configuration
# =============================================================================
class BenchmarkConfig(BaseModel):
    """Public configuration for :func:`run_benchmark`.

    Built on Pydantic v2 to mirror the validation style of
    :class:`core.api.OptimisationConfig`. Every algorithm in
    ``algorithms`` runs ``n_rep`` independent repetitions, each
    capped at ``budget_evals`` real objective evaluations. The
    repetition seed is ``base_seed + rep``, so two ``run_benchmark``
    calls with the same configuration produce the same per-rep
    trajectories.

    :param algorithms: Tuple with the algorithms to compare. Must
                       contain at least one entry. Order is preserved
                       in the output history
    :param budget_evals: Maximum number of real objective evaluations
                         per repetition (shared by every algorithm)
    :param n_rep: Number of independent repetitions per algorithm
    :param base_seed: Seed used to derive ``rep_seed = base_seed + rep``
    :param h_min_m: Lower bound for ``h_x, h_y, h_z`` [m]
    :param h_max_m: Upper bound for ``h_x, h_y, h_z`` [m]
    :param lhs_n_pop: LHS initial population size used by EGO. Must be
                      ``< ego_budget_evals`` so EGO has room for at least
                      one surrogate iteration
    :param ego_budget_evals: Maximum number of real objective evaluations
                             per repetition **for EGO only**. Independent
                             from ``budget_evals`` (used by GA/PSO/GWO).
                             Typical values: 100–300 (EGO is efficient;
                             it does not need thousands of evaluations)
    :param meta_pop_size: Population size used by GA / PSO / GWO
    :param kernel_index: Index in ``constroi_kernel()`` for the EGO GPR
    :param ga_pop_size: Population of the GA that maximises EI **inside**
                        EGO (does **not** touch the real objective —
                        only the surrogate). Independent from
                        ``meta_pop_size`` so the EI optimiser can be
                        tuned separately from the pure metaheuristics
    :param ga_epoch: Number of epochs of the GA that maximises EI
                     inside EGO
    :param penalty: Penalty factor applied to constraint violations
                    (positive when set; ``None`` falls back to the
                    engineering default)
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    algorithms: tuple[Algorithm, ...] = Field(
        default=ALL_ALGORITHMS,
        min_length=1,
        description="Algorithms to compare (subset of EGO / GA / PSO / GWO / random)",
    )
    budget_evals: int = Field(default=200, ge=10, description="Real evaluations per repetition for GA/PSO/GWO")
    ego_budget_evals: int = Field(
        default=150, ge=10,
        description=(
            "Real evaluations per repetition for EGO only. "
            "Independent from budget_evals (GA/PSO/GWO). "
            "Typical: 100-300 (EGO is sample-efficient by design)."
        ),
    )
    n_rep: int = Field(default=10, ge=1, description="Independent repetitions per algorithm")
    base_seed: int = Field(default=42, description="rep_seed = base_seed + rep")
    h_min_m: float = Field(default=0.60, gt=0.0, description="Lower bound for h_x, h_y, h_z [m]")
    h_max_m: float = Field(default=1.50, gt=0.0, description="Upper bound for h_x, h_y, h_z [m]")
    lhs_n_pop: int = Field(default=50, ge=2, description="EGO LHS initial population")
    meta_pop_size: int = Field(default=30, ge=2, description="GA/PSO/GWO population size")
    kernel_index: int = Field(default=-1, description="Index in constroi_kernel(); -1 = last kernel")
    ga_pop_size: int = Field(default=150, ge=2, description="Inner-EI GA population (surrogate only)")
    ga_epoch: int = Field(default=50, ge=1, description="Inner-EI GA epochs (surrogate only)")
    penalty: float | None = Field(default=None, gt=0.0, description="Penalty for constraint violations")
    cbo_constraint_restarts: int = Field(
        default=5, ge=1,
        description=(
            "n_restarts_optimizer of the CBO constraint GPs (the volume "
            "GP keeps the production value of 5). Lower values trade "
            "hyperparameter-fit quality of the smoother constraint "
            "targets for wall time; recorded so every run is "
            "reproducible from config.json."
        ),
    )
    fo_variant: str = Field(
        default="fast",
        description=(
            "Implementacao da funcao objetivo a usar: "
            "'fast' = _avaliar_projeto_fast (numpy vetorizado, ~0,1 ms/eval, Sprint 3.9); "
            "'legacy' = _avaliar_projeto via pandas/df.apply (~10 ms/eval, versao original). "
            "Use 'legacy' apenas para benchmarks de comparacao de desempenho."
        ),
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> "BenchmarkConfig":
        if self.h_min_m >= self.h_max_m:
            raise ValueError(
                f"h_min_m must be strictly less than h_max_m; "
                f"got h_min_m={self.h_min_m}, h_max_m={self.h_max_m}."
            )
        if self.lhs_n_pop >= self.ego_budget_evals:
            raise ValueError(
                f"lhs_n_pop ({self.lhs_n_pop}) must be strictly less than "
                f"ego_budget_evals ({self.ego_budget_evals}); EGO needs room "
                f"for at least one surrogate iteration."
            )
        seen: set[str] = set()
        for alg in self.algorithms:
            if alg in seen:
                raise ValueError(f"algorithm {alg!r} repeated in `algorithms`.")
            seen.add(alg)
        return self


# =============================================================================
# Result types
# =============================================================================
@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Structured answer produced by :func:`run_benchmark`.

    :param history: Long-format DataFrame with one row per real
                    objective evaluation. Columns:
                    ``algorithm`` (str), ``rep`` (int), ``seed`` (int),
                    ``eval_idx`` (int, 1-based), ``of_value`` (float),
                    ``of_best_so_far`` (float), ``time_eval_s`` (float),
                    ``time_total_s`` (float)
    :param summary: Wide-format DataFrame with one row per algorithm.
                    Columns: ``algorithm``, ``label``, ``n_rep``,
                    ``best``, ``mean``, ``std``, ``median``,
                    ``auc_mean``, ``auc_std``, ``conv_eval_mean``,
                    ``conv_eval_std``, ``wall_time_mean_s``,
                    ``wall_time_std_s``
    :param pvalues: Pairwise Wilcoxon signed-rank two-sided p-values on
                    the per-rep best, paired by repetition and adjusted
                    with Holm's step-down correction within each matrix
                    (one row + column per algorithm). Diagonal is ``NaN``
    :param config: Echo of the :class:`BenchmarkConfig` that produced
                   this result
    :param per_rep: One row per (algorithm, repetition) with the final
                    outcome of that repetition: ``best`` (penalised OF),
                    ``volume_m3`` (raw concrete volume of the best
                    design), ``feasible`` (every constraint
                    ``g <= FEASIBILITY_TOL``), ``max_violation`` and the
                    per-group worst constraint values (``viol_sob``,
                    ``viol_pun``, ``viol_ten``, ``viol_geo``), plus
                    ``seed``, ``n_evals`` and ``wall_time_s``
    """

    history: pd.DataFrame
    summary: pd.DataFrame
    pvalues: pd.DataFrame
    config: BenchmarkConfig = field()
    best_sapatas: "Sequence[Any] | None" = field(default=None)
    best_algorithm: str | None = field(default=None)
    best_of_value: float = field(default=float("inf"))
    per_rep: pd.DataFrame | None = field(default=None)


# =============================================================================
# Traced objective
# =============================================================================
class TracedObjective:
    """Callable that wraps the real objective with budget control + tracing.

    Accepts both the EGO call style (``obj(x, args=args)``, kwarg) and
    the mealpy call style (``obj(x)``, positional only). The fixed
    project arguments are captured at construction time so the inner
    optimisers do not need to be aware of them.
    """

    def __init__(
        self,
        base_obj: Callable,
        base_args: tuple,
        budget: int,
        algorithm: str,
        rep: int,
        seed: int,
    ) -> None:
        self._obj = base_obj
        self._args = base_args
        self._budget = int(budget)
        self.algorithm = str(algorithm)
        self.rep = int(rep)
        self.seed = int(seed)
        self._best: float = float("inf")
        self._best_x: list[float] | None = None
        self._records: list[dict[str, Any]] = []
        self._t_start = time.perf_counter()

    # ------------------------------------------------------------------
    @property
    def n_evals(self) -> int:
        return len(self._records)

    @property
    def best(self) -> float:
        return self._best

    @property
    def best_x(self) -> list[float] | None:
        """Design vector that produced the best OF seen so far.

        :return: Copy of the best-seen design vector, or ``None`` if no
                 evaluation has been recorded yet.
        """
        return list(self._best_x) if self._best_x is not None else None

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def history_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(
                columns=["algorithm", "rep", "seed", "eval_idx",
                         "of_value", "of_best_so_far",
                         "time_eval_s", "time_total_s"]
            )
        df = pd.DataFrame(self._records)
        df.insert(0, "algorithm", self.algorithm)
        df.insert(1, "rep", self.rep)
        df.insert(2, "seed", self.seed)
        return df

    # ------------------------------------------------------------------
    def __call__(self, x, args=None):   # noqa: D401  (callable, not docstring sentence)
        if len(self._records) >= self._budget:
            raise _BudgetExhausted()
        t0 = time.perf_counter()
        # Always evaluate against the fixed project args, ignoring whatever
        # the inner optimiser tries to pass in.
        of = float(self._obj(list(x), self._args))
        dt = time.perf_counter() - t0
        if of < self._best:
            self._best = of
            self._best_x = list(x)
        self._records.append({
            "eval_idx": len(self._records) + 1,
            "of_value": of,
            "of_best_so_far": self._best,
            "time_eval_s": float(dt),
            "time_total_s": float(time.perf_counter() - self._t_start),
        })
        return of


class _TracedComponents(TracedObjective):
    """Budget-capped tracer for component-returning objectives (CBO).

    Same budget accounting and per-evaluation trace as
    :class:`TracedObjective` — the recorded ``of_value`` is the
    penalised Theta, so every algorithm is compared on the identical
    metric — but ``__call__`` returns the full ``(theta, volume, g)``
    tuple that :func:`core.optimization.cbo_01_architecture` consumes.
    """

    def __call__(self, x, args=None):   # noqa: D401
        if len(self._records) >= self._budget:
            raise _BudgetExhausted()
        t0 = time.perf_counter()
        theta, volume, g = self._obj(list(x), self._args)
        dt = time.perf_counter() - t0
        theta = float(theta)
        if theta < self._best:
            self._best = theta
            self._best_x = list(x)
        self._records.append({
            "eval_idx": len(self._records) + 1,
            "of_value": theta,
            "of_best_so_far": self._best,
            "time_eval_s": float(dt),
            "time_total_s": float(time.perf_counter() - self._t_start),
        })
        return theta, volume, g


# =============================================================================
# Feasibility report
# =============================================================================
FEASIBILITY_TOL: float = 1e-9
"""Tolerance used to declare a constraint satisfied (``g <= tol``)."""

_CONSTRAINT_COLUMNS: dict[str, str] = {
    "viol_sob": "g sobreposicao",
    "viol_pun": "g punção",       # worst of the C and C' contours
    "viol_ten": "g tensao",
    "viol_geo": "g geometria",
}


def _solution_report(best_x: Sequence[float] | None, args_obj: tuple) -> dict[str, Any]:
    """Evaluate the engineering feasibility of a final design vector.

    Runs the annotated (legacy) evaluation once on ``best_x`` and
    extracts the raw, pre-penalty constraint values so the benchmark can
    report *engineering-meaningful* metrics alongside the penalised OF:
    the raw concrete volume, the worst violation per constraint group
    and a feasibility verdict (every ``g <= FEASIBILITY_TOL``).

    Because the exterior penalty is linear (``alpha = 10``, ``p = 1``),
    a slightly infeasible design can carry a marginally lower penalised
    OF than a feasible one — reporting feasibility explicitly is what
    keeps the algorithm comparison honest.

    :param best_x: Design vector that produced the best OF of a
                   repetition, or ``None`` when the repetition recorded
                   no evaluation
    :param args_obj: Same args tuple consumed by the objective function

    :return: Mapping with ``volume_m3``, ``feasible``, ``max_violation``
             and the per-group worst values (``viol_sob``, ``viol_pun``,
             ``viol_ten``, ``viol_geo``); NaN/False placeholders when
             ``best_x`` is ``None``
    """
    if best_x is None:
        report: dict[str, Any] = {k: float("nan") for k in _CONSTRAINT_COLUMNS}
        report.update({
            "volume_m3": float("nan"),
            "max_violation": float("nan"),
            "feasible": False,
        })
        return report

    from fundacao import obj_teste  # annotated evaluation (legacy core)

    _of, df_annot = obj_teste(list(best_x), args_obj)
    report = {
        key: float(df_annot[col].max())
        for key, col in _CONSTRAINT_COLUMNS.items()
    }
    max_violation = max(report.values())
    report.update({
        "volume_m3": float(df_annot["volume (m3)"].sum()),
        "max_violation": float(max_violation),
        "feasible": bool(max_violation <= FEASIBILITY_TOL),
    })
    return report


# =============================================================================
# Per-algorithm runners
# =============================================================================
def _run_ego(
    traced: TracedObjective,
    *,
    dim: int,
    config: BenchmarkConfig,
    rep_seed: int,
) -> None:
    """Run EGO until either ``n_gen_cap`` surrogate iterations are spent
    or the budget is exhausted (whichever happens first).

    EGO's internals do not care about budget — they raise ``BudgetExhausted``
    on the first call to the traced objective after the cap is reached.
    """
    x_lower = [config.h_min_m] * dim
    x_upper = [config.h_max_m] * dim
    x_ini = initial_population_01(
        config.lhs_n_pop, dim, x_lower, x_upper,
        seed=rep_seed, use_lhs=True,
    )
    paras_opt = {
        "optimizer algorithm": GA.BaseGA(
            epoch=config.ga_epoch, pop_size=config.ga_pop_size,
        )
    }
    kernel_pool = constroi_kernel()
    paras_kernel = {"kernel": kernel_pool[config.kernel_index]}

    # The cap below is just a safety upper bound: BudgetExhausted will
    # almost always fire first.
    n_gen_cap = max(1, config.ego_budget_evals - config.lhs_n_pop)
    try:
        ego_01_architecture(
            traced, n_gen_cap, x_ini, x_lower, x_upper,
            paras_opt, paras_kernel,
            args=None,
            seed=rep_seed,
        )
    except _BudgetExhausted:
        pass


def _run_cbo(
    traced: "_TracedComponents",
    *,
    dim: int,
    config: BenchmarkConfig,
    rep_seed: int,
) -> None:
    """Run constrained Bayesian optimisation under the EGO budget.

    Shares every protocol lever with :func:`_run_ego` (same LHS size,
    same inner-EI genetic algorithm, same production kernel, same
    ``ego_budget_evals`` cap) so the CBO-vs-EGO comparison isolates a
    single factor: how constraints are handled — exterior penalisation
    absorbed by one surrogate versus independent surrogates with the
    constrained acquisition of Gardner et al. (2014).

    :param traced: Budget-capped component tracer
    :param dim: Number of design variables (3 * n_fund)
    :param config: Benchmark configuration
    :param rep_seed: Seed of this repetition (``base_seed + rep``)

    :return: None (the trace lives inside ``traced``)
    """
    x_lower = [config.h_min_m] * dim
    x_upper = [config.h_max_m] * dim
    x_ini = initial_population_01(
        config.lhs_n_pop, dim, x_lower, x_upper,
        seed=rep_seed, use_lhs=True,
    )
    paras_opt = {
        "optimizer algorithm": GA.BaseGA(
            epoch=config.ga_epoch, pop_size=config.ga_pop_size,
        )
    }
    kernel_pool = constroi_kernel()
    paras_kernel = {"kernel": kernel_pool[config.kernel_index]}

    n_gen_cap = max(1, config.ego_budget_evals - config.lhs_n_pop)
    try:
        cbo_01_architecture(
            traced, n_gen_cap, x_ini, x_lower, x_upper,
            paras_opt, paras_kernel,
            args=None,
            seed=rep_seed,
            constraint_n_restarts=config.cbo_constraint_restarts,
        )
    except _BudgetExhausted:
        pass


def _run_random(
    traced: TracedObjective,
    *,
    dim: int,
    config: BenchmarkConfig,
    rep_seed: int,
) -> None:
    """Run the Monte Carlo (uniform random search) baseline.

    Draws ``budget_evals`` independent uniform samples from the search
    box and evaluates each one through the traced objective. This is the
    "tentativa aleatória" baseline used by the manuscript: no memory, no
    learning — the floor any guided search must beat under the same
    budget of real evaluations.

    :param traced: Budget-capped traced objective shared by every algorithm
    :param dim: Number of design variables (3 * n_fund)
    :param config: Benchmark configuration (bounds + budget)
    :param rep_seed: Seed of this repetition (``base_seed + rep``)

    :return: None (the trace lives inside ``traced``)
    """
    rng = np.random.default_rng(rep_seed)
    x_lower = np.full(dim, config.h_min_m)
    x_upper = np.full(dim, config.h_max_m)
    try:
        for _ in range(config.budget_evals):
            x = rng.uniform(x_lower, x_upper)
            traced(x.tolist())
    except _BudgetExhausted:   # pragma: no cover  (loop stops at the budget)
        pass


def _meta_optimizer(alg: str, *, pop_size: int, epoch: int):
    """Instantiate a fresh mealpy optimiser for one repetition.

    Pop size is shared across algorithms (``meta_pop_size``); epoch is
    set high enough that the budget cuts it off first.
    """
    if alg == "ga":
        return GA.BaseGA(epoch=epoch, pop_size=pop_size)
    if alg == "pso":
        return PSO.OriginalPSO(epoch=epoch, pop_size=pop_size)
    if alg == "gwo":
        return GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)
    raise ValueError(f"unknown metaheuristic: {alg!r}")


def _run_metaheuristic(
    traced: TracedObjective,
    alg: str,
    *,
    dim: int,
    config: BenchmarkConfig,
    rep_seed: int,
) -> None:
    """Run a pure metaheuristic until the budget is exhausted.

    Mealpy's ``solve`` is given an ``epoch`` count that intentionally
    overshoots the budget — the ``BudgetExhausted`` sentinel cuts the
    loop at the exact evaluation count.
    """
    x_lower = [config.h_min_m] * dim
    x_upper = [config.h_max_m] * dim

    # epoch chosen so that pop_size * (epoch + 1) > budget by a healthy
    # margin. The exact count never matters because the budget fires first.
    epoch_cap = max(
        2,
        int(np.ceil(config.budget_evals / max(config.meta_pop_size, 1))) + 5,
    )
    optimizer = _meta_optimizer(alg, pop_size=config.meta_pop_size, epoch=epoch_cap)
    problem = {
        "obj_func": traced,
        "bounds": FloatVar(lb=x_lower, ub=x_upper),
        "minmax": "min",
        "log_to": None,
    }
    try:
        try:
            optimizer.solve(problem, seed=int(rep_seed))
        except TypeError:   # pragma: no cover  (older mealpy without seed kwarg)
            optimizer.solve(problem)
    except _BudgetExhausted:
        pass


# =============================================================================
# Public entry point
# =============================================================================
def run_benchmark(
    projeto: FundacaoProjeto,
    config: BenchmarkConfig,
    *,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> BenchmarkResult:
    """Run the head-to-head benchmark and return a typed result.

    For every (algorithm, repetition) pair the function:

    1. Builds a fresh :class:`TracedObjective` capped at
       ``config.budget_evals`` real evaluations.
    2. Delegates to the appropriate runner (``_run_ego`` or
       ``_run_metaheuristic``) with ``rep_seed = base_seed + rep``.
    3. Appends the recorded history to the global history table and
       collects the per-rep best and wall time for the summary.

    :param projeto: Validated FundacaoProjeto root aggregator
    :param config: Validated BenchmarkConfig
    :param progress: Optional callable invoked at every milestone with
                     a dict carrying ``event`` (``"benchmark.start"``,
                     ``"benchmark.rep_start"``, ``"benchmark.rep_end"``,
                     ``"benchmark.end"``, ``"benchmark.cancelled"``) plus
                     contextual fields. Errors raised by the callback
                     are swallowed so a buggy UI hook never aborts the
                     benchmark. ``None`` disables the hook
    :param should_stop: Optional zero-argument callable that returns
                        ``True`` when the user has requested
                        cancellation. Polled between repetitions

    :return: :class:`BenchmarkResult` with the unified history, the
             per-algorithm summary and the p-value matrix

    :raises RuntimeError: If no repetition completed successfully (e.g.
                          cancelled before any algorithm ran)
    :raises ValueError: When ``config.h_min_m <= projeto.cobrimento_m``
                        (candidates with non-positive punching-shear
                        effective depth would enter the search space)
    """
    if config.h_min_m <= projeto.cobrimento_m:
        raise ValueError(
            f"config.h_min_m ({config.h_min_m} m) must be strictly greater "
            f"than projeto.cobrimento_m ({projeto.cobrimento_m} m): the "
            f"punching-shear check requires a positive effective depth "
            f"d = h_z - cob for every candidate in the search space."
        )
    df_input = projeto_to_dataframe(projeto)
    dim = 3 * projeto.n_fund

    args_obj: tuple = (
        df_input,
        projeto.n_comb,
        projeto.f_ck_kpa,
        projeto.cobrimento_m,
    )
    if config.penalty is not None:
        args_obj = args_obj + (config.penalty,)

    def _emit(payload: Mapping[str, Any]) -> None:
        if progress is None:
            return
        try:
            progress(dict(payload))
        except Exception:   # pragma: no cover (UI hook must not abort)
            pass

    histories: list[pd.DataFrame] = []
    per_rep_records: list[dict[str, Any]] = []

    total_units = len(config.algorithms) * config.n_rep
    units_done = 0

    with run_context(None):
        _log.info("benchmark start",
                  extra={"event": "benchmark.start",
                         "algorithms": list(config.algorithms),
                         "n_rep": int(config.n_rep),
                         "budget_evals": int(config.budget_evals)})
        _emit({"event": "benchmark.start",
               "algorithms": list(config.algorithms),
               "n_rep": int(config.n_rep),
               "budget_evals": int(config.budget_evals),
               "total_units": int(total_units)})

        cancelled = False
        for alg in config.algorithms:
            if cancelled:
                break
            for rep in range(config.n_rep):
                if should_stop is not None and should_stop():
                    cancelled = True
                    break
                rep_seed = int(config.base_seed) + rep
                _fo_fn = (
                    avaliar_projeto_fast
                    if getattr(config, "fo_variant", "fast") == "fast"
                    else avaliar_projeto_legacy
                )
                _budget = (
                    config.ego_budget_evals
                    if alg in ("ego", "cbo")
                    else config.budget_evals
                )
                if alg == "cbo":
                    traced = _TracedComponents(
                        avaliar_projeto_componentes, args_obj,
                        budget=_budget,
                        algorithm=alg, rep=rep, seed=rep_seed,
                    )
                else:
                    traced = TracedObjective(
                        _fo_fn, args_obj,
                        budget=_budget,
                        algorithm=alg, rep=rep, seed=rep_seed,
                    )
                _emit({"event": "benchmark.rep_start",
                       "algorithm": alg, "rep": int(rep),
                       "seed": int(rep_seed),
                       "n_rep": int(config.n_rep),
                       "units_done": int(units_done),
                       "total_units": int(total_units)})
                t0 = time.perf_counter()
                try:
                    if alg == "ego":
                        _run_ego(traced, dim=dim, config=config, rep_seed=rep_seed)
                    elif alg == "cbo":
                        _run_cbo(traced, dim=dim, config=config,
                                 rep_seed=rep_seed)
                    elif alg == "random":
                        _run_random(traced, dim=dim, config=config,
                                    rep_seed=rep_seed)
                    else:
                        _run_metaheuristic(traced, alg, dim=dim, config=config,
                                           rep_seed=rep_seed)
                except _BudgetExhausted:
                    # Defensive — runners already swallow this internally,
                    # but if any inner layer re-raises we still finish
                    # the rep gracefully with whatever was recorded.
                    pass
                wall = time.perf_counter() - t0
                hist = traced.history_dataframe()
                if not hist.empty:
                    histories.append(hist)
                per_rep_records.append({
                    "algorithm": alg,
                    "rep": int(rep),
                    "seed": int(rep_seed),
                    "best": float(traced.best),
                    "n_evals": int(traced.n_evals),
                    "wall_time_s": float(wall),
                    "best_x": traced.best_x,
                    **_solution_report(traced.best_x, args_obj),
                })
                units_done += 1
                _emit({"event": "benchmark.rep_end",
                       "algorithm": alg, "rep": int(rep),
                       "seed": int(rep_seed),
                       "best": float(traced.best),
                       "n_evals": int(traced.n_evals),
                       "wall_time_s": float(wall),
                       "n_rep": int(config.n_rep),
                       "units_done": int(units_done),
                       "total_units": int(total_units)})

        if cancelled:
            _emit({"event": "benchmark.cancelled",
                   "units_done": int(units_done),
                   "total_units": int(total_units)})

    if not histories:
        raise RuntimeError(
            "run_benchmark produced no history "
            "(every repetition was cancelled before its first evaluation)."
        )

    history_df = pd.concat(histories, ignore_index=True)
    per_rep_df = pd.DataFrame(per_rep_records)
    summary_df = _build_summary(history_df, per_rep_df, config)
    pvalues_df = _build_pvalues(per_rep_df, config.algorithms)

    _emit({"event": "benchmark.end",
           "n_rows": int(len(history_df)),
           "n_algorithms": int(per_rep_df["algorithm"].nunique())})

    # Decode the best solution vector back to Sapata entities
    _best_sapatas = None
    _best_algorithm: str | None = None
    _best_of_value: float = float("inf")
    if per_rep_records:
        _best_rec = min(per_rep_records, key=lambda r: r["best"])
        _best_of_value = float(_best_rec["best"])
        _best_algorithm = str(_best_rec["algorithm"])
        _best_x_vec = _best_rec.get("best_x")
        if _best_x_vec is not None:
            try:
                _best_sapatas = design_vector_to_sapatas(_best_x_vec, projeto)
            except Exception:
                _best_sapatas = None

    return BenchmarkResult(
        history=history_df,
        summary=summary_df,
        pvalues=pvalues_df,
        config=config,
        best_sapatas=_best_sapatas,
        best_algorithm=_best_algorithm,
        best_of_value=_best_of_value,
        per_rep=per_rep_df.drop(columns=["best_x"]),
    )


# =============================================================================
# Summary / statistics
# =============================================================================
def _convergence_eval(group: pd.DataFrame, target: float, tol: float = 1e-3) -> float:
    """First evaluation index where ``of_best_so_far <= target * (1 + tol)``.

    Returns the rep's ``n_evals`` when the target was never reached, so
    a non-converged run hurts the mean rather than being silently
    dropped.
    """
    g = group.sort_values("eval_idx")
    cutoff = target * (1.0 + tol) if target > 0 else target + tol
    reached = g[g["of_best_so_far"] <= cutoff]
    if reached.empty:
        return float(g["eval_idx"].max())
    return float(reached["eval_idx"].iloc[0])


def _auc_per_rep(group: pd.DataFrame) -> float:
    """Trapezoidal AUC of the ``of_best_so_far`` curve on eval index.

    Lower is better (the curve is monotonically non-increasing). The
    AUC is normalised by the eval span so different budgets remain
    comparable.
    """
    g = group.sort_values("eval_idx")
    x = g["eval_idx"].to_numpy(dtype=float)
    y = g["of_best_so_far"].to_numpy(dtype=float)
    if x.size < 2:
        return float(y[0]) if y.size else float("nan")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def _build_summary(
    history_df: pd.DataFrame,
    per_rep_df: pd.DataFrame,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Aggregate per-algorithm statistics for the report table."""
    # Reference target = global best across every rep (so every algorithm
    # is judged against the same target).
    global_best = float(per_rep_df["best"].min())

    rows: list[dict[str, Any]] = []
    for alg in config.algorithms:
        per_rep = per_rep_df[per_rep_df["algorithm"] == alg]
        if per_rep.empty:
            continue
        bests = per_rep["best"].to_numpy(dtype=float)
        walls = per_rep["wall_time_s"].to_numpy(dtype=float)

        alg_hist = history_df[history_df["algorithm"] == alg]
        aucs: list[float] = []
        conv_evals: list[float] = []
        for rep_id, group in alg_hist.groupby("rep", sort=True):
            aucs.append(_auc_per_rep(group))
            conv_evals.append(_convergence_eval(group, target=global_best))
        aucs_arr = np.array(aucs, dtype=float) if aucs else np.array([np.nan])
        conv_arr = np.array(conv_evals, dtype=float) if conv_evals else np.array([np.nan])

        # Engineering-facing metrics: feasibility of the final designs
        # and the raw (pre-penalty) concrete volume of the feasible ones.
        feas_mask = per_rep["feasible"].to_numpy(dtype=bool) \
            if "feasible" in per_rep.columns else np.zeros(len(per_rep), dtype=bool)
        volumes = per_rep["volume_m3"].to_numpy(dtype=float) \
            if "volume_m3" in per_rep.columns else np.full(len(per_rep), np.nan)
        feas_volumes = volumes[feas_mask]
        max_viol = per_rep["max_violation"].to_numpy(dtype=float) \
            if "max_violation" in per_rep.columns else np.full(len(per_rep), np.nan)

        rows.append({
            "algorithm":         alg,
            "label":             ALGORITHM_LABELS.get(alg, alg),
            "n_rep":             int(len(per_rep)),
            "best":              float(np.min(bests)),
            "mean":              float(np.mean(bests)),
            "std":               float(np.std(bests, ddof=1)) if bests.size > 1 else 0.0,
            "median":            float(np.median(bests)),
            "feasibility_rate":  float(np.mean(feas_mask)),
            "best_feasible_volume_m3": (
                float(np.min(feas_volumes)) if feas_volumes.size else float("nan")
            ),
            "mean_max_violation": float(np.nanmean(max_viol)) if max_viol.size else float("nan"),
            "auc_mean":          float(np.nanmean(aucs_arr)),
            "auc_std":           float(np.nanstd(aucs_arr, ddof=1)) if aucs_arr.size > 1 else 0.0,
            "conv_eval_mean":    float(np.nanmean(conv_arr)),
            "conv_eval_std":     float(np.nanstd(conv_arr, ddof=1)) if conv_arr.size > 1 else 0.0,
            "wall_time_mean_s":  float(np.mean(walls)),
            "wall_time_std_s":   float(np.std(walls, ddof=1)) if walls.size > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values, preserving the input order."""
    m = len(p_values)
    adjusted = [float("nan")] * m
    finite = [(idx, float(p)) for idx, p in enumerate(p_values) if np.isfinite(p)]
    running = 0.0
    for rank, (idx, p) in enumerate(sorted(finite, key=lambda item: item[1]), start=1):
        running = max(running, (len(finite) - rank + 1) * p)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _build_pvalues(per_rep_df: pd.DataFrame, algorithms: Sequence[str]) -> pd.DataFrame:
    """Pairwise Wilcoxon-Holm p-values on per-rep best.

    The benchmark uses the same seeds across algorithms, so observations
    are paired by ``rep``. For each algorithm pair, the two-sided
    Wilcoxon signed-rank test is applied to the paired best values; the
    resulting raw p-values are then corrected with Holm's step-down
    procedure within the returned matrix. Diagonal is ``NaN``.
    """
    from scipy.stats import wilcoxon

    algs = list(algorithms)
    out = pd.DataFrame(np.nan, index=algs, columns=algs, dtype=float)
    pair_keys: list[tuple[str, str]] = []
    raw_p: list[float] = []

    for i, a in enumerate(algs[:-1]):
        a_vals = per_rep_df.loc[
            per_rep_df["algorithm"] == a, ["rep", "best"]
        ].rename(columns={"best": "best_a"})
        for b in algs[i + 1:]:
            b_vals = per_rep_df.loc[
                per_rep_df["algorithm"] == b, ["rep", "best"]
            ].rename(columns={"best": "best_b"})
            paired = a_vals.merge(b_vals, on="rep", how="inner").sort_values("rep")
            pair_keys.append((a, b))
            if paired.shape[0] < 2:
                raw_p.append(float("nan"))
                continue
            va = paired["best_a"].to_numpy(dtype=float)
            vb = paired["best_b"].to_numpy(dtype=float)
            if np.allclose(va, vb, rtol=0.0, atol=1e-15):
                raw_p.append(1.0)
                continue
            try:
                raw_p.append(float(wilcoxon(va, vb, alternative="two-sided").pvalue))
            except ValueError:
                raw_p.append(1.0)

    for (a, b), p in zip(pair_keys, _holm_adjust(raw_p)):
        out.loc[a, b] = p
        out.loc[b, a] = p
    return out


__all__ = [
    "ALL_ALGORITHMS",
    "ALGORITHM_LABELS",
    "Algorithm",
    "BenchmarkConfig",
    "BenchmarkResult",
    "FEASIBILITY_TOL",
    "TracedObjective",
    "run_benchmark",
]
