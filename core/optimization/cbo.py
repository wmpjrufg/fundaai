"""Constrained Bayesian Optimization via constrained Expected Improvement.

Implements the constrained acquisition of Gardner et al. (2014,
"Bayesian Optimization with Inequality Constraints", ICML/PMLR v32 —
PDF in ``docs/articles/05_frente_c_cbo/``): the objective (raw concrete
volume) and each aggregated constraint group are modelled by
*independent* Gaussian Process surrogates, and the next candidate
maximises

    ECI(x) = EI(x | f_min_feas) * prod_k P(g_k(x) <= 0),

where ``EI`` is the classical Expected Improvement over the best
*strictly feasible* observed volume and ``P(g_k <= 0) =
Phi(-mu_k(x) / sigma_k(x))`` is the probability of feasibility of
constraint ``k`` under its GP posterior. While no feasible point has
been observed, the acquisition reduces to the product of feasibility
probabilities alone (Gardner et al., 2014, sec. 3.2), steering the
search toward the feasible region first.

Motivation registered in the manuscript (secao 6.2/6.6): under exterior
penalisation the surrogate must reproduce the artificial penalty cliff
— aggressive factors inflate the feasible-region RMSE by five orders of
magnitude — and the linear alpha = 10 admits residual violations in the
final designs. Modelling volume and constraints separately removes the
cliff from every regression target: each GP sees a smooth physical
response.

Design notes
------------
* Constraint targets with zero observed variance (e.g. the AABB overlap
  group, identically zero in the frozen cases) carry no information for
  a GP; they are replaced by a deterministic :class:`_ConstantConstraint`
  whose feasibility probability is exactly 0 or 1.
* All surrogates share the production pipeline
  (``StandardScaler -> GaussianProcessRegressor(normalize_y=True,
  alpha=0.1)``) and kernel; ``constraint_n_restarts`` allows cheaper
  hyperparameter restarts for the (smoother) constraint targets and is
  recorded by the benchmark configuration.
* The history DataFrame mirrors ``ego_01_architecture`` (ID, ITER,
  X_*, OF, FIT, ...) with extra columns VOLUME and G_SOB/PUN/TEN/GEO,
  so downstream tooling keeps working. ``OF`` stores the penalised
  pseudo-objective Theta computed by the *same* shared numerical core
  as every other algorithm — the comparison metric stays identical.

Resumo em português:
    Otimização Bayesiana com restrições (CBO): um GP para o volume e um
    GP por grupo de restrição; aquisição = EI condicionada ao melhor
    ponto factível vezes o produto das probabilidades de factibilidade
    (Gardner et al., 2014). Sem ponto factível observado, maximiza-se
    apenas a probabilidade de factibilidade. Remove da regressão o
    "penhasco" artificial da penalização.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
import scipy as sc
import sklearn as sk
import mealpy as mp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.observability import get_logger
from core.optimization import funcs
from core.optimization.ego import _CancelSentinel

_log = get_logger("cbo")

_SIGMA_FLOOR = 1e-10
_G_COLS = ("G_SOB", "G_PUN", "G_TEN", "G_GEO")


def _expected_improvement(mu: float, sigma: float, f_min: float) -> float:
    """Classical Expected Improvement for minimisation.

    Same closed form used by the EGO acquisition (Jones et al., 1998),
    with the same numerical floor on sigma.

    :param mu: Posterior mean of the objective at the candidate
    :param sigma: Posterior standard deviation at the candidate
    :param f_min: Best (feasible) objective value observed so far

    :return: Expected improvement (non-negative)
    """
    sigma = sigma if sigma >= _SIGMA_FLOOR else _SIGMA_FLOOR
    z = (f_min - mu) / sigma
    return float((f_min - mu) * sc.stats.norm.cdf(z)
                 + sigma * sc.stats.norm.pdf(z))


def _prob_feasibility(mu: float, sigma: float) -> float:
    """Probability that a constraint is satisfied, P(g <= 0), under a GP posterior.

    :param mu: Posterior mean of the constraint at the candidate
    :param sigma: Posterior standard deviation at the candidate

    :return: Phi(-mu / sigma) in [0, 1]
    """
    sigma = sigma if sigma >= _SIGMA_FLOOR else _SIGMA_FLOOR
    return float(sc.stats.norm.cdf(-mu / sigma))


class _ConstantConstraint:
    """Deterministic stand-in for a constraint target with zero variance.

    A GP fitted on a constant column carries no information (and the
    normalised target is degenerate), so the group is treated as the
    deterministic value it is: feasibility probability exactly 1 when
    the constant satisfies ``g <= 0`` and exactly 0 otherwise.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def prob_feasible(self, _x_row: np.ndarray) -> float:
        """Return the degenerate feasibility probability.

        :param _x_row: Ignored (kept for interface symmetry)
        :return: 1.0 when ``value <= 0``, else 0.0
        """
        return 1.0 if self.value <= 0.0 else 0.0


class _GPConstraint:
    """GP-backed constraint wrapper exposing the feasibility probability.

    :param model: Fitted ``Pipeline(StandardScaler, GaussianProcessRegressor)``
    """

    def __init__(self, model: Pipeline) -> None:
        self.model = model

    def prob_feasible(self, x_row: np.ndarray) -> float:
        """Posterior probability of ``g(x) <= 0`` at one candidate.

        :param x_row: Candidate of shape (1, d)
        :return: Phi(-mu / sigma)
        """
        mu, sig = self.model.predict(x_row, return_std=True)
        return _prob_feasibility(float(mu[0]), float(sig[0]))


def _make_pipeline(kernel, n_restarts: int, random_state: int) -> Pipeline:
    """Build the production surrogate pipeline for one target.

    :param kernel: sklearn kernel instance (cloned internally by the GPR)
    :param n_restarts: ``n_restarts_optimizer`` for the marginal-likelihood fit
    :param random_state: Seed of the GPR restarts

    :return: Unfitted ``Pipeline(StandardScaler, GaussianProcessRegressor)``
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=0.1,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
        )),
    ])


def cbo_01_architecture(
    obj_components: Callable,
    n_gen: int,
    initial_population: list,
    x_lower: list,
    x_upper: list,
    params_opt: dict,
    params_kernel: Optional[dict] = None,
    args: Optional[tuple] = None,
    seed: Optional[int] = None,
    constraint_n_restarts: int = 5,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> tuple[list, float, pd.DataFrame]:
    """Run the constrained-EI Bayesian optimisation loop (Gardner et al., 2014).

    Interface mirrors :func:`core.optimization.ego.ego_01_architecture`
    with one difference: ``obj_components(x, args) -> (theta, volume, g)``
    must return the penalised scalar (for comparable tracking), the raw
    volume (regression target of the objective GP) and the aggregated
    constraint vector ``g`` of shape (4,) (one regression target each).

    :param obj_components: Component evaluator, e.g.
                           ``core.api.objective.avaliar_projeto_componentes``
    :param n_gen: Number of CBO iterations beyond the initial population
    :param initial_population: Initial sample, list with shape (n_pop, d)
    :param x_lower: Lower bounds of the design variables
    :param x_upper: Upper bounds of the design variables
    :param params_opt: Inner optimiser of the acquisition — same contract
                       as the EGO architecture ('scipy_*' string or a
                       mealpy optimizer instance under
                       ``'optimizer algorithm'``)
    :param params_kernel: Kernel for every GP (``{'kernel': ...}``);
                          RBF when ``None``
    :param args: Extra arguments forwarded to ``obj_components``
    :param seed: Seed propagated to the GPs, to mealpy (``seed + t``)
                 and to the SciPy starting points
    :param constraint_n_restarts: ``n_restarts_optimizer`` of the
                                  constraint GPs (the objective GP keeps
                                  5, as in production). Recorded by the
                                  benchmark configuration
    :param progress: Optional milestone callback (events ``lhs.start``,
                     ``lhs.eval``, ``lhs.done``, ``cbo.iter``); errors
                     raised by the callback are swallowed
    :param should_stop: Optional cancellation poll, as in the EGO loop

    :return: [0] best design vector (by penalised Theta, comparable to
             every other algorithm), [1] its Theta, [2] history DataFrame
             with ID, ITER, X_*, OF, VOLUME, G_SOB..G_GEO, FIT,
             OF EVALUATIONS and TIME CONSUMPTION (s)
    """
    x_t0 = [list(map(float, row)) for row in initial_population]
    d = len(x_t0[0])
    n_pop = len(x_t0)

    rng = np.random.default_rng(seed)
    gpr_random_state = 42 if seed is None else int(seed)

    if params_kernel is not None:
        kernel = params_kernel["kernel"]
    else:
        kernel = sk.gaussian_process.kernels.RBF()

    vol_pipe = _make_pipeline(kernel, n_restarts=5, random_state=gpr_random_state)
    g_pipes = [
        _make_pipeline(kernel, n_restarts=int(constraint_n_restarts),
                       random_state=gpr_random_state)
        for _ in _G_COLS
    ]

    def _emit(payload: Mapping[str, Any]) -> None:
        if progress is None:
            return
        try:
            progress(dict(payload))
        except Exception:   # pragma: no cover  (UI hook must not abort)
            pass

    def _evaluate_row(idx: int, x_vec: list, t: int) -> dict:
        t0 = time.perf_counter()
        theta, volume, g = (obj_components(x_vec, args)
                            if args is not None else obj_components(x_vec))
        row = {
            "ID": idx, "ITER": t,
            **{f"X_{j}": float(v) for j, v in enumerate(x_vec)},
            "OF": float(theta),
            "VOLUME": float(volume),
            **{col: float(gv) for col, gv in zip(_G_COLS, g)},
            "FIT": funcs.fit_value(float(theta)),
            "OF EVALUATIONS": 1,
            "TIME CONSUMPTION (s)": time.perf_counter() - t0,
        }
        return row

    # ------------------------------------------------------------------
    # Initial sample
    # ------------------------------------------------------------------
    _emit({"event": "lhs.start", "n_pop": int(n_pop)})
    rows = []
    for i, x_vec in enumerate(x_t0):
        if should_stop is not None and should_stop():
            raise _CancelSentinel()
        rows.append(_evaluate_row(i, x_vec, 0))
        if (i + 1) % 10 == 0 or i == n_pop - 1:
            _emit({"event": "lhs.eval", "n": int(i + 1), "n_pop": int(n_pop)})
    df = pd.DataFrame(rows)
    x_cols = [c for c in df.columns if c.startswith("X_")]
    _emit({"event": "lhs.done", "n_pop": int(n_pop),
           "of_min": float(df["OF"].min())})

    # ------------------------------------------------------------------
    # CBO iterations
    # ------------------------------------------------------------------
    for t in range(1, n_gen + 1):
        if should_stop is not None and should_stop():
            raise _CancelSentinel()

        x_train = df[x_cols].to_numpy(dtype=np.float64)
        y_vol = df["VOLUME"].to_numpy(dtype=np.float64)
        g_matrix = df[list(_G_COLS)].to_numpy(dtype=np.float64)

        vol_model = vol_pipe.fit(x_train, y_vol)

        constraints = []
        for k, pipe in enumerate(g_pipes):
            y_k = g_matrix[:, k]
            if float(np.std(y_k)) == 0.0:
                constraints.append(_ConstantConstraint(float(y_k[0])))
            else:
                constraints.append(_GPConstraint(pipe.fit(x_train, y_k)))

        feas_mask = (g_matrix <= 0.0).all(axis=1)
        has_feasible = bool(feas_mask.any())
        f_min_feas = float(y_vol[feas_mask].min()) if has_feasible else np.inf
        n_feas = int(feas_mask.sum())

        _log.debug("cbo iteration",
                   extra={"event": "cbo.iter", "iter": int(t),
                          "of_min": float(df["OF"].min()),
                          "n_train": int(len(df)), "n_feas": n_feas})
        _emit({"event": "cbo.iter", "iter": int(t), "n_gen": int(n_gen),
               "of_min": float(df["OF"].min()),
               "n_train": int(len(df)), "n_feas": n_feas})

        def acq_neg(x, _coef=None) -> float:
            """Negative constrained acquisition at one candidate.

            :param x: Candidate design vector
            :param _coef: Unused (partial-application symmetry)
            :return: ``-(EI * prod PoF)`` — or ``-prod PoF`` while no
                     feasible point has been observed
            """
            x_row = np.asarray(x, dtype=np.float64).reshape(1, -1)
            pof = 1.0
            for c in constraints:
                pof *= c.prob_feasible(x_row)
                if pof == 0.0:
                    break
            if not has_feasible:
                return -pof
            mu_v, sig_v = vol_model.predict(x_row, return_std=True)
            ei = _expected_improvement(float(mu_v[0]), float(sig_v[0]),
                                       f_min_feas)
            return -(ei * pof)

        wrapped_obj = partial(acq_neg, _coef=None)
        opt = params_opt["optimizer algorithm"]

        if isinstance(opt, str) and opt.lower().startswith("scipy"):
            method = {"scipy_lbfgs": "L-BFGS-B", "scipy_tnc": "TNC",
                      "scipy_slsqp": "SLSQP",
                      "scipy_trust": "trust-constr"}[opt.lower()]
            bounds = list(zip(x_lower, x_upper))
            x0 = (rng.uniform(x_lower, x_upper) if seed is not None
                  else np.random.uniform(x_lower, x_upper))
            res = sc.optimize.minimize(wrapped_obj, x0, method=method,
                                       bounds=bounds,
                                       options={"maxiter": 300, "ftol": 1e-5})
            x_new = res.x.tolist()
        else:
            problem_dict = {
                "obj_func": wrapped_obj,
                "bounds": mp.FloatVar(lb=x_lower, ub=x_upper),
                "minmax": "min",
                "log_to": None,
            }
            if seed is not None:
                try:
                    g_best = opt.solve(problem_dict, seed=int(seed) + t)
                except TypeError:   # pragma: no cover (older mealpy)
                    g_best = opt.solve(problem_dict)
            else:
                g_best = opt.solve(problem_dict)
            x_new = list(g_best.solution)

        new_id = int(df["ID"].max()) + 1
        df = pd.concat([df, pd.DataFrame([_evaluate_row(new_id, x_new, t)])],
                       ignore_index=True)

    idx_min = df["OF"].idxmin()
    best_x = df.loc[idx_min, x_cols].tolist()
    return best_x, float(df["OF"].min()), df
