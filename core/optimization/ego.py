"""Efficient Global Optimization (EGO) related functions."""
from typing import Any, Callable, Mapping, Optional
from functools import partial

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy as sc
import mealpy as mp

from core.observability import get_logger
from core.optimization import funcs
from core.optimization.cache import SurrogateCache, fit_or_get_cached

_log = get_logger("ego")


class _CancelSentinel(BaseException):
    """Internal sentinel raised when ``should_stop()`` returns True.

    Inherits from ``BaseException`` (not ``Exception``) so it bypasses
    blanket ``except Exception`` clauses that might otherwise eat it
    (e.g. inside the SciPy / mealpy inner optimisers).
    """


def ego_01_architecture(obj: Callable, n_gen: int, initial_population: list, x_lower: list, x_upper: list, params_opt: dict, params_kernel: Optional[dict] = None, args: Optional[tuple] = None, seed: Optional[int] = None, cache: Optional[SurrogateCache] = None, progress: Optional[Callable[[Mapping[str, Any]], None]] = None, should_stop: Optional[Callable[[], bool]] = None) -> tuple[list, float, pd.DataFrame]:
    """This function performs the hybrid Efficient Global Optimization (EGO) loop.

    Em cada iteração ajusta um modelo substituto Gaussian Process Regressor
    aos pares (x, OF) coletados, maximiza a função de aquisição Expected
    Improvement por meio de um otimizador interno (SciPy ou Mealpy) e
    avalia o ponto candidato na função objetivo real, atualizando a base
    sequencialmente.

    :param obj: Objective function `obj(x, args) -> float` (or `obj(x) -> float` when args is None)
    :param n_gen: Number of EGO iterations beyond the initial population
    :param initial_population: Initial population, list with shape (n_pop, d)
    :param x_lower: Lower limit of the design variables
    :param x_upper: Upper limit of the design variables
    :param params_opt: Internal optimizer configuration. Strings 'scipy_lbfgs', 'scipy_tnc', 'scipy_slsqp', 'scipy_trust' or any mealpy algorithm instance
    :param params_kernel: Kernel configuration for the Gaussian Process Regressor (optional). Defaults to RBF when None
    :param args: Extra arguments forwarded to the objective function (optional)
    :param seed: Random seed propagated to the GPR (`random_state`), to NumPy (initial points of SciPy minimizers) and to mealpy via `seed=seed`. Default `None` keeps the historical behaviour (`random_state=42` in the GPR; non-deterministic SciPy x0)
    :param cache: Optional :class:`core.optimization.cache.SurrogateCache`. When provided, the GPR is fit through :func:`fit_or_get_cached` so identical (X, y, pipeline) tuples are reused across replications, notebook re-runs and batch experiments instead of being refit from scratch. Default `None` keeps the historical behaviour (always refit)
    :param progress: Optional callable invoked at every milestone with a dict carrying ``event`` (``"lhs.start"``, ``"lhs.eval"``, ``"lhs.done"``, ``"ego.iter"``) plus contextual fields (``iter``, ``n_gen``, ``of_min``, ``n_train``, ...). Errors raised by the callback are intentionally swallowed so a buggy UI hook does not abort the optimisation. Default ``None`` disables the hook
    :param should_stop: Optional zero-argument callable returning ``True`` when the user has requested cancellation. Polled at the start of every EGO iteration and after every LHS evaluation; on the first ``True`` the function raises an internal sentinel that the caller (typically :func:`core.api.optimize`) translates into :class:`core.api.OptimisationCancelled`. ``None`` disables cancellation

    :return: [0] = Best solution found, list with shape (d,) [best_x]
             [1] = Best objective function value [best_of]
             [2] = DataFrame with the full optimisation history. Columns include
                   ID, ITER, X_0..X_{d-1}, OF, FIT, OF EVALUATIONS and
                   TIME CONSUMPTION. Each row of the initial sample has
                   ITER=0; each iteration `t` of the EGO appends one row
                   with ITER=t and a fresh ID = max(ID)+1 [df_history]

    Example 1: Using SciPy (SLSQP) as optimizer algorithm and RBF kernel from sklearn
        >>> from sklearn.gaussian_process.kernels import RBF
        >>> from function import f
        >>> 
        >>> # Function in python file (function.py)
        >>> def f(x):
        >>>     of = (x[0] - 3.5) * np.sin((x[0] - 3.5) / (np.pi))
        >>>     return of
        >>>
        >>> x_ini = [[0.0], [4.5], [7.0], [10.0], [15.0], [20.0], [25.0]]
        >>> paras_opt = {'optimizer algorithm': 'scipy_slsqp'}
        >>> paras_kernel = {'kernel': RBF()}
        >>> 
        >>> x_new, best_of, df = ego_01_architecture(obj=f, n_gen=30, initial_population=x_ini, x_lower=[-10.0], x_upper=[25.0], params_opt=paras_opt, params_kernel=paras_kernel)
        >>> print(f"Best solution: {x_new} -> OF: {best_of}")

    Example 2: Using SciPy (TNC) as optimizer algorithm and RBF kernel from sklearn
        >>> from sklearn.gaussian_process.kernels import RBF
        >>> from function import f
        >>> 
        >>> # Function in python file (function.py)
        >>> def f(x):
        >>>     of = (x[0] - 3.5) * np.sin((x[0] - 3.5) / (np.pi))
        >>>     return of
        >>>
        >>> x_ini = [[0.0], [4.5], [7.0], [10.0], [15.0], [20.0], [25.0]]
        >>> paras_opt = {'optimizer algorithm': 'scipy_tnc'}
        >>> paras_kernel = {'kernel': RBF()}
        >>> 
        >>> x_new, best_of, df = ego_01_architecture(obj=f, n_gen=30, initial_population=x_ini, x_lower=[-10.0], x_upper=[25.0], params_opt=paras_opt, params_kernel=paras_kernel)
        >>> print(f"Best solution: {x_new} -> OF: {best_of}")
    
    Example 3: Using Mealpy – Genetic Algorithm (GA)
        >>> from sklearn.gaussian_process.kernels import RBF
        >>> from mealpy import GA
        >>> from function import f
        >>> 
        >>> # Function in python file (function.py)
        >>> def f(x):
        >>>     of = (x[0] - 3.5) * np.sin((x[0] - 3.5) / (np.pi))
        >>>     return of
        >>>
        >>> x_ini = [[0.0], [4.5], [7.0], [10.0], [15.0], [20.0], [25.0]]
        >>> paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=40, pop_size=50)}
        >>> # You can improve the GA parameters. Use this documentation for that: https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.evolutionary_based.html#module-mealpy.evolutionary_based.GA
        >>> paras_kernel = {'kernel': RBF()}
        >>> 
        >>> x_new, best_of, df = ego_01_architecture(obj=f, n_gen=30, initial_population=x_ini, x_lower=[-10.0], x_upper=[25.0], params_opt=paras_opt, params_kernel=paras_kernel)
        >>> print(f"Best solution: {x_new} -> OF: {best_of}")
    
    Example 4: Using Mealpy – Particle Swarm Optimization (PSO)
        >>> from sklearn.gaussian_process.kernels import RBF
        >>> from mealpy import PSO
        >>> from function import f
        >>> 
        >>> # Function in python file (function.py)
        >>> def f(x):
        >>>     of = (x[0] - 3.5) * np.sin((x[0] - 3.5) / (np.pi))
        >>>     return of
        >>>
        >>> x_ini = [[0.0], [4.5], [7.0], [10.0], [15.0], [20.0], [25.0]]
        >>> paras_opt = {'optimizer algorithm': PSO.AIW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, alpha=0.4)}
        >>> # You can improve the PSO parameters. Use this documentation for that: https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.swarm_based.html#module-mealpy.swarm_based.PSO
        >>> paras_kernel = {'kernel': RBF()}
        >>> 
        >>> x_new, best_of, df = ego_01_architecture(obj=f, n_gen=30, initial_population=x_ini, x_lower=[-10.0], x_upper=[25.0], params_opt=paras_opt, params_kernel=paras_kernel)
        >>> print(f"Best solution: {x_new} -> OF: {best_of}")
    """

    # Initialize variables and dataframes (Don't remove this part)
    x_t0 = initial_population.copy()
    d = len(x_t0[0])
    n_pop = len(x_t0)
    all_results = []

    # Random seed propagation. None preserves the historical default
    # (random_state=42 hardcoded in the GPR; non-deterministic SciPy x0).
    rng = np.random.default_rng(seed)
    gpr_random_state = 42 if seed is None else int(seed)

    # GPR organization and optimization loop
    sca = ("scaler", StandardScaler())
    if params_kernel is not None:
        kernel = params_kernel['kernel']
    else:
        kernel = sk.gaussian_process.kernels.RBF()
    gp = ("gp", GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=0.1,
        n_restarts_optimizer=5,
        random_state=gpr_random_state,
    ))
    pipe = Pipeline([sca, gp])

    # Initial population evaluation (Don't remove this part).
    # Each row receives ID = n and ITER = 0.
    if progress is not None:
        try:
            progress({"event": "lhs.start", "n_pop": int(n_pop)})
        except Exception:   # pragma: no cover  (UI hook must not abort)
            pass
    for n in range(n_pop):
        if should_stop is not None and should_stop():
            raise _CancelSentinel()
        aux_df = funcs.evaluation(obj, n, x_t0[n], 0, args=args) if args is not None else funcs.evaluation(obj, n, x_t0[n], 0)
        all_results.append(aux_df)
        # Emit only every 10 evaluations (or the last one) so a 250-pop
        # LHS does not flood the queue with quasi-instantaneous events.
        if progress is not None and (n + 1) % 10 == 0 or n == n_pop - 1:
            try:
                progress({"event": "lhs.eval",
                          "n": int(n + 1), "n_pop": int(n_pop)})
            except Exception:   # pragma: no cover
                pass
    df = pd.concat(all_results, ignore_index=True)
    x_cols = [col for col in df.columns if col.startswith("X_")]
    if progress is not None:
        try:
            progress({"event": "lhs.done", "n_pop": int(n_pop),
                      "of_min": float(df["OF"].min())})
        except Exception:   # pragma: no cover
            pass

    # Iterations
    for t in range(1, n_gen + 1):
        if should_stop is not None and should_stop():
            raise _CancelSentinel()
        # Training the surrogate model. When `cache` is provided, identical
        # (X, y, pipeline) tuples short-circuit the kernel-hyperparameter
        # optimisation (the dominant cost of `fit`); otherwise behaviour is
        # the historical "always refit".
        x_train = df[x_cols]
        y_train = df[['OF']]
        model = fit_or_get_cached(pipe, x_train, y_train, cache)
        of_min_now = float(df["OF"].min())
        n_train_now = int(len(df))
        _log.debug(
            "ego iteration",
            extra={"event": "ego.iter", "iter": int(t),
                   "of_min": of_min_now, "n_train": n_train_now},
        )
        if progress is not None:
            try:
                progress({
                    "event": "ego.iter",
                    "iter": int(t),
                    "n_gen": int(n_gen),
                    "of_min": of_min_now,
                    "n_train": n_train_now,
                })
            except Exception:   # pragma: no cover  (UI hook must not abort)
                pass

        # Acquisition function: maximise Expected Improvement (EI)
        argss = (model, df['OF'].min())

        def obj_ego(x, coef):
            model, fmin = coef
            x_df = pd.DataFrame([x], columns=model.feature_names_in_)
            mu, sig = model.predict(x_df, return_std=True)
            sigma = sig[0] if sig[0] >= 1e-10 else 1e-10
            z = (fmin - mu[0]) / sigma
            of = (fmin - mu[0]) * sc.stats.norm.cdf(z) + sigma * sc.stats.norm.pdf(z)
            return -of

        wrapped_obj = partial(obj_ego, coef=argss)
        opt = params_opt["optimizer algorithm"]

        if isinstance(opt, str) and opt.lower().startswith("scipy"):
            if opt.lower() == "scipy_lbfgs":
                method = "L-BFGS-B"
            elif opt.lower() == "scipy_tnc":
                method = "TNC"
            elif opt.lower() == "scipy_slsqp":
                method = "SLSQP"
            elif opt.lower() == "scipy_trust":
                method = "trust-constr"
            bounds = list(zip(x_lower, x_upper))
            # Use the seeded RNG when seed is provided, otherwise the global numpy state
            x0 = rng.uniform(x_lower, x_upper) if seed is not None else np.random.uniform(x_lower, x_upper)
            res = sc.optimize.minimize(wrapped_obj, x0, method=method, bounds=bounds, options={"maxiter": 300, "ftol": 1e-5})
            x_new = res.x.tolist()
        else:
            problem_dict = {
                "obj_func": wrapped_obj,
                "bounds": mp.FloatVar(lb=x_lower, ub=x_upper),
                "minmax": "min",
                "log_to": None,
            }
            optimizer = params_opt["optimizer algorithm"]
            if seed is not None:
                # mealpy expone seed via solve(...) na maioria das versoes
                try:
                    g_best = optimizer.solve(problem_dict, seed=int(seed) + t)
                except TypeError:
                    g_best = optimizer.solve(problem_dict)
            else:
                g_best = optimizer.solve(problem_dict)
            x_new = g_best.solution

        # Add new training point with correct ITER=t and a fresh ID.
        # Antes desta correcao todos os pontos novos eram registrados com
        # ITER=0 e ID herdado do ultimo indice da populacao inicial, o que
        # corrompia o historico do EGO (ver issue: Historico do EGO com
        # ITER e ID incorretos).
        new_id = int(df['ID'].max()) + 1
        aux_df = funcs.evaluation(obj, new_id, x_new, t, args=args) if args is not None else funcs.evaluation(obj, new_id, x_new, t)
        df = pd.concat([df, aux_df], ignore_index=True)

    # Best solution extraction
    x_cols = [col for col in df.columns if col.startswith("X_")]
    idx_min = df["OF"].idxmin()
    best_x = df.loc[idx_min, x_cols].tolist()

    return best_x, df['OF'].min(), df
