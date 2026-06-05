"""Optimisation entry point — runs EGO with best-of-N independent repetitions.

This is the function consumed by the Streamlit page and (eventually)
by the CLI/notebook clients. It encapsulates the orchestration that
used to live inline in ``pages/sapatas.py``:

    1. Build a Latin Hypercube initial population per repetition with
       ``rep_seed = base_seed + rep``.
    2. Run ``ego_01_architecture`` with the same seed propagated to the
       GPR / SciPy / mealpy stack.
    3. Keep the best objective among all ``n_rep`` repetitions.
    4. Convert the winning design vector back to a list of ``Sapata``
       entities and return an immutable ``OptimisationResult``.

Resumo em português:
    Função pura ``optimize(projeto, config)`` que orquestra LHS +
    EGO+GA+GPR ao longo de ``n_rep`` repetições independentes,
    devolvendo um ``OptimisationResult`` tipado.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional

import numpy as np
from mealpy import GA

from core.api._adapter import (
    design_vector_to_sapatas,
    projeto_to_dataframe,
)
from core.api.types import OptimisationConfig, OptimisationResult
from core.domain import FundacaoProjeto
from core.io.experiments import ExperimentRecorder
from core.observability import get_logger, run_context
from core.optimization.cache import SurrogateCache
from core.optimization import ego_01_architecture, initial_population_01
from core.optimization.ego import _CancelSentinel
from core.api.objective import avaliar_projeto_fast
from fundacao import constroi_kernel  # moved to core.optimization in Sprint 5.x

_log = get_logger("optimize")


class OptimisationCancelled(Exception):
    """Raised by :func:`optimize` when the user requested cancellation.

    The exception is the public counterpart of the internal sentinel
    used inside ``core.optimization.ego.ego_01_architecture``. As soon
    as the ``should_stop`` callable returns ``True``, the optimisation
    aborts cooperatively at the next safe point (next LHS evaluation
    or next EGO iteration); the recorder is marked as ``failed``
    (with ``error="cancelled by user"``), and this exception is
    raised so the caller can render an explicit "cancelled" message
    instead of a generic crash.
    """


def optimize(
    projeto: FundacaoProjeto,
    config: OptimisationConfig,
    *,
    recorder: Optional[ExperimentRecorder] = None,
    cache: Optional[SurrogateCache] = None,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> OptimisationResult:
    """This function runs the EGO+GPR+GA pipeline with independent repetitions.

    Every repetition produces its own LHS initial population and is
    seeded by ``config.base_seed + rep``. The best repetition (lowest
    pseudo-objective) wins and its design vector is decoded back into
    domain ``Sapata`` entities.

    Behaviour matches the legacy block of ``pages/sapatas.py`` exactly,
    so the Sprint 2 regression baseline ``of = 19.70604234767181`` is
    still reachable when the same seed and the same dataset are used.

    :param projeto: Validated FundacaoProjeto root aggregator
    :param config: OptimisationConfig with bounds, seeds and EGO/GA settings
    :param recorder: Optional :class:`core.io.experiments.ExperimentRecorder`.
                     When provided, the run is persisted as a self-describing
                     folder under the recorder's root (manifest, config, env,
                     project fingerprint, per-rep history in Parquet, summary
                     CSV, paper-grade metrics JSON, optional artifacts).
                     ``None`` keeps the historical behaviour (no disk writes)
    :param cache: Optional :class:`core.optimization.cache.SurrogateCache`.
                  When provided, the GPR fits are looked up by content hash
                  so identical (X, y, pipeline) tuples are not re-fit. ``None``
                  reproduces the historical "always refit" behaviour
    :param progress: Optional callable invoked at every milestone of the
                     pipeline with a dict carrying ``event``
                     (``"optimize.start"``, ``"optimize.rep_start"``,
                     ``"lhs.start"``, ``"lhs.eval"``, ``"lhs.done"``,
                     ``"ego.iter"``, ``"optimize.rep_end"``,
                     ``"optimize.recording"``, ``"optimize.end"``,
                     ``"optimize.failed"``, ``"optimize.cancelled"``)
                     plus contextual fields (``rep``, ``seed``,
                     ``iter``, ``n``, ``n_pop``, ``n_gen``, ``n_rep``,
                     ``of_min``, ``of_rep``, ``best_of`` …). Errors
                     raised by the callback are swallowed so a buggy
                     UI hook never aborts the optimisation. ``None``
                     disables the hook
    :param should_stop: Optional zero-argument callable that returns
                        ``True`` when the user has requested
                        cancellation. Polled per LHS evaluation and per
                        EGO iteration. The first ``True`` aborts the
                        run cooperatively, marks the recorder as
                        ``failed`` with ``error="cancelled by user"``
                        and raises :class:`OptimisationCancelled`.
                        ``None`` disables cancellation

    :return: OptimisationResult with the winning sapatas, the best
             objective, the seed that produced it and the per-rep
             trajectory of best objective values
    """
    df_input = projeto_to_dataframe(projeto)
    n_fund = projeto.n_fund
    dim = 3 * n_fund

    x_lower = [config.h_min_m] * dim
    x_upper = [config.h_max_m] * dim

    paras_opt = {
        "optimizer algorithm": GA.BaseGA(
            epoch=config.ga_epoch, pop_size=config.ga_pop_size
        )
    }
    kernel_pool = constroi_kernel()
    paras_kernel = {"kernel": kernel_pool[config.kernel_index]}

    args_obj = (
        df_input,
        projeto.n_comb,
        projeto.f_ck_kpa,
        projeto.cobrimento_m,
    )
    if config.penalty is not None:
        args_obj = args_obj + (config.penalty,)

    best_x: list[float] | None = None
    best_of: float = float("inf")
    best_seed: int = config.base_seed
    per_rep_of: list[float] = []

    if recorder is not None:
        recorder.begin(config, projeto)

    def _emit(payload: Mapping[str, Any]) -> None:
        """Forward a progress payload to the user callback, if any."""
        if progress is None:
            return
        try:
            progress(dict(payload))
        except Exception:   # pragma: no cover  (UI hook must not abort)
            pass

    run_id = recorder.run_id if recorder is not None else None
    with run_context(run_id):
        _log.info("optimize start",
                  extra={"event": "optimize.start", "n_rep": int(config.n_rep),
                         "n_pop": int(config.n_pop), "n_gen": int(config.n_gen),
                         "n_fund": int(n_fund),
                         "base_seed": int(config.base_seed)})
        _emit({"event": "optimize.start",
               "n_rep": int(config.n_rep), "n_gen": int(config.n_gen),
               "n_pop": int(config.n_pop), "n_fund": int(n_fund),
               "base_seed": int(config.base_seed)})
        t_start = time.perf_counter()
        try:
            for rep in range(config.n_rep):
                rep_seed = config.base_seed + rep
                _emit({"event": "optimize.rep_start", "rep": int(rep),
                       "seed": int(rep_seed),
                       "n_rep": int(config.n_rep), "n_gen": int(config.n_gen)})
                x_ini = initial_population_01(
                    config.n_pop, dim, x_lower, x_upper, seed=rep_seed, use_lhs=True
                )

                _log.info("rep start",
                          extra={"event": "optimize.rep_start",
                                 "rep": int(rep), "seed": int(rep_seed)})
                t0 = time.perf_counter()

                # Forward EGO-iter events with rep context attached.
                def _iter_progress(payload: Mapping[str, Any], _rep=rep,
                                   _seed=rep_seed) -> None:
                    _emit({**payload, "rep": int(_rep), "seed": int(_seed),
                           "n_rep": int(config.n_rep)})

                x_new, of_rep, history_df = ego_01_architecture(
                    avaliar_projeto_fast,
                    config.n_gen,
                    x_ini,
                    x_lower,
                    x_upper,
                    paras_opt,
                    paras_kernel,
                    args=args_obj,
                    seed=rep_seed,
                    cache=cache,
                    progress=_iter_progress,
                    should_stop=should_stop,
                )
                wall_time_s = time.perf_counter() - t0
                per_rep_of.append(float(of_rep))
                _log.info("rep end",
                          extra={"event": "optimize.rep_end",
                                 "rep": int(rep), "seed": int(rep_seed),
                                 "of_rep": float(of_rep),
                                 "wall_time_s": wall_time_s})
                _emit({"event": "optimize.rep_end",
                       "rep": int(rep), "seed": int(rep_seed),
                       "of_rep": float(of_rep),
                       "wall_time_s": float(wall_time_s),
                       "n_rep": int(config.n_rep)})

                if recorder is not None:
                    _emit({"event": "optimize.recording",
                           "rep": int(rep), "n_rep": int(config.n_rep)})
                    recorder.record_rep(
                        rep_id=rep,
                        seed=rep_seed,
                        history=history_df,
                        wall_time_s=wall_time_s,
                    )

                if of_rep < best_of:
                    best_of = float(of_rep)
                    best_x = list(map(float, x_new))
                    best_seed = rep_seed

                # Honour cancellation between repetitions even when the
                # caller disabled per-iter polling at the EGO level.
                if should_stop is not None and should_stop():
                    raise _CancelSentinel()

            if best_x is None:   # pragma: no cover  (only when config.n_rep == 0, blocked by validator)
                raise RuntimeError("optimize did not run any repetition.")

            sapatas = design_vector_to_sapatas(best_x, projeto)
            result = OptimisationResult(
                sapatas=tuple(sapatas),
                best_of=best_of,
                best_seed=best_seed,
                per_rep_of=tuple(per_rep_of),
            )
        except _CancelSentinel:
            wall = time.perf_counter() - t_start
            _log.warning(
                "optimize cancelled by user",
                extra={"event": "optimize.cancelled",
                       "wall_time_s": wall},
            )
            _emit({"event": "optimize.cancelled",
                   "wall_time_s": wall})
            if recorder is not None:
                recorder.cancel("cancelled by user")
            raise OptimisationCancelled(
                "Optimisation cancelled by user request."
            ) from None
        except BaseException as exc:
            _log.error("optimize failed",
                       extra={"event": "optimize.failed",
                              "error": repr(exc),
                              "wall_time_s": time.perf_counter() - t_start})
            _emit({"event": "optimize.failed", "error": repr(exc),
                   "wall_time_s": time.perf_counter() - t_start})
            if recorder is not None:
                recorder.cancel(repr(exc))
            raise

        if recorder is not None:
            recorder.end()
        _log.info("optimize end",
                  extra={"event": "optimize.end",
                         "best_of": float(best_of),
                         "best_seed": int(best_seed),
                         "wall_time_s": time.perf_counter() - t_start})
        _emit({"event": "optimize.end", "best_of": float(best_of),
               "best_seed": int(best_seed),
               "wall_time_s": time.perf_counter() - t_start})
        return result
