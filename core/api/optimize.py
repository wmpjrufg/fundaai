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

import numpy as np
from mealpy import GA

from core.api._adapter import (
    design_vector_to_sapatas,
    projeto_to_dataframe,
)
from core.api.types import OptimisationConfig, OptimisationResult
from core.domain import FundacaoProjeto
from fundacao import constroi_kernel, obj_felipe_lucas
from metapy_toolbox import ego_01_architecture, initial_population_01


def optimize(
    projeto: FundacaoProjeto,
    config: OptimisationConfig,
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

    for rep in range(config.n_rep):
        rep_seed = config.base_seed + rep
        x_ini = initial_population_01(
            config.n_pop, dim, x_lower, x_upper, seed=rep_seed, use_lhs=True
        )

        x_new, of_rep, _ = ego_01_architecture(
            obj_felipe_lucas,
            config.n_gen,
            x_ini,
            x_lower,
            x_upper,
            paras_opt,
            paras_kernel,
            args=args_obj,
            seed=rep_seed,
        )
        per_rep_of.append(float(of_rep))

        if of_rep < best_of:
            best_of = float(of_rep)
            best_x = list(map(float, x_new))
            best_seed = rep_seed

    if best_x is None:   # pragma: no cover  (only when config.n_rep == 0, blocked by validator)
        raise RuntimeError("optimize did not run any repetition.")

    sapatas = design_vector_to_sapatas(best_x, projeto)
    return OptimisationResult(
        sapatas=tuple(sapatas),
        best_of=best_of,
        best_seed=best_seed,
        per_rep_of=tuple(per_rep_of),
    )
