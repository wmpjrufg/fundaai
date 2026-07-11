"""Frente C — CBO (constrained EI) under the frozen S1 protocol.

Runs the ``cbo`` algorithm alone on the three frozen cases with the
exact levers of the S1 scenario of ``run_final_benchmark.py`` — same
budget of 150 real evaluations, same paired seeds (42–71), same LHS
size (10 d), same inner-EI genetic algorithm and same production
kernel — so the results are directly comparable (and statistically
pairable) with the persisted EGO/GA/PSO/GWO/random runs without
re-executing them.

Outputs, mirroring the main protocol layout:

    experiments/protocolo_final/<caso>/S1_cbo/
        history.parquet  per_rep.csv  summary.csv  pvalues.csv
        config.json      meta.json

Usage (from the repository root):

    .venv/bin/python scripts/run_cbo_benchmark.py

Resumo em português:
    Executa o CBO (aquisição ECI de Gardner et al., 2014) sob o mesmo
    protocolo S1 congelado, com as mesmas 30 seeds pareadas, e persiste
    os artefatos ao lado dos demais algoritmos para comparação direta.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.api import BenchmarkConfig, run_benchmark  # noqa: E402
from core.io import read_projeto_from_excel  # noqa: E402
from run_final_benchmark import (  # noqa: E402  (frozen protocol levers)
    BASE_SEED, BUDGET_S1, CASES, COBRIMENTO_M, F_CK_KPA, GA_EPOCH,
    GA_POP_SIZE, H_MAX_M, H_MIN_M, KERNEL_INDEX, LHS_FACTOR, META_POP_SIZE,
    N_REP, OUT_ROOT, _env_snapshot, _persist, _progress,
)

CBO_CONSTRAINT_RESTARTS = 5   # pilot showed the cheaper setting saves <10%


def main() -> None:
    """Run CBO on the three frozen cases and persist everything.

    :return: None
    """
    t_all = time.perf_counter()
    for case_name, rel_path in CASES.items():
        projeto = read_projeto_from_excel(
            REPO_ROOT / rel_path, f_ck_kpa=F_CK_KPA, cobrimento_m=COBRIMENTO_M
        )
        dim = 3 * projeto.n_fund
        cfg = BenchmarkConfig(
            algorithms=("cbo",),
            budget_evals=BUDGET_S1,
            ego_budget_evals=BUDGET_S1,
            n_rep=N_REP,
            base_seed=BASE_SEED,
            h_min_m=H_MIN_M,
            h_max_m=H_MAX_M,
            lhs_n_pop=LHS_FACTOR * dim,
            meta_pop_size=META_POP_SIZE,
            kernel_index=KERNEL_INDEX,
            ga_pop_size=GA_POP_SIZE,
            ga_epoch=GA_EPOCH,
            cbo_constraint_restarts=CBO_CONSTRAINT_RESTARTS,
        )
        print(f"[{case_name}] CBO S1: {BUDGET_S1} evals × {N_REP} reps "
              f"(dim={dim}, lhs={LHS_FACTOR * dim}) ...", flush=True)
        t0 = time.perf_counter()
        res = run_benchmark(projeto, cfg, progress=_progress)
        wall = time.perf_counter() - t0
        _persist(res, OUT_ROOT / case_name / "S1_cbo", wall)
        print(f"[{case_name}] CBO done in {wall/60:.1f} min", flush=True)
    print(f"CBO PROTOCOL COMPLETE in {(time.perf_counter() - t_all)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
