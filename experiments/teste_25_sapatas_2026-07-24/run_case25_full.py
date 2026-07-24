"""Full 25-footing scaling probe (dim=75).

Regime: fixed, well-spaced 5x5 grid (6 m spacing) -> non-overlap inactive by
construction; same regime as the frozen article-1 cases, scaled to N=25.

Loads are correlated to soil capacity so a healthy feasible region exists.

Produces:
  * S1 equal-budget comparison: ego, cbo, ga, pso, gwo, random
  * S2 extended-budget for the cheap baselines (wall-clock regime)
  * per-footing DE decomposition reference (near-true optimum, since separable)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

REPO = Path("/Users/lucasteixeira/Documents/Iniciacao_cientifica/fundaIA")
sys.path.insert(0, str(REPO))
SCRATCH = Path(__file__).resolve().parent

F_CK_KPA = 25_000.0
COBRIMENTO_M = 0.04
H_MIN, H_MAX = 0.60, 3.00
N = 25
BASE_SEED = 42
N_REP = 3
BUDGET_S1 = 250          # equal budget (real evals) for every algorithm
EGO_BUDGET = 250
LHS_NPOP = 120           # 1.6*dim; 10*dim=750 is infeasible under this budget
BUDGET_S2 = 5000         # extended budget for cheap baselines only
CBO_RESTARTS = 2         # constraint-GP restarts (cut from 5 to keep runtime sane)

FEAS_PENALTY_LINEAR = 1e6
FEAS_PENALTY_QUAD = 1e8
DE_MAXITER = 220
DE_POPSIZE = 20


def build_df(n: int, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    spacing = 6.0
    rows = []
    for i in range(n):
        gx = (i % side) * spacing
        gy = (i // side) * spacing
        ap = float(rng.uniform(0.20, 0.50))
        bp = float(rng.uniform(0.20, 0.50))
        if rng.random() < 0.3:
            solo, div = "areia", 40.0
        else:
            solo, div = "argila", 50.0
        spt = float(rng.integers(20, 45))
        sigma_adm = spt / div * 1e3                     # kPa
        # required axial area target in [1.2, 2.2] m^2 -> load scaled to soil
        area_target = float(rng.uniform(1.2, 2.2))
        base_fz = sigma_adm * area_target               # kN
        row = {
            "Elemento": f"P{i+1:02d}", "ap (m)": round(ap, 3), "bp (m)": round(bp, 3),
            "spt": spt, "solo": solo, "xg (m)": round(gx, 3), "yg (m)": round(gy, 3),
        }
        for c in (1, 2, 3):
            fz = base_fz * float(rng.uniform(0.9, 1.1))
            row[f"Fz-c{c}"] = round(fz, 2)
            row[f"Mx-c{c}"] = round(float(rng.uniform(-20.0, 20.0)), 2)
            row[f"My-c{c}"] = round(float(rng.uniform(-20.0, 20.0)), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def decomposition_reference(projeto, args_full) -> dict:
    """Per-footing feasibility-first DE, reassembled and re-evaluated globally."""
    from core.api._adapter import projeto_to_dataframe
    from core.api.objective import avaliar_projeto_componentes

    df_full = projeto_to_dataframe(projeto)
    ncomb, fck, cob = projeto.n_comb, projeto.f_ck_kpa, projeto.cobrimento_m

    def one_obj(z, df1):
        a = (df1, ncomb, fck, cob)
        try:
            _t, vol, g = avaliar_projeto_componentes(list(z), a)
        except ValueError:
            return float("inf")
        v = np.clip(g, 0.0, None)
        return float(vol + FEAS_PENALTY_LINEAR * v.sum() + FEAS_PENALTY_QUAD * np.square(v).sum())

    xs = []
    t0 = time.perf_counter()
    for i in range(len(df_full)):
        df1 = df_full.iloc[[i]].reset_index(drop=True)
        r = differential_evolution(
            lambda z: one_obj(z, df1), bounds=[(H_MIN, H_MAX)] * 3,
            seed=1000 + i, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
            tol=1e-8, atol=0.0, polish=True, updating="immediate", workers=1,
        )
        xs.extend(np.asarray(r.x, float).tolist())
    wall = time.perf_counter() - t0
    theta, vol, g = avaliar_projeto_componentes(xs, args_full)
    return {
        "theta": float(theta), "volume_m3": float(vol),
        "max_violation": float(g.max()), "feasible": bool((g <= 1e-9).all()),
        "wall_time_s": wall, "g": np.round(g, 6).tolist(),
    }


def main() -> None:
    from core.io import read_projeto_from_excel
    from core.api._adapter import projeto_to_dataframe
    from core.api.benchmark import BenchmarkConfig, run_benchmark

    df = build_df(N)
    xlsx = SCRATCH / f"caso_{N}_v2.xlsx"
    df.to_excel(xlsx, index=False)
    projeto = read_projeto_from_excel(xlsx, f_ck_kpa=F_CK_KPA, cobrimento_m=COBRIMENTO_M)
    dim = 3 * projeto.n_fund
    df_in = projeto_to_dataframe(projeto)
    args = (df_in, projeto.n_comb, projeto.f_ck_kpa, projeto.cobrimento_m)
    print(f"=== CASE: n_fund={projeto.n_fund} n_comb={projeto.n_comb} dim={dim} ===", flush=True)

    # feasibility sanity at max box
    from core.api.objective import avaliar_projeto_componentes
    _t, _v, g3 = avaliar_projeto_componentes(np.full(dim, 3.0).tolist(), args)
    print(f"max-box feasible={bool((g3<=0).all())} g={np.round(g3,4)}", flush=True)

    def prog(ev):
        if ev.get("event") == "benchmark.rep_end":
            print(f"    {ev['algorithm']:>6} rep {ev['rep']+1} best={ev['best']:.3f} "
                  f"t={ev['wall_time_s']:.1f}s", flush=True)

    # --- S1: equal budget, all algorithms
    cfg1 = BenchmarkConfig(
        algorithms=("ego", "cbo", "ga", "pso", "gwo", "random"),
        budget_evals=BUDGET_S1, ego_budget_evals=EGO_BUDGET,
        n_rep=N_REP, base_seed=BASE_SEED, h_min_m=H_MIN, h_max_m=H_MAX,
        lhs_n_pop=LHS_NPOP, meta_pop_size=30, kernel_index=-1,
        ga_pop_size=50, ga_epoch=30, cbo_constraint_restarts=CBO_RESTARTS,
    )
    print(f"\n[S1] equal budget={BUDGET_S1}, n_rep={N_REP}", flush=True)
    t0 = time.perf_counter()
    res1 = run_benchmark(projeto, cfg1, progress=prog)
    print(f"[S1] done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    cols = ["label", "best", "mean", "median", "feasibility_rate",
            "best_feasible_volume_m3", "wall_time_mean_s"]
    print(res1.summary[cols].to_string(index=False), flush=True)
    res1.summary.to_csv(SCRATCH / "s1_summary.csv", index=False)
    res1.per_rep.to_csv(SCRATCH / "s1_per_rep.csv", index=False)
    res1.pvalues.to_csv(SCRATCH / "s1_pvalues.csv")

    # --- S2: extended budget for cheap baselines
    cfg2 = BenchmarkConfig(
        algorithms=("ga", "pso", "gwo", "random"),
        budget_evals=BUDGET_S2, ego_budget_evals=EGO_BUDGET,
        n_rep=N_REP, base_seed=BASE_SEED, h_min_m=H_MIN, h_max_m=H_MAX,
        lhs_n_pop=LHS_NPOP, meta_pop_size=30, kernel_index=-1,
        ga_pop_size=50, ga_epoch=30,
    )
    print(f"\n[S2] extended budget={BUDGET_S2}, n_rep={N_REP}", flush=True)
    t0 = time.perf_counter()
    res2 = run_benchmark(projeto, cfg2, progress=prog)
    print(f"[S2] done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    print(res2.summary[cols].to_string(index=False), flush=True)
    res2.summary.to_csv(SCRATCH / "s2_summary.csv", index=False)

    # --- decomposition reference
    print(f"\n[DECOMP] per-footing DE reference ...", flush=True)
    dref = decomposition_reference(projeto, args)
    print(f"[DECOMP] vol={dref['volume_m3']:.3f} feasible={dref['feasible']} "
          f"max_g={dref['max_violation']:.2e} wall={dref['wall_time_s']:.1f}s", flush=True)

    # --- final comparison table
    print("\n===== SUMMARY (best feasible volume, m3) =====", flush=True)
    print(f"{'decomp (per-footing DE)':<26} {dref['volume_m3']:.3f}  feasible={dref['feasible']}", flush=True)
    for _, r in res1.summary.iterrows():
        print(f"{r['label']+' [S1]':<26} {r['best_feasible_volume_m3']:.3f}  "
              f"feas_rate={r['feasibility_rate']:.2f}  wall={r['wall_time_mean_s']:.1f}s", flush=True)
    for _, r in res2.summary.iterrows():
        print(f"{r['label']+' [S2]':<26} {r['best_feasible_volume_m3']:.3f}  "
              f"feas_rate={r['feasibility_rate']:.2f}  wall={r['wall_time_mean_s']:.1f}s", flush=True)

    import json
    (SCRATCH / "decomp_ref.json").write_text(json.dumps(dref, indent=2))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
