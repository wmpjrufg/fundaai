"""Generate a synthetic 25-footing case (dim=75) and time one rep of each solver.

Regime: fixed, well-spaced column positions (5x5 grid, 6 m spacing) so the
non-overlap constraint is inactive by construction — the same regime as the
frozen article-1 cases, but scaled to N=25. This is a scaling/stress probe.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/lucasteixeira/Documents/Iniciacao_cientifica/fundaIA")
sys.path.insert(0, str(REPO))
SCRATCH = Path(__file__).resolve().parent

F_CK_KPA = 25_000.0
COBRIMENTO_M = 0.04
H_MIN, H_MAX = 0.60, 3.00


def build_df(n: int, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    spacing = 6.0
    rows = []
    solos = np.array(["argila", "areia"])
    for i in range(n):
        gx = (i % side) * spacing
        gy = (i // side) * spacing
        # modest columns so geometry constraint stays easy under h<=3.0
        ap = float(rng.uniform(0.20, 0.50))
        bp = float(rng.uniform(0.20, 0.50))
        solo = str(rng.choice(solos, p=[0.7, 0.3]))
        spt = float(rng.integers(10, 45))
        # three load combinations, varied vertical load and small moments
        base_fz = float(rng.uniform(300.0, 1500.0))
        row = {
            "Elemento": f"P{i+1:02d}",
            "ap (m)": round(ap, 3),
            "bp (m)": round(bp, 3),
            "spt": spt,
            "solo": solo,
            "xg (m)": round(gx, 3),
            "yg (m)": round(gy, 3),
        }
        for c in (1, 2, 3):
            fz = base_fz * float(rng.uniform(0.9, 1.1))
            mx = float(rng.uniform(-40.0, 40.0))
            my = float(rng.uniform(-40.0, 40.0))
            row[f"Fz-c{c}"] = round(fz, 2)
            row[f"Mx-c{c}"] = round(mx, 2)
            row[f"My-c{c}"] = round(my, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    n = 25
    df = build_df(n)
    xlsx = SCRATCH / f"caso_{n}_sapatas.xlsx"
    df.to_excel(xlsx, index=False)
    print(f"wrote {xlsx} shape={df.shape}", flush=True)

    from core.io import read_projeto_from_excel
    from core.api._adapter import projeto_to_dataframe
    from core.api.objective import avaliar_projeto_componentes, avaliar_projeto_fast
    from core.api.benchmark import BenchmarkConfig, run_benchmark

    projeto = read_projeto_from_excel(xlsx, f_ck_kpa=F_CK_KPA, cobrimento_m=COBRIMENTO_M)
    dim = 3 * projeto.n_fund
    print(f"projeto: n_fund={projeto.n_fund} n_comb={projeto.n_comb} dim={dim}", flush=True)

    # sanity: evaluate a mid-box design, check per-group feasibility
    df_in = projeto_to_dataframe(projeto)
    args = (df_in, projeto.n_comb, projeto.f_ck_kpa, projeto.cobrimento_m)
    xmid = np.full(dim, 1.5)
    theta, vol, g = avaliar_projeto_componentes(xmid.tolist(), args)
    print(f"mid-box (h=1.5): theta={theta:.3f} vol={vol:.3f} g(sob,pun,ten,geo)={np.round(g,4)}", flush=True)
    xbig = np.full(dim, 3.0)
    theta2, vol2, g2 = avaliar_projeto_componentes(xbig.tolist(), args)
    print(f"max-box (h=3.0): theta={theta2:.3f} vol={vol2:.3f} g={np.round(g2,4)} feasible={bool((g2<=0).all())}", flush=True)

    # timing of a single rep per solver at a small budget
    for alg, budget in [("random", 300), ("ga", 300), ("ego", 200), ("cbo", 200)]:
        cfg = BenchmarkConfig(
            algorithms=(alg,),
            budget_evals=budget,
            ego_budget_evals=200,
            n_rep=1,
            base_seed=42,
            h_min_m=H_MIN, h_max_m=H_MAX,
            lhs_n_pop=120,          # 1.6*dim (10*dim=750 infeasible under this budget)
            meta_pop_size=30,
            kernel_index=-1,
            ga_pop_size=50, ga_epoch=30,
        )
        t0 = time.perf_counter()
        res = run_benchmark(projeto, cfg)
        dt = time.perf_counter() - t0
        s = res.summary.iloc[0]
        print(f"[{alg:>6}] budget={budget} wall={dt:6.1f}s best={s['best']:.3f} "
              f"feas_rate={s['feasibility_rate']:.2f} best_feas_vol={s['best_feasible_volume_m3']:.3f} "
              f"nevals~{res.per_rep.iloc[0]['n_evals']}", flush=True)


if __name__ == "__main__":
    main()
