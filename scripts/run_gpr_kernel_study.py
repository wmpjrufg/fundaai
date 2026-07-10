"""Kernel × penalty study of the GPR surrogate for the IC manuscript.

Reproducible re-execution of the historical kernel screening with a
frozen design: for each independent replication (seeded LHS sample +
seeded train/test split), the penalised pseudo-objective Θ(x) of the
three-foundation case is evaluated on 900 points, and every kernel in
``constroi_kernel()`` (21 configurations, ``k00``–``k20``) is fitted on
the 70% train partition under the *production* pipeline
(``StandardScaler → GaussianProcessRegressor(normalize_y=True,
alpha=0.1, n_restarts_optimizer=5)``). Predictive quality is scored on
the 30% test partition with R², MAE and RMSE.

The same X sample is labelled twice — once with the project default
penalty (α = 10) and once with the aggressive penalty (α = 10⁶) — so
the study isolates the effect of the penalty factor on the surrogate's
learnability, replicating the diagnostic reported in the partial
report under controlled seeds.

Outputs, under ``experiments/estudo_gpr/``:

    metrics.csv        one row per (kernel, penalty, replication)
    predictions.parquet y_test / y_pred for every fitted model
    meta.json          environment snapshot + frozen parameters

Usage (from the repository root):

    .venv/bin/python scripts/run_gpr_kernel_study.py

Resumo em português:
    Estudo kernels × penalidade do surrogate GPR com seeds registradas.
    Amostra LHS de 900 pontos do caso de 3 sapatas, split 70/30, 21
    kernels, penalidades α=10 e α=10⁶ sobre o mesmo X. Gera métricas
    R²/MAE/RMSE por réplica e as predições para os gráficos
    observado × predito do artigo.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from core.api._adapter import projeto_to_dataframe  # noqa: E402
from core.api.objective import avaliar_projeto_fast  # noqa: E402
from core.io import read_projeto_from_excel  # noqa: E402
from core.optimization import initial_population_01  # noqa: E402
from fundacao import constroi_kernel  # noqa: E402

# =============================================================================
# Frozen study parameters
# =============================================================================
OUT_DIR = REPO_ROOT / "experiments" / "estudo_gpr"
CASE_PATH = REPO_ROOT / "assets" / "data" / "problema_fund_três.xlsx"
F_CK_KPA = 25_000.0
COBRIMENTO_M = 0.04
H_MIN_M, H_MAX_M = 0.60, 3.00   # manuscript bounds (secao 6)
N_SAMPLES = 900           # LHS sample size per replication
TEST_FRACTION = 0.30      # 70/30 train/test split
SPLIT_SEEDS = (101, 102, 103)   # one independent replication per seed
PENALTIES = (1e1, 1e6)    # project default vs aggressive penalty
GPR_ALPHA = 0.1           # jitter — mirrors the production EGO pipeline
GPR_N_RESTARTS = 5
GPR_RANDOM_STATE = 42


def _env_snapshot() -> dict:
    """Capture the reproducibility metadata of this run.

    :return: Mapping with python/platform/library versions and git state
    """
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    libs = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn"):
        try:
            libs[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            libs[pkg] = "absent"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "libs": libs,
        "git_rev": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "case": CASE_PATH.name,
            "n_samples": N_SAMPLES,
            "test_fraction": TEST_FRACTION,
            "split_seeds": list(SPLIT_SEEDS),
            "penalties": list(PENALTIES),
            "gpr_alpha": GPR_ALPHA,
            "gpr_n_restarts": GPR_N_RESTARTS,
            "gpr_random_state": GPR_RANDOM_STATE,
            "bounds_m": [H_MIN_M, H_MAX_M],
        },
    }


def main() -> None:
    """Run the kernel × penalty screening and persist metrics + predictions.

    :return: None
    """
    t_study = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    projeto = read_projeto_from_excel(
        CASE_PATH, f_ck_kpa=F_CK_KPA, cobrimento_m=COBRIMENTO_M
    )
    df_input = projeto_to_dataframe(projeto)
    dim = 3 * projeto.n_fund
    base_args = (df_input, projeto.n_comb, projeto.f_ck_kpa, projeto.cobrimento_m)

    kernels = constroi_kernel()
    metric_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for seed in SPLIT_SEEDS:
        # LHS sample and targets for this replication (both penalties
        # share the same X so the comparison isolates the label scale).
        x_pop = np.asarray(initial_population_01(
            N_SAMPLES, dim, [H_MIN_M] * dim, [H_MAX_M] * dim,
            seed=seed, use_lhs=True,
        ))
        targets = {
            pen: np.array([
                avaliar_projeto_fast(x, base_args + (pen,)) for x in x_pop
            ])
            for pen in PENALTIES
        }

        for pen in PENALTIES:
            x_tr, x_te, y_tr, y_te = train_test_split(
                x_pop, targets[pen],
                test_size=TEST_FRACTION, random_state=seed,
            )
            for idx, ker in enumerate(kernels):
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("gp", GaussianProcessRegressor(
                        kernel=ker,
                        normalize_y=True,
                        alpha=GPR_ALPHA,
                        n_restarts_optimizer=GPR_N_RESTARTS,
                        random_state=GPR_RANDOM_STATE,
                    )),
                ])
                t0 = time.perf_counter()
                pipe.fit(x_tr, y_tr)
                fit_s = time.perf_counter() - t0
                y_pred = pipe.predict(x_te)

                kernel_id = f"k{idx:02d}"
                metric_rows.append({
                    "kernel_id": kernel_id,
                    "kernel_repr": str(ker),
                    "penalty": pen,
                    "seed": seed,
                    "n_train": len(x_tr),
                    "n_test": len(x_te),
                    "r2": float(r2_score(y_te, y_pred)),
                    "mae": float(mean_absolute_error(y_te, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
                    "fit_time_s": float(fit_s),
                })
                pred_rows.append(pd.DataFrame({
                    "kernel_id": kernel_id,
                    "penalty": pen,
                    "seed": seed,
                    "y_test": y_te,
                    "y_pred": y_pred,
                }))
                print(f"seed={seed} pen={pen:g} {kernel_id}: "
                      f"R2={metric_rows[-1]['r2']:.4f} "
                      f"RMSE={metric_rows[-1]['rmse']:.4f} "
                      f"({fit_s:.1f}s)", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_parquet(
        OUT_DIR / "predictions.parquet", index=False
    )
    meta = _env_snapshot()
    meta["wall_time_total_s"] = time.perf_counter() - t_study
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"STUDY COMPLETE in {meta['wall_time_total_s']/60:.1f} min "
          f"({len(metrics)} fits)", flush=True)


if __name__ == "__main__":
    main()
