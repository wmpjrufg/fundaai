"""Decomposed Differential Evolution baseline for the manuscript audit.

The final IC protocol intentionally compares global optimisers on the full
design vector. In the three frozen study cases, however, the footing overlap
constraint is inactive by geometry; consequently, the current problem is almost
separable by footing. This script quantifies that structural property with a
simple feasibility-first baseline:

1. split each case into one-footing subproblems;
2. optimise each subproblem with SciPy's Differential Evolution;
3. assemble the full design and re-evaluate it with the same objective kernel
   used by the benchmark.

The result is not a new paired S1 algorithm: it uses a larger deterministic
diagnostic budget and serves as an optimum-proximity audit for the simplified
instances.

Usage, from the repository root:

    .venv/bin/python scripts/run_decomposition_baseline.py

Outputs:

    experiments/protocolo_final/decomposicao_de/summary.csv
    experiments/protocolo_final/decomposicao_de/designs.csv
    experiments/protocolo_final/decomposicao_de/config.json

Resumo em português:
    Executa um baseline de Differential Evolution por sapata, monta a solução
    completa e compara o volume factível contra o melhor volume factível do
    protocolo global. Serve para documentar a quase separabilidade dos casos
    atuais.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.api._adapter import projeto_to_dataframe  # noqa: E402
from core.api.objective import CONSTRAINT_GROUPS, avaliar_projeto_componentes  # noqa: E402
from core.io import read_projeto_from_excel  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "protocolo_final" / "decomposicao_de"

CASES: dict[str, dict[str, str]] = {
    "caso1_um": {"label": "Caso 1", "path": "assets/data/problema_fund_um.xlsx"},
    "caso2_dois": {"label": "Caso 2", "path": "assets/data/problema_fund_dois.xlsx"},
    "caso3_tres": {"label": "Caso 3", "path": "assets/data/problema_fund_três.xlsx"},
}

F_CK_KPA = 25_000.0
COBRIMENTO_M = 0.04
H_MIN_M, H_MAX_M = 0.60, 3.00
PENALTY_THETA = 10.0
FEAS_TOL = 1e-9

# Feasibility-first scalar used only by this diagnostic baseline. The reported
# value remains the unpenalised volume after strict re-evaluation.
FEAS_PENALTY_LINEAR = 1e6
FEAS_PENALTY_QUAD = 1e8

DE_MAXITER = 220
DE_POPSIZE = 20
DE_TOL = 1e-9
DE_ATOL = 0.0
DE_SEED_BASE = 2_042

ALG_LABEL = {
    "ego": "EGO+GPR",
    "cbo": "CBO (ECI)",
    "ga": "GA",
    "pso": "PSO",
    "gwo": "GWO",
    "random": "Aleatória",
}


def _env_snapshot() -> dict:
    """Capture minimal reproducibility metadata.

    :return: Mapping with runtime versions and timestamp
    """
    libs = {}
    for pkg in ("numpy", "pandas", "scipy"):
        try:
            libs[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            libs[pkg] = "absent"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libs": libs,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _objective_feasibility_first(z: np.ndarray, args: tuple) -> float:
    """Feasibility-first scalar for one-footing DE subproblems.

    :param z: Candidate ``[h_x, h_y, h_z]``
    :param args: Objective args for a one-row DataFrame
    :return: Scalar minimised by Differential Evolution
    """
    try:
        _theta, volume, g = avaliar_projeto_componentes(
            z, args, penalty=PENALTY_THETA
        )
    except ValueError:
        return float("inf")
    violation = np.clip(g, 0.0, None)
    return float(
        volume
        + FEAS_PENALTY_LINEAR * violation.sum()
        + FEAS_PENALTY_QUAD * np.square(violation).sum()
    )


def _best_protocol_reference(case: str) -> dict:
    """Return the best strict feasible volume already found by the protocol.

    :param case: Case folder name
    :return: Mapping with best S1, best S2 and overall protocol references
    """
    base = REPO_ROOT / "experiments" / "protocolo_final" / case
    refs: list[pd.DataFrame] = []
    for scenario in ("S1_orcamento_igual", "S1_cbo", "S2_orcamento_estendido"):
        path = base / scenario / "summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "scenario", scenario)
        refs.append(frame)
    if not refs:
        return {
            "protocol_best_algorithm": "",
            "protocol_best_scenario": "",
            "protocol_best_feasible_volume_m3": np.nan,
        }
    all_refs = pd.concat(refs, ignore_index=True)
    all_refs = all_refs[np.isfinite(all_refs["best_feasible_volume_m3"])]
    row = all_refs.loc[all_refs["best_feasible_volume_m3"].idxmin()]
    return {
        "protocol_best_algorithm": str(row["algorithm"]),
        "protocol_best_label": ALG_LABEL.get(str(row["algorithm"]), str(row["algorithm"])),
        "protocol_best_scenario": str(row["scenario"]),
        "protocol_best_feasible_volume_m3": float(row["best_feasible_volume_m3"]),
    }


def _solve_case(case: str, meta: dict[str, str]) -> tuple[dict, list[dict]]:
    """Solve every one-footing subproblem of a case and re-evaluate globally.

    :param case: Case id
    :param meta: Case metadata
    :return: ``(summary_row, design_rows)``
    """
    projeto = read_projeto_from_excel(
        REPO_ROOT / meta["path"],
        f_ck_kpa=F_CK_KPA,
        cobrimento_m=COBRIMENTO_M,
    )
    df = projeto_to_dataframe(projeto)
    full_x: list[float] = []
    design_rows: list[dict] = []
    t0 = time.perf_counter()

    for i, row in df.reset_index(drop=True).iterrows():
        df_one = df.iloc[[i]].reset_index(drop=True)
        args_one = (
            df_one,
            projeto.n_comb,
            projeto.f_ck_kpa,
            projeto.cobrimento_m,
            PENALTY_THETA,
        )
        seed = DE_SEED_BASE + 100 * list(CASES).index(case) + i
        result = differential_evolution(
            lambda z: _objective_feasibility_first(z, args_one),
            bounds=[(H_MIN_M, H_MAX_M)] * 3,
            seed=seed,
            maxiter=DE_MAXITER,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            atol=DE_ATOL,
            polish=True,
            updating="immediate",
            workers=1,
        )
        x = np.asarray(result.x, dtype=float)
        theta, volume, g = avaliar_projeto_componentes(
            x, args_one, penalty=PENALTY_THETA
        )
        full_x.extend(x.tolist())
        design_rows.append({
            "case": case,
            "case_label": meta["label"],
            "element": str(row["Elemento"]),
            "seed": int(seed),
            "hx_m": float(x[0]),
            "hy_m": float(x[1]),
            "hz_m": float(x[2]),
            "theta": float(theta),
            "volume_m3": float(volume),
            "max_violation": float(np.max(g)),
            "feasible": bool(np.all(g <= FEAS_TOL)),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "success": bool(result.success),
            "message": str(result.message),
        })

    args_full = (
        df,
        projeto.n_comb,
        projeto.f_ck_kpa,
        projeto.cobrimento_m,
        PENALTY_THETA,
    )
    theta, volume, g = avaliar_projeto_componentes(
        np.asarray(full_x, dtype=float), args_full, penalty=PENALTY_THETA
    )
    wall = time.perf_counter() - t0
    ref = _best_protocol_reference(case)
    protocol_volume = ref["protocol_best_feasible_volume_m3"]
    reduction_pct = (
        100.0 * (protocol_volume - volume) / protocol_volume
        if np.isfinite(protocol_volume) else np.nan
    )
    summary = {
        "case": case,
        "case_label": meta["label"],
        "n_fund": int(projeto.n_fund),
        "dim": int(3 * projeto.n_fund),
        "theta": float(theta),
        "volume_m3": float(volume),
        "max_violation": float(np.max(g)),
        "feasible": bool(np.all(g <= FEAS_TOL)),
        "total_nfev": int(sum(r["nfev"] for r in design_rows)),
        "wall_time_s": float(wall),
        "protocol_gap_reduction_pct": float(reduction_pct),
        **{f"g_{name}": float(value) for name, value in zip(CONSTRAINT_GROUPS, g)},
        **ref,
    }
    return summary, design_rows


def main() -> None:
    """Run the decomposed baseline and persist all artefacts.

    :return: None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    designs: list[dict] = []
    for case, meta in CASES.items():
        print(f"[{case}] decomposed DE ...", flush=True)
        summary, rows = _solve_case(case, meta)
        summaries.append(summary)
        designs.extend(rows)
        print(
            f"  V={summary['volume_m3']:.6f} m³, "
            f"max g={summary['max_violation']:.2e}, "
            f"Δ={summary['protocol_gap_reduction_pct']:.2f}%",
            flush=True,
        )

    pd.DataFrame(summaries).to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(designs).to_csv(OUT_DIR / "designs.csv", index=False)
    config = {
        "case_paths": CASES,
        "f_ck_kpa": F_CK_KPA,
        "cobrimento_m": COBRIMENTO_M,
        "h_min_m": H_MIN_M,
        "h_max_m": H_MAX_M,
        "penalty_theta": PENALTY_THETA,
        "feas_tol": FEAS_TOL,
        "feas_penalty_linear": FEAS_PENALTY_LINEAR,
        "feas_penalty_quad": FEAS_PENALTY_QUAD,
        "de_maxiter": DE_MAXITER,
        "de_popsize": DE_POPSIZE,
        "de_tol": DE_TOL,
        "de_atol": DE_ATOL,
        "de_seed_base": DE_SEED_BASE,
        "env": _env_snapshot(),
    }
    (OUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"ARTEFATOS: {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
