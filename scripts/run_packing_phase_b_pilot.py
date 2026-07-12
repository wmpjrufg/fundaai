"""Phase B pilot: coupled footing layout with active packing constraints.

This script starts the second research front without changing the frozen
article-1 protocol. It builds one synthetic clustered case from the existing
two-footing spreadsheet, forcing the non-overlap constraint to become active,
then solves two diagnostic variants:

* ``individual_centered``: each footing is optimised independently and then
  assembled at the clustered column centroids;
* ``fixed_centers``: dimensions only, with footing centres locked at the
  column centroids;
* ``packing_offsets``: dimensions plus plan offsets ``dx, dy`` for each
  footing centre, with extra constraints for column containment and lot bounds.

The engineering checks still use ``avaliar_projeto_componentes``. For shifted
footings, the DataFrame passed to the objective receives updated footing-centre
coordinates and effective moments

    Mx_eff = Mx_input - Fz * dx
    My_eff = My_input - Fz * dy

following the current FundaIA convention that ``Mx = Fz * e_x`` and
``My = Fz * e_y``. The sign convention is harmless for the present tension
check because the objective uses absolute moments, but it is documented here
so the Phase B formulation can be refined explicitly.

Outputs:

    experiments/phase_b_packing_pilot/summary.csv
    experiments/phase_b_packing_pilot/designs.csv
    experiments/phase_b_packing_pilot/config.json

Resumo em português:
    Piloto da Fase B: cria um caso sintético com pilares próximos,
    ativa a sobreposição e testa a inclusão de deslocamentos em planta
    das sapatas como variáveis de projeto.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
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

OUT_DIR = REPO_ROOT / "experiments" / "phase_b_packing_pilot"

SOURCE_CASE = REPO_ROOT / "assets" / "data" / "problema_fund_dois.xlsx"
F_CK_KPA = 25_000.0
COBRIMENTO_M = 0.04
H_MIN_M, H_MAX_M = 0.60, 3.00
SHIFT_MAX_M = 1.20
BALANCO_MIN_M = 0.10
PENALTY_THETA = 10.0
FEAS_TOL = 1e-9

# Clustered pillar positions [m] deliberately make centred footings overlap
# after dimension optimisation, but still allow feasible shifted layouts under
# the same h_max = 3.0 m used by article 1.
PILLAR_X = np.array([0.00, 2.00], dtype=float)
PILLAR_Y = np.array([0.00, 0.00], dtype=float)
LOT_X = (-2.50, 5.00)
LOT_Y = (-2.50, 2.50)

# Diagnostic DE budget: enough for a pilot, not a final benchmark.
DE_SEED = 7_311
DE_MAXITER_FIXED = 180
DE_MAXITER_PACKING = 260
DE_POPSIZE = 15
DE_TOL = 1e-8
DE_ATOL = 0.0

FEAS_PENALTY_LINEAR = 1e6
FEAS_PENALTY_QUAD = 1e8

EXTRA_GROUPS = ("contain", "boundary")
GROUPS = CONSTRAINT_GROUPS + EXTRA_GROUPS


@dataclass(frozen=True)
class PilotCase:
    """Container for the synthetic coupled case.

    :param df: Base DataFrame with clustered column positions
    :param n_comb: Number of load combinations
    :param f_ck_kpa: Concrete compressive strength [kPa]
    :param cobrimento_m: Concrete cover [m]
    :param pillar_x: Column-centroid x coordinates [m]
    :param pillar_y: Column-centroid y coordinates [m]
    """

    df: pd.DataFrame
    n_comb: int
    f_ck_kpa: float
    cobrimento_m: float
    pillar_x: np.ndarray
    pillar_y: np.ndarray


def _env_snapshot() -> dict:
    """Capture minimal runtime metadata.

    :return: Reproducibility metadata
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


def _build_case() -> PilotCase:
    """Build the clustered two-footing Phase B pilot case.

    :return: Synthetic coupled case
    """
    projeto = read_projeto_from_excel(
        SOURCE_CASE, f_ck_kpa=F_CK_KPA, cobrimento_m=COBRIMENTO_M
    )
    df = projeto_to_dataframe(projeto)
    if len(df) != len(PILLAR_X):
        raise RuntimeError(
            f"expected {len(PILLAR_X)} footings, got {len(df)} from {SOURCE_CASE}"
        )
    df = df.copy()
    df["xg (m)"] = PILLAR_X
    df["yg (m)"] = PILLAR_Y
    return PilotCase(
        df=df,
        n_comb=projeto.n_comb,
        f_ck_kpa=projeto.f_ck_kpa,
        cobrimento_m=projeto.cobrimento_m,
        pillar_x=PILLAR_X.copy(),
        pillar_y=PILLAR_Y.copy(),
    )


def _packing_x0(n: int) -> np.ndarray | None:
    """Return a feasible warm start for the current two-footing pilot.

    The warm start is deliberately simple and is persisted in ``config.json``;
    it prevents the first Phase B run from spending all its effort merely
    discovering that shifted layouts can be feasible.

    :param n: Number of footings
    :return: Flat ``[hx, hy, hz, dx, dy, ...]`` vector, or ``None``
    """
    if n != 2:
        return None
    return np.array([
        2.60, 2.99, 1.00, -0.60, 0.00,
        2.75, 2.20, 0.61,  0.00, 0.00,
    ], dtype=float)


def _decode(z: np.ndarray, *, movable: bool, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a design vector.

    :param z: Flat design vector
    :param movable: Whether ``dx, dy`` are present
    :param n: Number of footings
    :return: ``(dims, dx, dy)`` where dims has columns ``hx, hy, hz``
    """
    arr = np.asarray(z, dtype=float)
    if movable:
        block = arr.reshape(n, 5)
        dims = block[:, :3]
        dx = block[:, 3]
        dy = block[:, 4]
    else:
        dims = arr.reshape(n, 3)
        dx = np.zeros(n, dtype=float)
        dy = np.zeros(n, dtype=float)
    return dims, dx, dy


def _candidate_df(case: PilotCase, dims: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> pd.DataFrame:
    """Create the objective DataFrame for one candidate.

    :param case: Pilot case
    :param dims: Candidate dimensions, unused here but kept for symmetry
    :param dx: Footing-centre offset from column in x [m]
    :param dy: Footing-centre offset from column in y [m]
    :return: DataFrame consumed by ``avaliar_projeto_componentes``
    """
    _ = dims
    df = case.df.copy()
    df["xg (m)"] = case.pillar_x + dx
    df["yg (m)"] = case.pillar_y + dy
    for ci in range(1, case.n_comb + 1):
        fz = df[f"Fz-c{ci}"].to_numpy(dtype=float)
        df[f"Mx-c{ci}"] = df[f"Mx-c{ci}"].to_numpy(dtype=float) - fz * dx
        df[f"My-c{ci}"] = df[f"My-c{ci}"].to_numpy(dtype=float) - fz * dy
    return df


def _extra_constraints(
    case: PilotCase,
    dims: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    """Return containment and lot-boundary constraints.

    ``g <= 0`` is feasible.

    :param case: Pilot case
    :param dims: Candidate dimensions
    :param dx: Footing-centre offset from column in x [m]
    :param dy: Footing-centre offset from column in y [m]
    :return: ``[g_contain, g_boundary]``
    """
    hx, hy = dims[:, 0], dims[:, 1]
    ap = case.df["ap (m)"].to_numpy(dtype=float)
    bp = case.df["bp (m)"].to_numpy(dtype=float)

    # Column must remain inside the shifted footing with the same minimum
    # overhang adopted by the current geometric check.
    g_contain_x = 2.0 * (np.abs(dx) + ap / 2.0 + BALANCO_MIN_M) / hx - 1.0
    g_contain_y = 2.0 * (np.abs(dy) + bp / 2.0 + BALANCO_MIN_M) / hy - 1.0
    g_contain = float(np.maximum(g_contain_x, g_contain_y).max())

    cx = case.pillar_x + dx
    cy = case.pillar_y + dy
    left, right = cx - hx / 2.0, cx + hx / 2.0
    bottom, top = cy - hy / 2.0, cy + hy / 2.0
    g_boundary = max(
        float((LOT_X[0] - left).max()),
        float((right - LOT_X[1]).max()),
        float((LOT_Y[0] - bottom).max()),
        float((top - LOT_Y[1]).max()),
    ) / H_MAX_M
    return np.array([g_contain, g_boundary], dtype=float)


def _evaluate(z: np.ndarray, case: PilotCase, *, movable: bool) -> dict:
    """Evaluate a candidate and return decomposed metrics.

    :param z: Candidate design vector
    :param case: Pilot case
    :param movable: Whether ``z`` includes offsets
    :return: Metrics dictionary
    """
    n = len(case.df)
    dims, dx, dy = _decode(z, movable=movable, n=n)
    df = _candidate_df(case, dims, dx, dy)
    args = (df, case.n_comb, case.f_ck_kpa, case.cobrimento_m, PENALTY_THETA)
    design_x = dims.reshape(-1)
    theta, volume, g_core = avaliar_projeto_componentes(
        design_x, args, penalty=PENALTY_THETA
    )
    g_extra = _extra_constraints(case, dims, dx, dy)
    g = np.concatenate([g_core, g_extra])
    return {
        "theta": float(theta),
        "volume_m3": float(volume),
        "g": g,
        "dims": dims,
        "dx": dx,
        "dy": dy,
    }


def _objective(z: np.ndarray, case: PilotCase, *, movable: bool) -> float:
    """Feasibility-first scalar objective for the diagnostic DE runs.

    :param z: Candidate design vector
    :param case: Pilot case
    :param movable: Whether ``z`` includes offsets
    :return: Scalar objective
    """
    try:
        metrics = _evaluate(z, case, movable=movable)
    except ValueError:
        return float("inf")
    violation = np.clip(metrics["g"], 0.0, None)
    return float(
        metrics["volume_m3"]
        + FEAS_PENALTY_LINEAR * violation.sum()
        + FEAS_PENALTY_QUAD * np.square(violation).sum()
    )


def _solve(case: PilotCase, *, mode: str, movable: bool, seed: int) -> tuple[dict, list[dict]]:
    """Solve one pilot mode.

    :param case: Pilot case
    :param mode: ``fixed_centers`` or ``packing_offsets``
    :param movable: Whether offsets are variables
    :param seed: Differential Evolution seed
    :return: ``(summary, design_rows)``
    """
    n = len(case.df)
    bounds = [(H_MIN_M, H_MAX_M)] * (3 * n)
    maxiter = DE_MAXITER_FIXED
    if movable:
        bounds = []
        for _ in range(n):
            bounds.extend([
                (H_MIN_M, H_MAX_M),
                (H_MIN_M, H_MAX_M),
                (H_MIN_M, H_MAX_M),
                (-SHIFT_MAX_M, SHIFT_MAX_M),
                (-SHIFT_MAX_M, SHIFT_MAX_M),
            ])
        maxiter = DE_MAXITER_PACKING

    t0 = time.perf_counter()
    result = differential_evolution(
        lambda z: _objective(z, case, movable=movable),
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        atol=DE_ATOL,
        polish=True,
        updating="immediate",
        workers=1,
        x0=_packing_x0(n) if movable else None,
    )
    wall = time.perf_counter() - t0
    metrics = _evaluate(result.x, case, movable=movable)
    g = metrics["g"]
    summary = {
        "mode": mode,
        "movable": bool(movable),
        "dim": int(len(bounds)),
        "theta": metrics["theta"],
        "volume_m3": metrics["volume_m3"],
        "max_violation": float(g.max()),
        "feasible": bool(np.all(g <= FEAS_TOL)),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "success": bool(result.success),
        "wall_time_s": float(wall),
        "message": str(result.message),
        **{f"g_{name}": float(value) for name, value in zip(GROUPS, g)},
    }

    dims = metrics["dims"]
    dx = metrics["dx"]
    dy = metrics["dy"]
    rows: list[dict] = []
    for i, row in case.df.reset_index(drop=True).iterrows():
        rows.append({
            "mode": mode,
            "element": str(row["Elemento"]),
            "pillar_x_m": float(case.pillar_x[i]),
            "pillar_y_m": float(case.pillar_y[i]),
            "center_x_m": float(case.pillar_x[i] + dx[i]),
            "center_y_m": float(case.pillar_y[i] + dy[i]),
            "dx_m": float(dx[i]),
            "dy_m": float(dy[i]),
            "hx_m": float(dims[i, 0]),
            "hy_m": float(dims[i, 1]),
            "hz_m": float(dims[i, 2]),
        })
    return summary, rows


def _solve_individual_centered(case: PilotCase) -> tuple[dict, list[dict]]:
    """Optimise each footing independently, then assemble with fixed centres.

    This diagnostic row exposes the core Phase B issue: independent footing
    optima can be strictly feasible in isolation and still overlap after
    assembly when columns are close.

    :param case: Pilot case
    :return: ``(summary, design_rows)``
    """
    xs: list[float] = []
    nfev = 0
    nit = 0
    t0 = time.perf_counter()
    for i in range(len(case.df)):
        one = PilotCase(
            df=case.df.iloc[[i]].reset_index(drop=True),
            n_comb=case.n_comb,
            f_ck_kpa=case.f_ck_kpa,
            cobrimento_m=case.cobrimento_m,
            pillar_x=np.array([case.pillar_x[i]], dtype=float),
            pillar_y=np.array([case.pillar_y[i]], dtype=float),
        )
        result = differential_evolution(
            lambda z: _objective(z, one, movable=False),
            bounds=[(H_MIN_M, H_MAX_M)] * 3,
            seed=DE_SEED + 100 + i,
            maxiter=DE_MAXITER_FIXED,
            popsize=DE_POPSIZE,
            tol=DE_TOL,
            atol=DE_ATOL,
            polish=True,
            updating="immediate",
            workers=1,
        )
        xs.extend(np.asarray(result.x, dtype=float).tolist())
        nfev += int(result.nfev)
        nit += int(result.nit)

    wall = time.perf_counter() - t0
    metrics = _evaluate(np.asarray(xs, dtype=float), case, movable=False)
    g = metrics["g"]
    summary = {
        "mode": "individual_centered",
        "movable": False,
        "dim": int(3 * len(case.df)),
        "theta": metrics["theta"],
        "volume_m3": metrics["volume_m3"],
        "max_violation": float(g.max()),
        "feasible": bool(np.all(g <= FEAS_TOL)),
        "nfev": int(nfev),
        "nit": int(nit),
        "success": bool(np.all(g <= FEAS_TOL)),
        "wall_time_s": float(wall),
        "message": "Independent one-footing optima assembled at clustered centres.",
        **{f"g_{name}": float(value) for name, value in zip(GROUPS, g)},
    }
    dims = metrics["dims"]
    rows: list[dict] = []
    for i, row in case.df.reset_index(drop=True).iterrows():
        rows.append({
            "mode": "individual_centered",
            "element": str(row["Elemento"]),
            "pillar_x_m": float(case.pillar_x[i]),
            "pillar_y_m": float(case.pillar_y[i]),
            "center_x_m": float(case.pillar_x[i]),
            "center_y_m": float(case.pillar_y[i]),
            "dx_m": 0.0,
            "dy_m": 0.0,
            "hx_m": float(dims[i, 0]),
            "hy_m": float(dims[i, 1]),
            "hz_m": float(dims[i, 2]),
        })
    return summary, rows


def main() -> None:
    """Run the Phase B packing pilot and persist artefacts.

    :return: None
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case = _build_case()
    summaries: list[dict] = []
    designs: list[dict] = []
    print("[phase-b] individual_centered ...", flush=True)
    summary, rows = _solve_individual_centered(case)
    summaries.append(summary)
    designs.extend(rows)
    print(
        f"  V={summary['volume_m3']:.6f} m³, "
        f"g_sob={summary['g_sob']:.3e}, "
        f"max g={summary['max_violation']:.3e}, "
        f"feasible={summary['feasible']}",
        flush=True,
    )

    for mode, movable, seed in (
        ("fixed_centers", False, DE_SEED),
        ("packing_offsets", True, DE_SEED + 1),
    ):
        print(f"[phase-b] {mode} ...", flush=True)
        summary, rows = _solve(case, mode=mode, movable=movable, seed=seed)
        summaries.append(summary)
        designs.extend(rows)
        print(
            f"  V={summary['volume_m3']:.6f} m³, "
            f"g_sob={summary['g_sob']:.3e}, "
            f"max g={summary['max_violation']:.3e}, "
            f"feasible={summary['feasible']}",
            flush=True,
        )

    pd.DataFrame(summaries).to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(designs).to_csv(OUT_DIR / "designs.csv", index=False)
    config = {
        "source_case": str(SOURCE_CASE.relative_to(REPO_ROOT)),
        "f_ck_kpa": F_CK_KPA,
        "cobrimento_m": COBRIMENTO_M,
        "h_min_m": H_MIN_M,
        "h_max_m": H_MAX_M,
        "shift_max_m": SHIFT_MAX_M,
        "balanco_min_m": BALANCO_MIN_M,
        "pillar_x_m": PILLAR_X.tolist(),
        "pillar_y_m": PILLAR_Y.tolist(),
        "lot_x_m": list(LOT_X),
        "lot_y_m": list(LOT_Y),
        "penalty_theta": PENALTY_THETA,
        "feas_tol": FEAS_TOL,
        "de_seed": DE_SEED,
        "de_maxiter_fixed": DE_MAXITER_FIXED,
        "de_maxiter_packing": DE_MAXITER_PACKING,
        "de_popsize": DE_POPSIZE,
        "de_tol": DE_TOL,
        "de_atol": DE_ATOL,
        "packing_x0": (_packing_x0(len(case.df)).tolist()
                       if _packing_x0(len(case.df)) is not None else None),
        "env": _env_snapshot(),
    }
    (OUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"ARTEFATOS: {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
