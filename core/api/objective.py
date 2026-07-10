"""Vectorised objective-function evaluator for the footing optimisation loop.

Sprint 3.9 — this module is the architecturally correct home for the
fast pseudo-objective. It sits in ``core.api`` because:

* It receives a ``pandas.DataFrame`` inside ``args`` (disqualifies it
  from ``core.engineering``, where every signature is DataFrame-free).
* It wires together multiple ``core.engineering`` checks in a single
  vectorised pass (that wiring role belongs to ``core.api``).

Two callables are exposed:

``avaliar_projeto_fast(x, args)``
    Fully vectorised implementation (numpy broadcasting). No ``df.apply``
    calls. Returns only the scalar OF value — sufficient for every
    optimisation loop. ~0.1 ms per evaluation (10–30 fund / 2–4 comb).

``avaliar_projeto_legacy(x, args)``
    Thin wrapper around the original ``fundacao._avaliar_projeto``
    (pandas/``df.apply``). Returns only the scalar OF. Kept for
    performance benchmarking and numerical cross-validation.
    ~6–13 ms per evaluation.

Both functions share the same public signature and are numerically
identical (``diff = 0.00e+00`` verified on 2026-06-05).

Benchmark (medido em 2026-06-05, Apple Silicon / sandbox Linux):

+---------------------+----------+---------+---------+
| Cenário             | legacy   | fast    | speedup |
+---------------------+----------+---------+---------+
| 3 fund / 2 comb     | 6.36 ms  | 0.09 ms |  ~70×   |
| 10 fund / 4 comb    | 10.29 ms | 0.13 ms |  ~78×   |
| 30 fund / 4 comb    | 12.79 ms | 0.15 ms |  ~86×   |
+---------------------+----------+---------+---------+

Dependency graph (no circular imports):

    core.api.objective
        └── core.engineering  (sobreposicao_matrix)
        └── numpy, pandas     (stdlib-level deps)

``fundacao._avaliar_projeto`` is imported only inside
``avaliar_projeto_legacy`` at call time (lazy import) to avoid the
circular dependency that would arise if ``fundacao`` imported this
module at the top level.

Resumo em português:
    Avaliador vetorizado da função pseudo-objetivo. ``avaliar_projeto_fast``
    substitui todos os ``df.apply`` por operações numpy broadcasting;
    ``avaliar_projeto_legacy`` mantém o comportamento original para
    comparação. Ambas têm a mesma assinatura e retornam o mesmo valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.engineering import sobreposicao_matrix

__all__ = ["avaliar_projeto_fast", "avaliar_projeto_legacy"]

_PENALTY_DEFAULT: float = 1e1


def _unpack(args: tuple) -> tuple:
    """Extract ``(df, n_comb, f_ck, cob_m, penalty)`` from the args tuple.

    Accepts 4 or 5 elements for backward compatibility with notebooks
    that always passed a silent fifth penalty value.

    :param args: ``(df, n_comb, f_ck, cob_m)`` or
                 ``(df, n_comb, f_ck, cob_m, penalty)``

    :return: Five-element tuple adding ``penalty`` as last element
    """
    df, n_comb, f_ck, cob_m = args[0], args[1], args[2], args[3]
    penalty = args[4] if len(args) >= 5 else _PENALTY_DEFAULT
    return df, int(n_comb), float(f_ck), float(cob_m), float(penalty)


def avaliar_projeto_fast(x, args, *, penalty: float | None = None) -> float:
    """Evaluate the penalised pseudo-objective with full numpy vectorisation.

    Replaces every ``df.apply`` in the original ``_avaliar_projeto`` with
    numpy broadcasting, eliminating the Python-interpreter-per-row overhead.
    Returns only the scalar OF value (no annotated DataFrame) — sufficient
    for any optimisation loop.

    All normative constraints are preserved:

    * ``g_sob`` — AABB overlap between adjacent footings
      (``core.engineering.sobreposicao_matrix``)
    * ``g_ten`` — soil bearing pressure (σ_max and σ_min vs σ_adm, NBR 6122)
    * ``g_pun`` — punching shear at the C critical section (NBR 6118 §19.5)
    * ``g_geo`` — minimum pillar-footing overhang (0.10 m each side)

    For a fully annotated DataFrame (all intermediate columns) use
    ``fundacao.obj_teste``, which still calls the legacy implementation.

    Preconditions (validated at the domain boundary, see
    ``core.domain.Combinacao`` and ``core.domain.FundacaoProjeto``):
    every ``Fz-c{i}`` must be strictly positive (the sigma formulas
    divide by ``Fz``) and ``f_ck`` must be given in kPa. The only guard
    enforced here is ``hz > cob_m`` because ``hz`` is a *design
    variable*: a candidate with non-positive effective depth would flip
    the punching-shear sign and read as feasible.

    :param x: Design vector of length ``3 * N_fund``, layout
              ``[hx_0, hy_0, hz_0, ..., hx_{N-1}, hy_{N-1}, hz_{N-1}]``
    :param args: Tuple ``(df, n_comb, f_ck_kpa, cob_m)`` or
                 ``(df, n_comb, f_ck_kpa, cob_m, penalty)``
    :param penalty: Override the constraint penalty factor; uses ``args[4]``
                    or the project default (10.0) when ``None``

    :return: Scalar penalised volume [m³]

    :raises ValueError: When any ``hz`` is not strictly greater than
                        ``cob_m`` (non-positive effective depth). Mirrors
                        the guard in
                        ``core.engineering.verificacao_puncao_sapata`` so
                        the fast and legacy paths fail identically
    """
    df, n_comb, f_ck, cob_m, pen_default = _unpack(args)
    pen = pen_default if penalty is None else float(penalty)

    n = len(df)
    x_arr = np.asarray(x, dtype=np.float64).reshape(n, 3)
    hx = x_arr[:, 0]
    hy = x_arr[:, 1]
    hz = x_arr[:, 2]

    if np.any(hz <= cob_m):
        bad = float(hz.min())
        raise ValueError(
            f"effective depth d = h_z - cob must be positive for every "
            f"footing; got min(h_z)={bad}, cob={cob_m}. Keep the lower "
            f"bound of h_z strictly above the concrete cover."
        )

    # --- volume bruto ---------------------------------------------------
    vol = hx * hy * hz  # (N,)

    # --- sobreposição AABB (vetorizada desde Sprint 3.8) ----------------
    xg = df["xg (m)"].to_numpy(dtype=np.float64)
    yg = df["yg (m)"].to_numpy(dtype=np.float64)
    if n == 1:
        g_sob = np.zeros(1, dtype=np.float64)
    else:
        overlap = sobreposicao_matrix(
            xg - hx / 2, xg + hx / 2, yg - hy / 2, yg + hy / 2
        )
        g_sob = overlap.sum(axis=1) / (hx * hy)  # (N,)

    # --- tensão admissível do solo (vetoriza tensao_adm_solo) -----------
    solo = np.char.lower(df["solo"].to_numpy(dtype=str))
    spt = df["spt"].to_numpy(dtype=np.float64)
    sig_adm = np.where(
        solo == "pedregulho", spt / 30.0 * 1e3,
        np.where(solo == "areia", spt / 40.0 * 1e3, spt / 50.0 * 1e3),
    )  # (N,) [kPa]

    # --- dimensões dos pilares ------------------------------------------
    ap = df["ap (m)"].to_numpy(dtype=np.float64)  # (N,)
    bp = df["bp (m)"].to_numpy(dtype=np.float64)  # (N,)

    # --- punção e tensão por combinação ---------------------------------
    g_tensao_mat = np.empty((n, n_comb), dtype=np.float64)
    g_puncao_mat = np.empty((n, n_comb), dtype=np.float64)

    # constantes de punção (dependem de f_ck e cob_m, não de x)
    d = hz - cob_m                                    # (N,)
    alpha_v2 = 1.0 - (f_ck / 1_000.0) / 250.0        # escalar
    tau_rd2 = 0.27 * alpha_v2 * (f_ck / 1.4)         # escalar [kPa]
    u_rd2 = 2.0 * (ap + bp)                           # (N,) [m]

    for ci in range(n_comb):
        lbl = f"c{ci + 1}"
        fz = df[f"Fz-{lbl}"].to_numpy(dtype=np.float64)
        mx = np.abs(df[f"Mx-{lbl}"].to_numpy(dtype=np.float64))
        my = np.abs(df[f"My-{lbl}"].to_numpy(dtype=np.float64))

        # vetoriza calcular_sigma_max_min
        s_fz = (fz / (hx * hy)) * 1.05
        aux_x = 6.0 * mx / (fz * hx)
        aux_y = 6.0 * my / (fz * hy)
        s_max = s_fz * (1.0 + aux_x + aux_y)
        s_max = np.where(s_max > 0.0, s_max * 1.30, s_max)
        s_min = s_fz * (1.0 - aux_x - aux_y)
        s_min = np.where(s_min > 0.0, s_min * 1.30, s_min)

        # vetoriza checagem_tensao_max_min  (g <= 0 → viável)
        g_max = np.where(s_max >= 0.0, s_max / sig_adm - 1.0, -s_max / sig_adm)
        g_min = np.where(s_min >= 0.0, s_min / sig_adm - 1.0, -s_min / sig_adm)
        g_tensao_mat[:, ci] = np.maximum(g_max, g_min)

        # vetoriza verificacao_puncao_sapata (seção C)
        tau_sd2 = (1.4 * fz) / (u_rd2 * d)
        g_puncao_mat[:, ci] = tau_sd2 / tau_rd2 - 1.0

    g_tensao = g_tensao_mat.max(axis=1)  # pior combinação (N,)
    g_puncao = g_puncao_mat.max(axis=1)  # pior combinação (N,)

    # --- geometria (vetoriza checagem_geometria, balanço 0.10 m) --------
    g_geo = np.maximum(
        1.0 + 2.0 * 0.10 / ap - hx / ap,
        1.0 + 2.0 * 0.10 / bp - hy / bp,
    )  # (N,)

    # --- pseudo-objetivo -------------------------------------------------
    return float(
        (
            vol
            + np.clip(g_sob,    0.0, None) * pen
            + np.clip(g_puncao, 0.0, None) * pen
            + np.clip(g_tensao, 0.0, None) * pen
            + np.clip(g_geo,    0.0, None) * pen
        ).sum()
    )


def avaliar_projeto_legacy(x, args) -> float:
    """Evaluate the penalised pseudo-objective using the original pandas implementation.

    Thin wrapper around ``fundacao._avaliar_projeto`` that discards the
    annotated DataFrame and returns only the scalar OF value. Kept for
    performance benchmarking and numerical cross-validation against
    ``avaliar_projeto_fast``.

    The import of ``fundacao`` is deferred to call time to avoid the
    circular dependency that arises because ``fundacao`` imports this
    module.

    :param x: Design vector of length ``3 * N_fund``
    :param args: Tuple ``(df, n_comb, f_ck_kpa, cob_m)`` or
                 ``(df, n_comb, f_ck_kpa, cob_m, penalty)``

    :return: Scalar penalised volume [m³]
    """
    import fundacao as _fundacao  # deferred — avoids circular import
    of_total, _ = _fundacao._avaliar_projeto(x, args)
    return of_total
