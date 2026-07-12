"""Punching-shear checks at the C and C' critical sections (NBR 6118 item 19.5).

Two perimeters are verified, following the slab-column provisions that
NBR 6118 prescribes for footings (cf. Santos, Lima Neto & Ferreira,
2018, IBRACON Structures and Materials Journal 11(2) — evaluation of
ACI 318 / EC2 / NBR 6118 for RC footings):

* **Contorno C** (column face): diagonal-compression crushing,
  ``tau_sd2 <= tau_rd2 = 0.27 * alpha_v2 * f_cd``.
* **Contorno C'** (at ``2d`` from the column face, rounded corners):
  punching proper without shear reinforcement,
  ``tau_sd1 <= tau_rd1 = 0.13 * (1 + sqrt(20/d_cm)) * (100 rho f_ck_MPa)^(1/3)``,
  with the moment transfer share ``K * M_sd / (W_p * d)`` taken from
  NBR 6118 Table 19.2.

Modelling assumptions (documented in the manuscript):

* The flexural reinforcement ratio is not designed by the tool, so
  ``rho`` defaults to the code minimum ``rho_min(f_ck)`` (NBR 6118
  Table 17.3) — conservative, since ``tau_rd1`` grows with ``rho``.
* The soil reaction inside the control perimeter is **not** deducted
  from the acting force. EC2 allows that reduction; the NBR provisions
  do not prescribe it, and omitting it is conservative.

Resumo em português:
    Verificação à punção nos dois contornos críticos da NBR 6118:
    C (face do pilar, esmagamento da biela) e C' (a 2d da face,
    punção propriamente dita, com taxa de armadura mínima da
    Tabela 17.3 e transferência de momentos pela Tabela 19.2).
"""

from __future__ import annotations

import math


def verificacao_puncao_sapata(
    h_z: float,
    f_ck: float,
    a_p: float,
    b_p: float,
    f_zk: float,
    cob: float = 0.025,
) -> tuple[float, float, float, float]:
    """This function performs the punching-shear check at the C critical section.

    Implements the formulation of NBR 6118 (item 19.5) for the
    pillar-face perimeter:

        d        = h_z - cob
        alpha_v2 = 1 - f_ck / 250         (with f_ck in MPa)
        f_cd     = f_ck / 1.4
        tau_rd2  = 0.27 * alpha_v2 * f_cd
        u_rd2    = 2 * (a_p + b_p)
        tau_sd2  = (1.4 * F_zk) / (u_rd2 * d)
        g_rd2    = tau_sd2 / tau_rd2 - 1   (constraint convention)

    The companion function ``verificacao_puncao_sapata_c_linha`` checks
    the C' critical section at ``2d`` from the pillar face.

    :param h_z: Footing height [m]
    :param f_ck: Characteristic concrete compressive strength [kPa]
    :param a_p: Pillar dimension on the X axis [m]
    :param b_p: Pillar dimension on the Y axis [m]
    :param f_zk: Characteristic vertical load on the pillar [kN]
    :param cob: Concrete cover [m], default 0.025 m

    :return: [0] = Acting punching-shear stress at the C section, tau_sd2 [kPa]
             [1] = Resistant punching-shear stress, tau_rd2 [kPa]
             [2] = Critical perimeter at the pillar face, u_rd2 [m]
             [3] = Constraint value g_rd2; g_rd2 <= 0 means feasible

    :raises ValueError: When ``h_z <= cob`` — the effective depth
                        ``d = h_z - cob`` would be non-positive, flipping
                        the sign of ``tau_sd2`` and making an unbuildable
                        footing look feasible
    """
    d = h_z - cob
    if d <= 0:
        raise ValueError(
            f"effective depth d = h_z - cob must be positive; got "
            f"h_z={h_z}, cob={cob} (d={d}). Keep the lower bound of h_z "
            f"strictly above the concrete cover."
        )
    alpha_v2 = 1 - (f_ck / 1000) / 250
    f_cd = f_ck / 1.4
    tau_rd2 = 0.27 * alpha_v2 * f_cd
    u_rd2 = 2 * (a_p + b_p)
    tau_sd2 = (1.4 * f_zk) / (u_rd2 * d)
    g_rd2 = tau_sd2 / tau_rd2 - 1
    return tau_sd2, tau_rd2, u_rd2, g_rd2


# =============================================================================
# C' critical section (NBR 6118 item 19.5, perimeter at 2d from the face)
# =============================================================================
# NBR 6118 Table 17.3 — minimum flexural reinforcement ratio [%] per fck [MPa].
_FCK_TAB_MPA = (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0)
_RHO_MIN_PCT = (0.150, 0.150, 0.150, 0.164, 0.179, 0.194, 0.208,
                0.211, 0.219, 0.226, 0.233, 0.239, 0.245, 0.251, 0.256)

# NBR 6118 Table 19.2 — moment-transfer coefficient K per c1/c2 ratio.
_C1_C2_TAB = (0.5, 1.0, 2.0, 3.0)
_K_TAB = (0.45, 0.60, 0.70, 0.80)


def rho_minimo_flexao(f_ck: float) -> float:
    """This function returns the minimum flexural reinforcement ratio of NBR 6118 Table 17.3.

    Linear interpolation between the tabulated concrete classes. Values
    below C20 (outside the structural range of the code) are floored at
    the C20 ratio; values above C90 are rejected upstream by the domain
    validation of ``f_ck_kpa``.

    :param f_ck: Characteristic concrete compressive strength [kPa]

    :return: Minimum reinforcement ratio rho_min [dimensionless, e.g. 0.0015]
    """
    f_ck_mpa = f_ck / 1_000.0
    if f_ck_mpa <= _FCK_TAB_MPA[0]:
        return _RHO_MIN_PCT[0] / 100.0
    if f_ck_mpa >= _FCK_TAB_MPA[-1]:
        return _RHO_MIN_PCT[-1] / 100.0
    for i in range(len(_FCK_TAB_MPA) - 1):
        x0, x1 = _FCK_TAB_MPA[i], _FCK_TAB_MPA[i + 1]
        if x0 <= f_ck_mpa <= x1:
            y0, y1 = _RHO_MIN_PCT[i], _RHO_MIN_PCT[i + 1]
            return (y0 + (y1 - y0) * (f_ck_mpa - x0) / (x1 - x0)) / 100.0
    raise ValueError(f"f_ck out of table range: {f_ck_mpa} MPa")  # pragma: no cover


def k_tabela_19_2(c1_c2: float) -> float:
    """This function returns the moment-transfer coefficient K of NBR 6118 Table 19.2.

    Linear interpolation on the c1/c2 ratio, saturated at the normative
    limits [0.5, 3.0]. ``c1`` is the column dimension parallel to the
    load eccentricity produced by the transferred moment.

    :param c1_c2: Ratio between the column dimensions c1/c2 [-]

    :return: Coefficient K [-]
    """
    r = min(max(float(c1_c2), _C1_C2_TAB[0]), _C1_C2_TAB[-1])
    for i in range(len(_C1_C2_TAB) - 1):
        x0, x1 = _C1_C2_TAB[i], _C1_C2_TAB[i + 1]
        if x0 <= r <= x1:
            y0, y1 = _K_TAB[i], _K_TAB[i + 1]
            return y0 + (y1 - y0) * (r - x0) / (x1 - x0)
    raise ValueError(f"c1/c2 out of range: {c1_c2}")  # pragma: no cover


def verificacao_puncao_sapata_c_linha(
    h_z: float,
    f_ck: float,
    a_p: float,
    b_p: float,
    f_zk: float,
    m_xk: float = 0.0,
    m_yk: float = 0.0,
    cob: float = 0.025,
    rho: float | None = None,
) -> tuple[float, float, float, float]:
    """This function performs the punching-shear check at the C' critical section.

    Implements the NBR 6118 (item 19.5) provisions for the control
    perimeter at ``2d`` from the column face, without shear
    reinforcement::

        d        = h_z - cob
        u_1'     = 2 (a_p + b_p) + 4 pi d          (rounded corners)
        W_px     = a_p^2/2 + a_p b_p + 4 b_p d + 16 d^2 + 2 pi d a_p
        W_py     = b_p^2/2 + a_p b_p + 4 a_p d + 16 d^2 + 2 pi d b_p
        tau_sd1  = 1.4 F_zk / (u_1' d)
                   + K_x 1.4 |M_xk| / (W_px d) + K_y 1.4 |M_yk| / (W_py d)
        tau_rd1  = 0.13 (1 + sqrt(20 / d_cm)) (100 rho f_ck_MPa)^(1/3)
        g_rd1    = tau_sd1 / tau_rd1 - 1           (g <= 0 feasible)

    Project conventions: ``M_x`` produces eccentricity along the x axis
    (paired with ``a_p`` and ``W_px``), mirroring the sigma_max/min
    formulation; moment magnitudes are taken in absolute value; the
    size-effect factor is capped at 2.0 (conservative, only reachable
    for d < 20 cm); the soil reaction inside the perimeter is not
    deducted (conservative; the EC2-only allowance is not taken).

    :param h_z: Footing height [m]
    :param f_ck: Characteristic concrete compressive strength [kPa]
    :param a_p: Pillar dimension on the X axis [m]
    :param b_p: Pillar dimension on the Y axis [m]
    :param f_zk: Characteristic vertical load on the pillar [kN]
    :param m_xk: Characteristic moment producing eccentricity along X [kN m]
    :param m_yk: Characteristic moment producing eccentricity along Y [kN m]
    :param cob: Concrete cover [m], default 0.025 m
    :param rho: Flexural reinforcement ratio [-]; ``None`` adopts the
                NBR 6118 Table 17.3 minimum for ``f_ck`` (declared
                modelling hypothesis — the tool does not design the
                reinforcement)

    :return: [0] = Acting punching-shear stress at the C' section, tau_sd1 [kPa]
             [1] = Resistant punching-shear stress, tau_rd1 [kPa]
             [2] = Critical perimeter at 2d from the pillar face, u_rd1 [m]
             [3] = Constraint value g_rd1; g_rd1 <= 0 means feasible

    :raises ValueError: When ``h_z <= cob`` (non-positive effective depth)
    """
    d = h_z - cob
    if d <= 0:
        raise ValueError(
            f"effective depth d = h_z - cob must be positive; got "
            f"h_z={h_z}, cob={cob} (d={d}). Keep the lower bound of h_z "
            f"strictly above the concrete cover."
        )
    if rho is None:
        rho = rho_minimo_flexao(f_ck)

    u_rd1 = 2.0 * (a_p + b_p) + 4.0 * math.pi * d
    w_px = a_p ** 2 / 2.0 + a_p * b_p + 4.0 * b_p * d + 16.0 * d ** 2 \
        + 2.0 * math.pi * d * a_p
    w_py = b_p ** 2 / 2.0 + a_p * b_p + 4.0 * a_p * d + 16.0 * d ** 2 \
        + 2.0 * math.pi * d * b_p
    k_x = k_tabela_19_2(a_p / b_p)
    k_y = k_tabela_19_2(b_p / a_p)

    tau_sd1 = (1.4 * f_zk) / (u_rd1 * d) \
        + k_x * (1.4 * abs(m_xk)) / (w_px * d) \
        + k_y * (1.4 * abs(m_yk)) / (w_py * d)

    k_e = min(1.0 + math.sqrt(20.0 / (d * 100.0)), 2.0)
    tau_rd1 = 0.13 * k_e * (100.0 * rho * (f_ck / 1_000.0)) ** (1.0 / 3.0) * 1_000.0
    g_rd1 = tau_sd1 / tau_rd1 - 1.0
    return tau_sd1, tau_rd1, u_rd1, g_rd1
