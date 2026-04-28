"""Punching-shear check at the C critical section (NBR 6118 item 19.5).

Resumo em português:
    Verificação à punção no perímetro crítico C (face do pilar). A
    verificação no perímetro C' (a d/2 da face) ainda não foi
    implementada na versão atual e está prevista para uma sprint
    futura, conforme issue rastreada no vault.
"""

from __future__ import annotations


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

    The verification at the C' critical section (perimeter at d/2 from
    the pillar face) is not implemented in the current release.

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
    """
    d = h_z - cob
    alpha_v2 = 1 - (f_ck / 1000) / 250
    f_cd = f_ck / 1.4
    tau_rd2 = 0.27 * alpha_v2 * f_cd
    u_rd2 = 2 * (a_p + b_p)
    tau_sd2 = (1.4 * f_zk) / (u_rd2 * d)
    g_rd2 = tau_sd2 / tau_rd2 - 1
    return tau_sd2, tau_rd2, u_rd2, g_rd2
