"""Soil-pressure helpers (composite bending on the footing base).

Resumo em português:
    Tensões máxima e mínima na interface solo-sapata por flexão composta
    oblíqua, e a restrição de projeto associada (g_tensao). Adota o
    coeficiente 1,05 para o peso próprio e o fator 1,30 quando a tensão
    é compressiva, em conformidade com a prática brasileira.
"""

from __future__ import annotations


def calcular_sigma_max_min(
    f_zk: float,
    m_xk: float,
    m_yk: float,
    h_x: float,
    h_y: float,
) -> tuple[float, float]:
    """This function returns the maximum and minimum soil pressures under composite bending.

    The vertical contribution sigma_Fz is multiplied by 1.05 to account
    for the footing self-weight. When the resulting pressure is
    compressive (positive), an additional 1.30 design factor is applied
    to obtain the design value; tensile (negative) pressures are kept
    unscaled so that they feed directly into the no-tension constraint.

    :param f_zk: Characteristic vertical load on the pillar [kN]
    :param m_xk: Characteristic bending moment on the X axis [kN m]
    :param m_yk: Characteristic bending moment on the Y axis [kN m]
    :param h_x: Footing dimension in the X direction [m]
    :param h_y: Footing dimension in the Y direction [m]

    :return: [0] = Maximum (most compressive) soil pressure [kPa]
             [1] = Minimum soil pressure [kPa] (negative if tensile)
    """
    m_xk = abs(m_xk)
    m_yk = abs(m_yk)
    sigma_fz = (f_zk / (h_x * h_y)) * 1.05
    aux_mx = 6 * m_xk / (f_zk * h_x)
    aux_my = 6 * m_yk / (f_zk * h_y)

    sigma_max = sigma_fz * (1 + aux_mx + aux_my)
    if sigma_max > 0:
        sigma_max *= 1.30

    sigma_min = sigma_fz * (1 - aux_mx - aux_my)
    if sigma_min > 0:
        sigma_min *= 1.30

    return sigma_max, sigma_min


def checagem_tensao_max_min(sigma: float, sigma_adm: float) -> float:
    """This function returns the soil-pressure design constraint.

    Constraint convention: g <= 0 means feasible. For compressive
    pressures (sigma >= 0), g = sigma / sigma_adm - 1. For tensile
    pressures (sigma < 0), g = -sigma / sigma_adm, which is always
    positive and therefore flags the infeasible case (no tension is
    admitted at the soil-footing interface).

    :param sigma: Acting soil pressure [kPa]
    :param sigma_adm: Admissible soil pressure [kPa]

    :return: Design constraint value g; g <= 0 means the constraint is satisfied
    """
    if sigma >= 0:
        return sigma / sigma_adm - 1
    return -sigma / sigma_adm
