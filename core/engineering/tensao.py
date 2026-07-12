"""Soil-pressure helpers (composite bending on the footing base).

Resumo em português:
    Tensões máxima e mínima na interface solo-sapata por flexão composta
    oblíqua, e a restrição de projeto associada (g_tensao).

    Convenção importante do FundaIA: por herança das planilhas e do
    artigo-base local, ``M_x`` e ``M_y`` são componentes de momento
    associadas às excentricidades nas direções ``x`` e ``y``,
    respectivamente, isto é, ``M_x = F_z e_x`` e ``M_y = F_z e_y``.
    Portanto ``M_x`` é dividido por ``h_x`` e ``M_y`` por ``h_y``. Se uma
    fonte externa fornecer momentos estruturais em torno dos eixos
    globais, a conversão usual para esta convenção troca os eixos:
    ``M_x`` do FundaIA recebe o momento em torno de ``y`` e ``M_y`` recebe
    o momento em torno de ``x``.
"""

from __future__ import annotations


PESO_ESPECIFICO_CONCRETO_KN_M3 = 25.0


def calcular_sigma_max_min(
    f_zk: float,
    m_xk: float,
    m_yk: float,
    h_x: float,
    h_y: float,
    h_z: float,
    *,
    peso_especifico_concreto: float = PESO_ESPECIFICO_CONCRETO_KN_M3,
) -> tuple[float, float]:
    """This function returns the maximum and minimum soil pressures under composite bending.

    ``m_xk`` and ``m_yk`` follow the project convention documented in
    the module header: they are moment components producing eccentricity
    along X and Y, not necessarily bending moments about the X and Y
    structural axes. The footing self-weight is computed explicitly as
    ``gamma_c * h_x * h_y * h_z`` and added only to the centred vertical
    force. Moment terms are not multiplied by the self-weight factor. No
    additional design factor is applied here: these pressures are meant
    to be compared with the admissible soil pressure used by the
    pre-design model; tensile (negative) pressures feed directly into the
    no-tension constraint.

    :param f_zk: Characteristic vertical load on the pillar [kN]
    :param m_xk: Characteristic moment producing eccentricity along X [kN m]
    :param m_yk: Characteristic moment producing eccentricity along Y [kN m]
    :param h_x: Footing dimension in the X direction [m]
    :param h_y: Footing dimension in the Y direction [m]
    :param h_z: Footing height [m]
    :param peso_especifico_concreto: Reinforced-concrete unit weight [kN/m³]

    :return: [0] = Maximum (most compressive) soil pressure [kPa]
             [1] = Minimum soil pressure [kPa] (negative if tensile)
    """
    if f_zk <= 0.0:
        raise ValueError(f"f_zk must be strictly positive; got {f_zk}.")
    if h_x <= 0.0 or h_y <= 0.0 or h_z <= 0.0:
        raise ValueError(
            "footing dimensions must be strictly positive; got "
            f"h_x={h_x}, h_y={h_y}, h_z={h_z}."
        )
    if peso_especifico_concreto < 0.0:
        raise ValueError(
            "peso_especifico_concreto must be non-negative; got "
            f"{peso_especifico_concreto}."
        )

    m_xk = abs(m_xk)
    m_yk = abs(m_yk)
    area = h_x * h_y
    peso_proprio = peso_especifico_concreto * area * h_z
    sigma_axial = (f_zk + peso_proprio) / area
    sigma_mx = 6.0 * m_xk / (area * h_x)
    sigma_my = 6.0 * m_yk / (area * h_y)

    sigma_max = sigma_axial + sigma_mx + sigma_my
    sigma_min = sigma_axial - sigma_mx - sigma_my

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
