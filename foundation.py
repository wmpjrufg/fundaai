"""This script contains functions for dimensioning isolated footings based on input data."""
import pandas as pd


def tensao_adm_solo(solo: str, spt: float) -> float:
    """
    Calcula a tensão admissível do solo com base no tipo de solo e no Nspt.

    :param solo: Tipo de solo ('pedregulho', 'areia', 'silte', 'argila')
    :param spt: Valor do Nspt
    
    :return: Tensão max admissível do solo em kPa
    """

    if solo.lower() == 'pedregulho':
        return spt / 30 * 1E3
    elif solo.lower() == 'areia':
        return spt / 40 * 1E3
    else: # silte ou argila
        return spt / 50 * 1E3


def calcular_sigma_max_min(f_z: float, m_x: float, m_y: float, h_x: float, h_y: float) -> tuple[float, float]:
    """
    Calcula as tensões máxima e mínima atuantes na sapata, considerando excentricidades nos dois eixos.

    :param f_z: Esforço axial (kN)
    :param m_x: Momento em x (kN·m)
    :param m_y: Momento em y (kN·m)
    :param h_x: Dimensão da sapata em x (m)
    :param h_y: Dimensão da sapata em y (m)

    :return: saida[0] = tensão máxima (kPa), saida[1] = tensão mínima (kPa)
    """
    
    m_x = abs(m_x)
    m_y = abs(m_y)
    sigma_fz = f_z / (h_x * h_y)
    aux_mx = 6 * (m_x / f_z) / h_x
    aux_my = 6 * (m_y / f_z) / h_y
    
    return (sigma_fz) * (1 + aux_mx + aux_my), (sigma_fz) * (1 - aux_mx - aux_my) 
