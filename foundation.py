"""Esse script contém as funções que verificam uma sapata"""
import pandas as pd


def tensao_adm_solo(solo: str, spt: float) -> float:
    """Calcula a tensão admissível do solo com base no tipo de solo e no Nspt.

    :param solo: Tipo de solo ('pedregulho', 'areia', 'silte', 'argila')
    :param spt: Valor do Nspt
    
    :return: Tensão max admissível do solo [kPa]
    """

    if solo.lower() == 'pedregulho':
        return spt / 30 * 1E3
    elif solo.lower() == 'areia':
        return spt / 40 * 1E3
    else: # silte ou argila
        return spt / 50 * 1E3


def calcular_sigma_max_min(f_z: float, m_x: float, m_y: float, h_x: float, h_y: float) -> tuple[float, float]:
    """Calcula as tensões máxima e mínima atuantes na sapata, considerando excentricidades nos dois eixos.

    :param f_z: Esforço axial kN)
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


def checagem_tensao_max_min(sigma: float, sigma_adm: float) -> float:
    """Determina a restrição de projeto da tensão admissível do solo.

    :param sigma: Tensão atuante no solo [kPa]
    :param sigma_adm: Tensão máxima admissível do solo [kPa]

    :return: restrição de projeto e g <= 0 para restrição ser satisfeita
    """

    if sigma >= 0: 
        g = (1.3 * sigma) / sigma_adm - 1
    else:
        g = -sigma / sigma_adm

    return g


def obj_felipe_lucas(x, args):

    # Arguments
    df = args[0].copy()
    n_comb = args[1]
    n_fun = df.shape[0]

    # Design variables
    h_x, h_y = x  ###
    h_z = 0.60
    df['h_x (m)'] = h_x
    df['h_y (m)'] = h_y
    df['h_z (m)'] = h_z

    # Volumn
    df['volume (m3)'] = df['h_x (m)'] * df['h_y (m)'] * df['h_z (m)']

    # Admissible stress of the soil
    df['tensao adm. (kPa)'] = df.apply(lambda row: tensao_adm_solo(row['solo'], row['spt']), axis=1)

    # Combination label
    labels_comb = [f'c{i}' for i in range(1, n_comb + 1)]


    # Computing max and min for all load combinations
    t_max_aux = []
    t_min_aux = []
    for i  in labels_comb:
        aux = f'{i}'
        df[[f'tensao max. (kPa) - {aux}', f'tensao min. (kPa) - {aux}']] = df.apply(lambda row: calcular_sigma_max_min(row[f'Fz-{aux}'], row[f'Mx-{aux}'], row[f'My-{aux}'], row['h_x (m)'], row['h_y (m)']), axis=1, result_type='expand')
        df[f'g tensao max. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao max. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao min. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao min. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao - {aux}'] = df[[f'g tensao max. - {aux}', f'g tensao min. - {aux}']].max(axis=1)

    df['g tensao'] = df[[f'g tensao - {i}' for i in labels_comb]].max(axis=1)
    df['volume final (m3)'] = df['volume (m3)'] + df['g tensao'].clip(lower=0) * 1E6
    of = df['volume final (m3)'].sum()

    return of


def obj_teste(x, args):

    # Arguments
    df = args[0].copy()
    n_comb = args[1]
    # Design variables
    h_x, h_y = x
    h_z = 0.60
    df['h_x (m)'] = h_x
    df['h_y (m)'] = h_y
    df['h_z (m)'] = h_z

    # Volumn
    df['volume (m3)'] = df['h_x (m)'] * df['h_y (m)'] * df['h_z (m)']

    # Admissible stress of the soil
    df['tensao adm. (kPa)'] = df.apply(lambda row: tensao_adm_solo(row['solo'], row['spt']), axis=1)

    # Combination label
    labels_comb = [f'c{i}' for i in range(1, n_comb + 1)]


    # Computing max and min for all load combinations
    t_max_aux = []
    t_min_aux = []
    for i  in labels_comb:
        aux = f'{i}'
        df[[f'tensao max. (kPa) - {aux}', f'tensao min. (kPa) - {aux}']] = df.apply(lambda row: calcular_sigma_max_min(row[f'Fz-{aux}'], row[f'Mx-{aux}'], row[f'My-{aux}'], row['h_x (m)'], row['h_y (m)']), axis=1, result_type='expand')
        df[f'g tensao max. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao max. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao min. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao min. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao - {aux}'] = df[[f'g tensao max. - {aux}', f'g tensao min. - {aux}']].max(axis=1)

    df['g tensao'] = df[[f'g tensao - {i}' for i in labels_comb]].max(axis=1)
    df['volume final (m3)'] = df['volume (m3)'] + df['g tensao'].clip(lower=0) * 1E6
    of = df['volume final (m3)'].sum()

    return of, df
