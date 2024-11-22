"""Função objetivo para o problema de fundações"""
from itertools import combinations
import numpy as np
import pandas as pd


def calcular_sigma_max(f_z: float, m_x: float, m_y: float, h_x: float, h_y: float) -> tuple[float, float]:
    """
    """
    m_x = abs(m_x)
    m_y = abs(m_y)
    sigma_fz = f_z / (h_x * h_y)
    aux_mx = 6 * (m_x / f_z) / h_x
    aux_my = 6 * (m_y / f_z) / h_y
    
    return (sigma_fz) * (1 + aux_mx + aux_my), (sigma_fz) * (1 - aux_mx - aux_my)


def volume_fundacao(h_x: float, h_y: float, h_z: float=0.60) -> float:
    """
    """
    return h_x * h_y * h_z


def cargas_combinacoes(cargas: list) -> list:
    """
    Esta função determina os pares de carga de cada elemento de fundação considerando todas as condições possíveis para fz, mx e my.

    Args: 
        cargas (list): Lista de cargas de cada elemento de fundação
    
    Returns:
        cargas_comb (list): Lista de pares de carga de cada elemento de fundação
    """
    
    cargas_comb = list(combinations(cargas, 3)) # Aqui tem erro
    return [list(comb) for comb in cargas_comb]


def tensao_adm_solo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a tensão admissível do solo.

    Args:
        df (DataFrame): DataFrame com os dados de entrada, contendo as colunas 'spt' e 'solo'.

    Returns:
        DataFrame: DataFrame com a coluna 'sigma_adm (kPa)' calculada.
    """
    # Verifica se as colunas necessárias estão presentes no DataFrame
    if 'spt' not in df.columns or 'solo' not in df.columns:
        raise KeyError("As colunas 'spt' e 'solo' devem estar presentes no DataFrame.")
    
    # Calcula a tensão admissível com base no tipo de solo
    condicoes = [
        df['solo'] == 'pedregulho',
        df['solo'] == 'areia',
        (df['solo'] == 'silte') | (df['solo'] == 'argila'),
    ]
    values = [
        df['spt'] / 30 * 1E3,
        df['spt'] / 40 * 1E3,
        df['spt'] / 50 * 1E3,
    ]
    
    # Cria a nova coluna com np.select
    df['sigma_adm (kPa)'] = np.select(condicoes, values, default=np.nan)
    
    return df

     
def obj_ic_fundacoes(x, none_variable):
    """
    Calcula a função objetivo para o problema de fundações.
    
    Args:
        x (list): Lista com as variáveis de projeto [h_x, h_y].
        none_variable (dict): Dicionário contendo os dados de entrada. 
                              Deve conter a chave 'dados' com o DataFrame.
    
    Returns:
        float: Valor da função objetivo.
    """
    # Variáveis de projeto
    h_x = x[0]
    h_y = x[1]
    f_zmax = none_variable['Fz,max (kN)']
    f_zmin = none_variable['Fz,min (kN)']
    m_xmax = none_variable['Mx,max (kN.m)']
    m_xmin = none_variable['Mx,min (kN.m)']
    m_ymax = none_variable['My,max (kN.m)']
    m_ymin = none_variable['My,min (kN.m)']
    sigma_lim = none_variable['sigma_adm (kPa)']

    # Determina o volume do elemento de fundação
    vol = volume_fundacao(h_x, h_y)

    # Criar dicionários para cada carga
    cargas_comb = cargas_combinacoes([f_zmax, f_zmin, m_xmax, m_xmin, m_ymax, m_ymin])

    # Verificação da restrição
    g = []
    for j in cargas_comb:
        f_z = j[0]
        m_x = j[1]
        m_y = j[2]
        sigma_sd_max, sigma_sd_min = calcular_sigma_max(f_z, m_x, m_y, h_x, h_y)  # Retorna um tuple
        g.append(sigma_sd_max / sigma_lim - 1)  # Usa o valor máximo como o mais crítico

    # Função objetivo e restrições
    of = vol
    for i in g:
        of += max(0, i) * 1E6

    return of



if __name__ == '__main__':
    import pandas as pd
    df = pd.read_excel('input.xlsx')
    x = [1, 1]
    none_variable = {'Fz,max (kN)': 750,
                     'Fz,min (kN)': 720,
                     'Mx,max (kN.m)': 325,
                     'Mx,min (kN.m)': -200,
                     'My,max (kN.m)': 300,
                     'My,min (kN.m)': -300,
                     'sigma_adm (kPa)': 333.33}
    print(obj_ic_fundacoes(x, none_variable))