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
    # Converta a coluna 'solo' para minúsculas
    solo_column = df[('solo', 'Unnamed: 2_level_1')]  
    solo_column = solo_column.str.lower()

    # Verifique se a coluna 'spt' existe para evitar erro
    if 'spt' not in df.columns:
        raise KeyError("A coluna 'spt' deve estar presente no DataFrame.")

    # Calcula a tensão admissível com base no tipo de solo
    condicoes = [
        solo_column == 'pedregulho',
        solo_column == 'areia',
        (solo_column == 'silte') | (solo_column == 'argila'),
    ]
    values = [
        df[('spt', 'Unnamed: 1_level_1')] / 30 * 1E3,  # Acessa a coluna 'spt' corretamente
        df[('spt', 'Unnamed: 1_level_1')] / 40 * 1E3,
        df[('spt', 'Unnamed: 1_level_1')] / 50 * 1E3,
    ]

    # Assegure que as condições e os valores sejam arrays 1D
    condicoes = [condicionamento.values for condicionamento in condicoes]
    values = [valor.values for valor in values]

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
    comb = none_variable['combinações']
    sigma_lim = none_variable['sigma_adm (kPa)']

    # Determina o volume do elemento de fundação
    vol = volume_fundacao(h_x, h_y)

    # Verificação da restrição
    g = []
    for key, values in comb.items():
        print(values)
        f_z = values[0]
        m_x = values[1]
        m_y = values[2]
        sigma_sd_max, sigma_sd_min = calcular_sigma_max(f_z, m_x, m_y, h_x, h_y)  # Retorna um tuple
        g.append(sigma_sd_max / sigma_lim - 1)  # Usa o valor máximo como o mais crítico

    # Função objetivo e restrições
    of = vol
    for i in g:
        of += max(0, i) * 1E6

    return of

def data_comb(df: pd.DataFrame) -> list:
    """
    Gera combinações 3 a 3 das colunas de uma planilha com header duplo,
    identificando as combinações automaticamente a partir do DataFrame,
    e retorna uma lista de dicionários no formato desejado, descartando
    colunas que não contêm 'combinação' no nome e valores nulos.

    Parâmetros:
    - df: DataFrame carregado com header duplo.

    Retorno:
    - Uma lista de dicionários contendo os valores das combinações, sem valores nulos.
    """
    # Filtrar as colunas que contêm a palavra "combinação" no nome
    df_filtered = df.loc[:, df.columns.get_level_values(0).str.contains("combinação", case=False)]

    # Identificar combinações automaticamente a partir do header
    combinacoes_disponiveis = {}
    for nome_combinacao, coluna in df_filtered.columns:
        if nome_combinacao not in combinacoes_disponiveis:
            combinacoes_disponiveis[nome_combinacao] = []
        combinacoes_disponiveis[nome_combinacao].append(coluna)

    # Filtrar apenas combinações que possuem todas as colunas ['Fz', 'Mx', 'My']
    combinacoes_filtradas = {
        nome: colunas for nome, colunas in combinacoes_disponiveis.items()
        if all(campo in colunas for campo in ['Fz', 'Mx', 'My'])
    }

    lista_resultados = []
    for _, row in df.iterrows():
        resultado_linha = {}
        for nome_combinacao, colunas_desejadas in combinacoes_filtradas.items():
            colunas_multiindex = [(nome_combinacao, coluna) for coluna in ['Fz', 'Mx', 'My']]
            # Filtrar os valores nulos (NaN) antes de adicionar à lista de resultados
            valores = [row[coluna] for coluna in colunas_multiindex if pd.notnull(row[coluna])]
            if valores:  # Só adiciona se houver valores não nulos
                resultado_linha[nome_combinacao] = valores
        
        if resultado_linha:  # Só adiciona linhas não vazias
            lista_resultados.append(resultado_linha)

    return lista_resultados

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