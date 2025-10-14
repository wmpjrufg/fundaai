"""Função objetivo para o problema de fundações"""
from itertools import combinations
import numpy as np
import pandas as pd
import math


# def restricao_tensao_solo(t_value: float, sigma_rd: float)-> float:
#     """
#     Verifica a restrição de tensão solicitante máxima na sapata, com majoração de 30% para incertezas na combinação de ações (ex: ação do vento).

#     :param t_value: Valor absoluto da tensão solicitante mais crítica (kPa)
#     :param sigma_rd: Tensão admissível do solo (kPa)

#     :return: Valor da restrição (g), onde g <= 0 é a situação aceitável
#     """

#     return t_value * 1.30 / sigma_rd - 1


# def volume_fundacao(h_x: float, h_y: float, h_z: float) -> float:
#     """
#     Calcula o volume de concreto da sapata.

#     :param h_x: Largura da sapata (m).
#     :param h_y: Comprimento da sapata (m).
#     :param h_z: Altura da sapata (m).

#     :returns: saida[0] = volume da sapata (m3)
#     """

#     h_z= 0.6 # valor definido e fixado provisóriamente para efeito de construção da ferramenta

#     return h_x * h_y * h_z


# def cargas_combinacoes(cargas: list) -> list:
#     """
#     Gera todas as combinações possíveis de cargas com Fz, Mx e My.

#     :param Cargas:  Lista com os valores individuais de cargas dos elemntos de fundação.
    
#     :return: saida[0] = Lista de trios de combinações.
#     """
    
#     cargas_comb = list(combinations(cargas, 3))

#     return [list(comb) for comb in cargas_comb]


def tensao_adm_solo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a tensão admissível do solo com base no tipo de solo e no Nspt.

    :param df: dados de entrada, contendo colunas com o 'spt' e o tipo de 'solo'
    :return: Tensão max admissível do solo em kPa
    """

    # Converta a coluna 'solo' para minúsculas
    solo_column = df[('solo')] 
    solo_column = solo_column.str.lower()
    
    # Calcula a tensão admissível com base no tipo de solo
    condicoes = [
                    solo_column == 'pedregulho',
                    solo_column == 'areia',
                    (solo_column == 'silte') | (solo_column == 'argila'),
                ]
    values = [
                df[('spt')] / 30 * 1E3,  # Acessa a coluna 'spt' corretamente
                df[('spt')] / 40 * 1E3,
                df[('spt')] / 50 * 1E3,
             ]

    # Assegure que as condições e os valores sejam arrays 1D
    condicoes = [condicionamento.values for condicionamento in condicoes]
    values = [valor.values for valor in values]

    # Cria a nova coluna com np.select
    df['sigma_adm (kPa)'] = np.select(condicoes, values, default=np.nan)

    return df


def calcular_sigma_max(f_z: float, m_x: float, m_y: float, h_x: float, h_y: float) -> tuple[float, float]:
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


def restricao_tensao(h_x: float, h_y: float, none_variable: dict, calcular_sigma_max)-> list[float]:
    """
    Aplica a verificação de tensão atuante máxima sobre a sapata em comparação com a tensão admissível do solo.

    :param h_x: Largura da sapata (m).
    :param h_y: Comprimento da sapata (m).
    :param none_variable: Dicionário com 'combinações' e 'sigma_adm (kPa)'.
    :param calcular_sigma_max: Função que calcula a tensão max.

    :returns: saida[0] = Lista de valores de restrição (g) por combinação.
    """

    comb = none_variable['combinações'] #é preciso fazer o for duplo
    sigma_lim = none_variable['sigma_adm (kPa)']

    # Verificação da restrição
    result = []
    for key, values in comb.items():
        f_z = values[0]
        m_x = values[1]
        m_y = values[2]
        sigma_sd_max, sigma_sd_min = calcular_sigma_max(f_z, m_x, m_y, h_x, h_y)  # Retorna um tuple
        result.append(sigma_sd_max / sigma_lim - 1)  # Usa o valor máximo como o mais crítico
    
    for i in result:
        result += max(0, i)

    return result


def restricao_geometrica_balanco_pilar_sapata(h_x: float, h_y: float, h_z: float, a_p: float, b_p: float) -> tuple[float, float, float, float, float , float]:
    """
    Verifica o balanço da sapata segundo os limites geométricos do método CEB-70.

    :param h_x: dimensões da sapata em x (m)
    :param h_y: dimensões da sapata em y (m)  
    :param a_p: dimensões dos pilares em x (m)
    :param b_p: dimensões dos pilares em y (m)
        
    :returns: saida[0]  = valor do balanço em x (m)
    :returns: saida[1]  = valor do balanço em y (m)
    :returns: saida [2, 3, 4, 5] = valor da penalidade para cada restrição (admensional)
    """

    # Balanço na direção X
    cap_value = (h_x - a_p) / 2

    # Balanço na direção Y
    cbp_value = (h_y - b_p) / 2
    
    # Restrições laterais do balanço
    g_0 = cap_value / (2 * h_z) - 1
    g_1 = cbp_value / (2 * h_z) - 1
    g_2 = (h_z / 2) / cap_value - 1
    g_3 = (h_z / 2) / cbp_value - 1
    
    return [cap_value, cbp_value, g_0, g_1, g_2, g_3]
    

def restricao_geometrica_pilar_sapata(h_x: float, h_y: float, a_p: float, b_p: float) -> tuple[float, float]:
    """
    Verifica se as dimensões do pilar são maiores que as dimensões da sapata

    :param h_x: dimensões da sapata em x (m)
    :param h_y: dimensões da sapata em y (m)  
    :param a_p: dimensões dos pilares em x (m)
    :param b_p: dimensões dos pilares em y (m)
        
    :returns: saida [0] = valor da penalidade (admensional)
    :returns: saida [1] = valor da penalidade (admensional)
    """
    
    # Restrição da dimensão do pilar em relação a dimensão da sapata
    g_4 = a_p / h_x - 1
    g_5 = b_p / h_y - 1

    return g_4, g_5


#Definição kx e ky para restrição de punção de acordo com NBR 6118
## para kx
def interpolar_kx(a_p: float, b_p: float) -> float:
    """
    Interpola o valor de Kx com base na proporção do pilar, segundo nbr 6118:2023
    (coeficiente que indica a parcela do momento transmitida ao pilar na punção)
    
    :param a_p : dimensões dos pilares em x (m)
    :param b_p : dimensões dos pilares em y (m)
    :returns: saida[0] = Coeficiente interpolado (adimensional).
    """
    
    b_a = b_p / a_p
    # Tabela 19.2 de valores conhecidos da NBR 6118:2023
    tabela = {
        0.5: 0.45,
        1.0: 0.60,
        2.0: 0.70,
        3.0: 0.80
    }

    # Ordena os pontos conhecidos para interpolação
    chaves = sorted(tabela.keys())

    # Se estiver abaixo do menor valor da tabela, retorna o menor K
    if b_a < chaves[0]:
        return tabela[chaves[0]]

    # Se estiver acima do maior valor da tabela, retorna o maior K
    if b_a > chaves[-1]:
        return tabela[chaves[-1]]

    # Se o valor estiver na tabela, retorna diretamente
    if b_a in tabela:
        return tabela[b_a]

    # Procura os dois pontos mais próximos para interpolação
    for i in range(len(chaves) - 1):
        x0, x1 = chaves[i], chaves[i + 1]
        if x0 <= b_a <= x1:
            y0, y1 = tabela[x0], tabela[x1]
            # Interpolação linear
            kx_interpolado = y0 + (y1 - y0) * ((b_a - x0) / (x1 - x0))
            return round(kx_interpolado, 4)

##para ky
def interpolar_ky(a_p: float, b_p: float):
    """
    Esta função interpola o valor de Ky com base nas proporções do pilar, segundo nbr 6118:2023
    (coeficiente que indica a parcela do momento transmitida ao pilar na punção)
    
    :param a_p: dimensões dos pilares em x (m)
    :param b_p: dimensões dos pilares em y (m)

    :returns: saida[0] = Coeficiente interpolado (adimensional).
    """

    a_b = a_p / b_p

    # Tabela de valores conhecidos da NBR...
    tabela = {
        0.5: 0.45,
        1.0: 0.60,
        2.0: 0.70,
        3.0: 0.80
    }

    # Ordena os pontos conhecidos para interpolação
    chaves = sorted(tabela.keys())

    # Se estiver abaixo do menor valor da tabela, retorna o menor K
    if a_b < chaves[0]:
        return tabela[chaves[0]]

    # Se estiver acima do maior valor da tabela, retorna o maior K
    if a_b > chaves[-1]:
        return tabela[chaves[-1]]

    # Se o valor estiver na tabela, retorna diretamente
    if a_b in tabela:
        return tabela[a_b]

    # Procura os dois pontos mais próximos para interpolação
    for i in range(len(chaves) - 1):
        x0, x1 = chaves[i], chaves[i + 1]
        if x0 <= a_b <= x1:
            y0, y1 = tabela[x0], tabela[x1]
            # Interpolação linear
            ky_interpolado = y0 + (y1 - y0) * ((a_b - x0) / (x1 - x0))
            return round(ky_interpolado, 4)


def restricao_puncao(h_x: float, h_y: float, h_z: float, a_p: float, b_p: float, f_z: float, m_x: float, m_y: float, ro: float, cob: float, fck: float, fcd: float) -> tuple[float, float, float, float, float, float, float,]:
    """
    Calcula a tensão resistente e solicitante que há na borda do pilar e no perímetro crítico C' da sapata em seguida contempla a verificações de restrição de acordo com a NBR6118-2023

    :param h_x: Altura da sapata na direção x (m)
    :paramh_y: Altura da sapata na direção y (m)
    :param h_z: Altura util da sapata (m)
    :param a_p: Comprimento do pilar (m)
    :param b_p: Largura do pilar (m)
    :param f_z: Esforço axial (KN)
    :param m_x: Momento fletor na direção x (KN.m)
    :param m_y: Momento fletor na direção y (KN.m)
    :param ro: taxa de armadura aderente (adimensional)
    :param cob: Comprimento (m) 
    :param fck: Resistência característica (MPa)
    :param fcd: Resistência de projeto (MPa)

    :returns: saida[0] = verificação das restrições da linha critica, da taxa de aço de flexão aderenente, da tensão de protensão e das tensões em C e C'
    """

    d = h_z - cob #altura util da sapata
    sigma_cp = 0 # tensão a mais devido a efeitos da protensão do concreto <= 3,5 MPa, depois criar uma def para calcular essa tensão!
    ke = 1 + math.sqrt(20 / (d * 100))  
    kx = interpolar_kx(a_p, b_p)
    ky = interpolar_ky(a_p, b_p)
    
    wpx = b_p**2 / 2 + b_p * a_p + 4 * a_p * d + 16 * d**2 + 2 * math.pi * b_p * d #módulo de resistencia plastica no perímetro crítico na direção x
    wpy = a_p**2 / 2 + a_p * b_p + 4 * b_p * d + 16 * d**2 + 2 * math.pi * a_p * d #módulo de resistencia plastica no perímetro crítico na direção y
    u = 2 * (a_p + b_p + math.pi * d) # perímero do contorno C'
    
    talsd2 = 0.001 * f_z / (2 * a_p + b_p * d) # (MPa)
    talrd2 = 0.27 * (1 - fck / 250) * fcd # (MPa)
    talsd1 = (0.001 * f_z) / (u * d) + kx * m_x * 0.001 / (wpx * d) + ky * m_y * 0.001 / (wpy * d) # (MPa)
    talrd1 = 0.13 * ke * (100 * ro * fck) ** (1 / 3) + 0.1 * sigma_cp # (MPa)

    g_6 = (h_x - a_p) / 4 * d - 1 #4 * d / (h_x - a_p) - 1 #se a area crítica esta dentro da sapata em x adiciona punição (preciso verificar isso!)
    g_7 = (h_y - b_p) / 4 * d - 1 #4 * d / (h_y - b_p) - 1 #se a area crítica esta dentro da sapata em y adiciona punição (preciso verificar isso!)
    g_8 = ro / 0.02 - 1 # taxa de aço de flexão aderente, precisa ser calculado
    g_9 =  ke / 2 - 1 
    g_10 = sigma_cp / 3.5 - 1
    g_11 = talsd1 / talrd1 - 1
    g_12 = talsd2 / talrd2 - 1
    
    return  g_6, g_7, g_8, g_9, g_10, g_11, g_12


def restricao_geometrica_sobreposicao(df: pd.DataFrame, h_x: float, h_y: float, idx: int)-> float:
    """
    Verifica a área de sobreposição da sapata (em sapata_index)
    com as demais sapatas do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame com posições das sapatas.
        h_x, h_y (float): Dimensões da sapata (m).
        idx (int): Índice da sapata atual.
    Retorna:
        float: penalização proporcional à área de sobreposição.
    """
    area_total = 0
    xi, yi = df.loc[idx, 'xg (m)'], df.loc[idx, 'yg (m)']

    xi_min, xi_max = xi - h_x / 2, xi + h_x / 2
    yi_min, yi_max = yi - h_y / 2, yi + h_y / 2

    for j, row in df.iterrows():
        if j == idx:
            continue  # Ignorar a própria sapata

        xj, yj = row['xg (m)'], row['yg (m)']
        xj_min, xj_max = xj - h_x / 2, xj + h_x / 2# Tá sobrando? pq ja foi calcualdo fora do loop
        yj_min, yj_max = yj - h_y / 2, yj + h_y / 2# Tá sobrando? pq ja foi calcualdo fora do loop

        # Calcular sobreposição
        overlap_x = max(0, min(xi_max, xj_max) - max(xi_min, xj_min))
        overlap_y = max(0, min(yi_max, yj_max) - max(yi_min, yj_min))
        area_overlap = overlap_x * overlap_y

        area_total += area_overlap

    return area_total / (h_x * h_y)  # penalização normalizada


def obj_ic_fundacoes(x: list, none_variable: dict)-> float:
    """
    Função objetivo para a otimização do dimensionamento de fundações rasas do tipo sapata.

    Esta função calcula o volume total das fundações e aplica penalizações associadas
    a violações de restrições geométricas, de punção e de tensões admissíveis,
    retornando um valor de função objetivo a ser minimizado por um algoritmo de otimização.

    Parâmetros:
        x (List[float]): Vetor com as variáveis de projeto (dimensões da sapata) [h_x, h_y].
        none_variable (dict): Dicionário contendo parâmetros auxiliares e dados estruturais:
            - 'cob (m)' (float): Cobrimento mínimo do concreto.
            - 'fck (kPa)' (float): Resistência característica do concreto.
            - 'número de combinações estruturais' (int): Número de combinações de carregamento.
            - 'dados estrutura' (DataFrame): Tabela com dados dos pilares e esforços.
            - 'h_z (m)' (float): Altura da sapata.

    Retorna:
        Tuple[float, pd.DataFrame]:
            - float: Valor da função objetivo (volume com penalização por restrições).
            - DataFrame: Tabela com os resultados de dimensionamento por pilar.
    """
    h_x = x[0]
    h_y = x[1]
    h_z = 0.6
    cob = none_variable['cob (m)']
    n_comb = none_variable['número de combinações estruturais']
    ro = 0.01 #esse valor deve ser calculado
    df = none_variable['dados estrutura']
    df = tensao_adm_solo(df)
    fck = none_variable['fck (kPa)']
    vol = 0

    t_max = []
    t_min = []
    fz_list = []
    mx_list = []
    my_list = []
    lista_restricoes = [] 

    h_x = []
    h_y = []
    h_z = [0.6] * len(df)
    for i in range(0, len(x), 2):
        print(i)
        h_x.append(x[i])
        h_y.append(x[i+1])

    # Volume total da fundação
    for i in range(len(df)):
        v_aux = volume_fundacao(h_x[i], h_y[i], h_z[i])
        vol += v_aux
    
    # Determinando a combinação mais desfavorável
    g_tensao = []
    for idx, row in df.iterrows():
        t_max_aux = []
        t_min_aux = []
        for i in range(1, n_comb+1):
            aux = f'c{i}'
            t_max_val, t_min_val = calcular_sigma_max(row[f'Fz-{aux}'], row[f'Mx-{aux}'], row[f'My-{aux}'], h_x, h_y)
            t_max_aux.append(t_max_val)
            t_min_aux.append(t_min_val)
        # t_max.append(max(t_max_aux))
        # t_min.append(min(t_min_aux))

        sigma_rd = row['sigma_adm (kPa)']
        g_tensao.append(restricao_tensao_solo(max(t_max_aux), sigma_rd))
        
    # for idx, row in df.iterrows():
    #     for i in range(1, n_comb+1):
    #         aux = f'c{i}'
    #         t_max_aux, t_min_aux = calcular_sigma_max(row[f'Fz-{aux}'], row[f'Mx-{aux}'], row[f'My-{aux}'], h_x, h_y)
    #         t_max.append(t_max_aux)
    #         t_min.append(t_min_aux)
    #         # if para garantir que os valores de FZ mx e my sejam correspondente à combinação mais desfavorável
    #         if t_max_aux >= t_max[-1]:
    #             fz_aux = row[f'Fz-{aux}']
    #             mx_aux = row[f'Mx-{aux}']
    #             my_aux = row[f'My-{aux}']
    #             fz_list.append(fz_aux)
    #             mx_list.append(mx_aux)
    #             my_list.append(my_aux)
    #     f_z = max(fz_list)
    #     m_x = max(mx_list)
    #     m_y = max(my_list)
    #     t_max_value = max(t_max)
    #     t_min_value = min(t_min)
    #     t_value = max(abs(t_max_value), abs(t_min_value))

    #     a_p = row['ap (m)']
    #     b_p = row['bp (m)']
    #     fcd = fck / 1.4
    #     sigma_rd = row['sigma_adm (kPa)']

    #     # verificando as restrições
    #     g_0, g_1, g_2, g_3 = restricao_geometrica_balanco_pilar_sapata(h_x, h_y, h_z, a_p, b_p)
    #     g_4, g_5 = restricao_geometrica_pilar_sapata(h_x, h_y, a_p, b_p)
    #     g_6, g_7, g_8, g_9, g_10, g_11, g_12 = restricao_puncao(h_x, h_y, h_z, a_p, b_p, f_z, m_x, m_y, ro, cob, fck, fcd)
    #     g_13 = restricao_tensao1(t_value, sigma_rd)
    #     g_14 = restricao_geometrica_sobreposicao(df, h_x, h_y, idx)

    #     restricoes = [g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8, g_9, g_10, g_11, g_12, g_13, g_14]
    #     lista_restricoes.append(restricoes)

    # Penalização no volume
    penalizacao = sum([sum(max(0, g) * 1e6 for g in linha) for linha in lista_restricoes])
    of = vol + penalizacao

    # Criação de DataFrame com as restrições
    colunas_g = [f'g_{i}' for i in range(15)]
    df_restricoes = pd.DataFrame(lista_restricoes, columns=colunas_g)

    df_resultado = pd.concat([df.reset_index(drop=True), df_restricoes], axis=1)

    return of, df_resultado


def data_comb(df: pd.DataFrame,) -> list:
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
    for _, row in df.iterrows(): #aqui esta correto
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


if __name__== '__main__':
    df = pd.read_excel("teste_wand.xlsx")
    df = tensao_adm_solo(df)
    a = 0.6
    b = 0.6
    x = [a, b]
    none_variable = {'dados estrutura': df, 'h_z (m)': 0.6, 'cob (m)': 0.025, 'fck (kPa)': 25000, 'número de combinações estruturais': 3}
    of = obj_ic_fundacoes(x, none_variable)
    
    print(of)