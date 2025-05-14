"""Função objetivo para o problema de fundações"""
from itertools import combinations
import numpy as np
import pandas as pd
import math

#declarando apenas por estética

def restricao_tensao1(t_value: float, sigma_rd: float)-> float:
    """
    Esta função verifica a restrição de tensão na fundação rasa do tipo sapata, com majoração devido a incerteza do tipo de carregamento

    Args:
        sigma_rd (float): tensão resistente na direção z (KPa)
        sigma_sd (float): tensão solicitante na direção z (KPa)

    Returns:
        g (float): restrição de tensão (admensional)
    """
    
    if t_value >= 0:
        g = t_value * 1.30 / sigma_rd - 1 #1,30 = majoração devido a não saber se é o vento é ou não o carregamento variável principal
    else:
        g = -t_value / sigma_rd
    return g

def calcular_sigma_max(f_z: float, m_x: float, m_y: float, h_x: float, h_y: float) -> tuple[float, float]:
    """
    Esta função determina a tensão máxima e a tensão mínima na fundação rasa do tipo sapata

    Args
    f_z: Carregamento na direção z, da combinação mais desfavorável (kN)
    m_x: Momento em x da combinação mais desfavorável (kN.m)
    m_y: Momento em y da combinação mais desfavorável (kN.m)

    Returns
    sigma_max: Tensão máxima que age na sapata (kPa)
    sigma_min: Tensão minima que age na sapata (kPa)

    """
    
    m_x = abs(m_x)
    m_y = abs(m_y)
    sigma_fz = f_z / (h_x * h_y)
    aux_mx = 6 * (m_x / f_z) / h_x
    aux_my = 6 * (m_y / f_z) / h_y
    
    return (sigma_fz) * (1 + aux_mx + aux_my), (sigma_fz) * (1 - aux_mx - aux_my)

def volume_fundacao(h_x, h_y, h_z):
    '''
    Esta função cálcula o volume da sapata

    args:
        h_x (float): dimensão hx(m)
        h_y (float): dimensão hy(m)
        h_z (float): dimensão hz(m)

    Returns:
        volume (float): volume da sapata (m3)
    '''

    h_z= 0.6

    return h_x * h_y * h_z

def cargas_combinacoes(cargas: list) -> list:
    """
    Esta função determina os pares de carga de cada elemento de fundação considerando todas as condições possíveis para fz, mx e my.

    Args: 
        cargas (list): Lista de cargas de cada elemento de fundação
    
    Returns:
        cargas_comb (list): Lista de pares de carga de cada elemento de fundação
    """
    
    cargas_comb = list(combinations(cargas, 3))
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
    solo_column = df[('solo')]  
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

#Restricao de tensão
def restricao_tensao(h_x: float, h_y: float, none_variable, calcular_sigma_max):
    '''
   Esta função verifica se a tensão máxima é superior ou não a tensão admissível.
   
   args:
       h_x (float): dimensão em x (m)
       h_y (float): dimensão em y (m) 
       comb (list): contendo as combinações de carregamento
       sigma_lim (float): tensão admissível (kPa)

   Returns:
        result (float): valor da penalidade (admensional)
    '''

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

#Restiricao geometrica - Balanço
def restricao_geometrica_balanco_pilar_sapata(h_x: float, h_y: float, h_z: float, a_p: float, b_p: float) -> tuple[float, float, float , float]:
    """
    Esta função calcula o balanço da sapata e verifica se esta apto a ser calculado de acordo com o método CEB-70

    args:
        h_x (float): dimensões da sapata em x (m)
        h_y (float): dimensões da sapata em y (m)  
        a_p (float): dimensões dos pilares em x (m)
        b_p (float): dimensões dos pilares em y (m)
        
    returns:
        result (float): valor da penalidade (admensional)
    """

    # Balanço na direção X
    cap = (h_x - a_p) / 2

    # Balanço na direção Y
    cbp = (h_y - b_p) / 2
    
    # Restrições laterais do balanço
    g_0 = cap / (2 * h_z) - 1
    g_1 = cbp / (2 * h_z) - 1
    g_2 = (h_z / 2) / cap - 1
    g_3 = (h_z / 2) / cbp - 1
    
    return g_0, g_1, g_2, g_3
    
#Restiricao geometrica - Balanço
def restricao_geometrica_pilar_sapata(h_x: float, h_y: float, a_p: float, b_p: float) -> tuple[float, float]:
    """
    Esta função verifica se a dimensão do pilar é maior ou menor que a da sapata

    args:
        h_x (float): dimensões da sapata em x (m)
        h_y (float): dimensões da sapata em y (m)  
        a_p (float): dimensões dos pilares em x (m)
        b_p (float): dimensões dos pilares em y (m)
        
    returns:
        result (float): valor da penalidade (admensional)
    """

        
    #Restrição da dimensão do pilar em relação a dimensão da sapata
    g_4 = a_p / h_x - 1
    g_5 = b_p / h_y - 1

    return g_4, g_5

#Definição kx e ky para restrição de punção de acordo com NBR 6118
## para kx
def interpolar_kx(a_p: float, b_p: float) -> float:
    '''
    Esta função interpola o valor de Kx (coeficiente que indica a parcela do momento transmitida ao pilar na punçao)
    

    args:
        a_p (float): dimensões dos pilares em x (m)
        b_p (float): dimensões dos pilares em y (m)
    returns:
        kx_interpolado (float): valor de Kx interpolado (admensional) 
    '''
    
    b_a = b_p / a_p
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
    '''
    Esta função interpola o valor de Ky (coeficiente que indica a parcela do momento transmitida ao pilar na punçao)
    

    args:
        a_p (float): dimensões dos pilares em x (m)
        b_p (float): dimensões dos pilares em y (m)
    returns:
        ky_interpolado (float): valor de Ky interpolado (admensional)
    '''

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
    Esta função  calcula a tensão resistente e solicitante que há na borda do pilar e no perímetro crítico C' da sapata
    em seguida contempla a verificações de restrição de acordo com a NBR6118-2023

    Args:
        h_x (float): Altura da sapata na direção x (m)
        h_y (float): Altura da sapata na direção y (m)
        h_z (float): Altura util da sapata (m)
        a_p (float): Comprimento do pilar (m)
        b_p (float): Largura do pilar (m)
        f_z (float): Esforço axial (KN)
        m_x (float): Momento fletor na direção x (KN.m)
        m_y (float): Momento fletor na direção y (KN.m)
        ro (float): Densidade do concreto (adimensional)
        rec (float): Comprimento de recobramento da sapata (m) 
        fck (float): Resistência característica a compressão do concreto (MPa)
        fcd (float): Resistência de projeto a compressão do concreto (MPa)

    Returns:
       g6, g7, g8, g9, g10, g11, g12 (tuple[float, float, float, float, float, float, float]): verificação das restrições 
       da linha critica, da taxa de aço de flexão aderenente, da tensão de protensão e das tensões em C e C'
    """
    d = h_z - cob #altura util da sapata
    sigma_cp = 0 # tensão a mais devido a efeitos da protensão do concreto <= 3,5 MPa, depois criar uma def para calcular essa tensão!
    # taxa geométrica de armadura de flexão aderente <0,02 (NBR6118-2023)
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

    g_6 = (h_x - a_p) / 4 * d - 1 #4 * d / (h_x - a_p) - 1 #se a area crítica esta dentro da sapata em x adiciona punição
    g_7 = (h_y - b_p) / 4 * d - 1 #4 * d / (h_y - b_p) - 1 #se a area crítica esta dentro da sapata em y adiciona punição
    g_8 = ro / 0.02 - 1 # taxa de aço de flexão aderente, 
    g_9 =  ke / 2 - 1 
    g_10 = sigma_cp / 3.5 - 1
    g_11 = talsd1 / talrd1 - 1
    g_12 = talsd2 / talrd2 - 1
    
    return  g_6, g_7, g_8, g_9, g_10, g_11, g_12
    
def obj_ic_fundacoes(x, none_variable):
    h_x = x[0]
    h_y = x[1]
    h_z = 0.6
    cob = none_variable['cob (m)']
    n_comb = none_variable['número de combinações estruturais']
    ro = 0.01
    df = none_variable['dados estrutura']
    fck = none_variable['fck (kPa)']
    vol = 0 
    n_comb = none_variable['número de combinações estruturais']
    g = []
    t_max = []
    t_min = []
    fz_list = []
    mx_list = []
    my_list = []  

    # Volume total da fundação
    for _ in range(len(df)):
        v_aux = volume_fundacao(h_x, h_y, h_z)
        vol += v_aux
    
    # determinando a combinação mais desfavorável
    for _, row in df.iterrows():
        for i in range(1, n_comb+1):
            aux = f'c{i}'
            t_max_aux, t_min_aux = calcular_sigma_max(row[f'Fz-{aux}'], row[f'Mx-{aux}'], row[f'My-{aux}'], h_x, h_y)
            t_max.append(t_max_aux)
            t_min.append(t_min_aux)
            if t_max_aux >= t_max[-1]:
                fz_aux = row[f'Fz-{aux}']
                mx_aux = row[f'Mx-{aux}']
                my_aux = row[f'My-{aux}']
                fz_list.append(fz_aux)
                mx_list.append(mx_aux)
                my_list.append(my_aux)
        f_z = max(fz_list)
        m_x = max(mx_list)
        m_y = max(my_list)
        t_max_value = max(t_max)
        t_min_value = min(t_min)
        
        print(f_z, m_x, m_y)
        print(t_max_value, t_min_value)

        a_p = row['ap (m)']
        b_p = row['bp (m)']
        fcd = fck / 1.4
        sigma_rd = row['sigma_adm (kPa)']

    # verificando as restrições
        g_0, g_1, g_2, g_3 = restricao_geometrica_balanco_pilar_sapata(h_x, h_y, h_z, a_p, b_p)
        g_4, g_5 = restricao_geometrica_pilar_sapata(h_x, h_y, a_p, b_p)
        g_6, g_7, g_8, g_9, g_10, g_11, g_12 = restricao_puncao(h_x, h_y, h_z, a_p, b_p, f_z, m_x, m_y, ro, cob, fck, fcd)
        g_13 = restricao_tensao1(t_max_value, sigma_rd)
        g.append(g_0)
        g.append(g_1)
        g.append(g_2)
        g.append(g_3)
        g.append(g_4)
        g.append(g_5)
        g.append(g_6)
        g.append(g_7)
        g.append(g_8)
        g.append(g_9)
        g.append(g_10)
        g.append(g_11)
        g.append(g_12)
        g.append(g_13)
        
    # Função pseudo-objetivo
    of = vol
    for i in g:
        of += max(0, i) * 1E6
    
    return of

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
'''
#esse if esta sendo usado para testar o codigo mas não traz valores corretos
if __name__ == 'IC_Filipe':
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
'''

if __name__== '__main__':
    df = pd.read_excel("teste_wand.xlsx")
    df = tensao_adm_solo(df)
    a = 0.6
    b = 0.6
    x = [a, b]
    none_variable = {'dados estrutura': df, 'h_z (m)': 0.6, 'cob (m)': 0.025, 'fck (kPa)': 25000, 'número de combinações estruturais': 3}
    of = obj_ic_fundacoes(x, none_variable)

    print(of)