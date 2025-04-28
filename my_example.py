"""Função objetivo para o problema de fundações"""
from itertools import combinations
import numpy as np
import pandas as pd
import math

#declarando apenas por estética
fck = []
a = []
b = []
A = []
B = []
rec = []
MX = []
MY = []
Fz = []
dimensoes_sapata_a = []
dimensoes_sapata_b = []
dados_fundacao = []
h_x = []
h_y = []
h_z = []
none_variable = []
of = []

def calcular_sigma_max(f_z: float, m_x: float, m_y: float, h_x: float, h_y: float) -> tuple[float, float]:
    """
    Esta função determina a tensão máxima e a tensão mínima na fundação rasa do tipo sapata

    Args
    f_z: carregamento na direção z (kN)

    Returns
    sigma_max: Tensão máxima suportada pela sapata (kN/m2)
    sigma_min: Tensão minima suportada pela sapata (kN/m2)

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
    solo_column = df[('solo', 'Unnamed: 4_level_1')]  
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
        df[('spt', 'Unnamed: 3_level_1')] / 30 * 1E3,  # Acessa a coluna 'spt' corretamente
        df[('spt', 'Unnamed: 3_level_1')] / 40 * 1E3,
        df[('spt', 'Unnamed: 3_level_1')] / 50 * 1E3,
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

    # Variáveis de projeto
    h_x = x[0]
    h_y = x[1]
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
    Esta função calcula o balanço da sapata e verifica se esta de acordo as dimensões permitidas

    args:
        h_x (float): dimensões da sapata em x (m)
        h_y (float): dimensões da sapata em y (m)  
        a_p (float): dimensões dos pilares em x (m)
        b_p (float): dimensões dos pilares em y (m)
        
    returns:
        result (float): valor da penalidade (admensional)
    """

    # Balanço na direção X
    cap = (h_x - a_p)/2

    # Balanço na direção Y
    cbp = (h_y - b_p)/2
    
    # Restrições laterais do balanço
    g_0 = (2 * h_z) / cap  - 1
    g_1 = (2 * h_z) / cbp  - 1
    g_2 = (h_z / 2) / cap - 1
    g_3 = (h_z / 2) / cbp - 1
    
    return g_0, g_1, g_2, g_3
    
#Restiricao geometrica - Balanço
def restricao_geometrica_pilar_sapata_b(h_x: float, h_y: float, h_z: float, a_p: float, b_p: float) -> tuple[float, float]:
    """
    Esta função calcula o balanço da sapata e verifica se esta de acordo as dimensões permitidas

    args:
        h_x (float): dimensões da sapata em x (m)
        h_y (float): dimensões da sapata em y (m)  
        a_p (float): dimensões dos pilares em x (m)
        b_p (float): dimensões dos pilares em y (m)
        
    returns:
        result (float): valor da penalidade (admensional)
    """

    # Balanço na direção X
    ap <= hx
    bp <= hy
    
    return g_0, g_1

    
#Definição kx e ky para restrição de punção de acordo com NBR ....
## para kx
def interpolar_kx(a: float, b: float):
    '''
    Esta função interpola o valor de Kx com base nas dimensões do pilar, necessário para a restrição de punção

    args:
        none_variable (dicionário): dicionário com as dimensões 'a' e 'b' da sapata
    returns:
        kx_interpolado (float): valor de Kx interpolado    
    '''
    a = none_variable ['dados estrutura'] ['ap (m)']
    b = none_variable ['dados estrutura'] ['bp (m)']
    b_a = a / b
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
def interpolar_ky(a: float, b: float):
    '''
    Esta função interpola o valor de Ky com base nas dimensões do pilar, para restrição de punção

    args:
        none_variable (dicionário): dicionário com as dimensões 'a' e 'b' da sapata
    returns:
        ky_interpolado (float): valor de Ky interpolado
    '''

    a = none_variable ['dados estrutura'] ['ap (m)']
    b = none_variable ['dados estrutura'] ['bp (m)']
    a_b = a / b

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

#restrição de punção para pilar central
def restricao_puncao(A: float, B: float, a: float, b: float, h_z: 0.6, rec: float):
    '''
    Esta função verifica a se há esforço de punção na sapata e atibui ou não uma penalidade 
    (Não é válida para fundações de canto, ou seja com o pilar na latera da sapata) 

    Args:
        x (list): Dimensões A e B (h_x e h_y) da sapata (m)
        none_variable (dict): Dicionário com as dimensões do pilar  da sapata (m)
        kx e ky (float): Coeficientes de restrição de punção (admensional)
        ro (float): Densidade do concreto (kg/m3)
        fcd (float): Resistencia a copressão do concreto (..)
        ...... : combinação dos esforços
        .....
    returns:
        g1 ou g2, (float): Maior valor da penalidades para linha critica 1 ou 2, da punção (admensional)
    
    '''
    
    #DImensões da sapata
    A= x[0] 
    B= x[1] 
    #Dimensões do pilar
    a = none_variable ['dados estrutras']['ap (m)']
    b = none_variable ['dados estrutras']['bp (m)']
    rec = none_variable ['rec (m)']
    d = h_z - rec #Altura util da sapata d = h - rec, 0,04 é um valor qualquer que deve ser especificado , quando tiver formato inclinado, d pode assumir valores diferentes em C e C'
    MX = dados_fundacao[1]['combinações'] #MX = combinations [1]
    MY = dados_fundacao[2]['combinções']  
    Fz = dados_fundacao[0]['combinações']

    #coeficientes
    sigma_cp = 0 # tensão a mais devido a efeitos da protensão
    ro = 0.01 # taxa geométrica de armadura de flexão aderente <0,02 (NBR6118-2023)
    ke = 1 + math.sqrt(20 / (d * 100))
    kx = interpolar_kx(none_variable)
    ky = interpolar_ky(none_variable)

    # calculando Fcd
    fcd = fck / 1.4

    # Verificando se C' está dentro da sapata!!!!!
    

    # tensão solicitante em C
    def calcular_TalSd2(FZ, a, b, d):
        return 0.001 * FZ / (2 * a + b * d) 
    
    # Tensão resistente em C
    def calcular_TalRd2(fck, fcd):
        return 0.27 * (1 - fck / 250) * fcd
    
    # Tensão solicitante em C'
    def calcular_TalSd1(Kx, Ky, MX, MY, A, B, a, b, d, FZ):
        Wpx = Kx * MX / (b**2 / 2 + b * a + 4 * a * d + 16 * d**2 + 2 * math.pi * b * d)
        Wpy = Ky * MY / (a**2 / 2 + a * b + 4 * b * d + 16 * d**2 + 2 * math.pi * a * d)
        u = 2 * (A + B + a + b + 2 * math.pi * d)
        return (0.001 * FZ) / (u * d) + Kx * MX * 0.001 / (Wpx * d) + Ky * MY * 0.001 / (Wpy * d) # x0.001 para colocar o resultado em MPa
    
    # tensão resististente em C'
    def calcular_TalRd1(Ke, ro, fck, sigma_cp):
        return  0.13 * Ke * (100 * ro * fck) ** (1 / 3) + 0.1 * sigma_cp

    TalRd1 = calcular_TalRd1(ke, ro, fck, sigma_cp)
    TalRd2 = calcular_TalRd2(fck, fcd)
    TalSd1 = calcular_TalSd1(kx, ky, MX, MY, A, B, a, b, d, Fz)
    TalSd2 = calcular_TalSd2(Fz, a, b, d)
    
    #Verificações
    # Verificação em C'  
    if 2 * d <= (A - d) / 2 and 2 * d <= (B - b)/ 2:
        if ro <= 0.02:
            if ke <= 2:
                if TalSd1 / TalRd1 - 1 > 0:
                    g1 = 1
                else:
                    g1 = 0
            else:
                g1 = 1
        else:
            g1 = 1
    else:
        g1 = 0

    #Verificações
    # Verificação em C 
    if TalSd2/ TalRd2 - 1 > 0 :
        g2 = 1
    else:
        g2 = 0
    
    print(g1, g2)
    return max(g1, g2)

def obj_ic_fundacoes(x, none_variable):
    # Organização variáveis de projeto e variáveis do problema de engenharia
    h_x = x[0]
    h_y = x[1]
    h_z = none_variable['h_z (m)']
    df = none_variable['dados estrutura']
    vol = 0 
    g = []
    for _ in range(len(df)):
        aux = volume_fundacao(h_x, h_y, h_z)
        vol += aux
    
    # Verificação geometria balanço dos pilares
    for _, row in df.iterrows():
        a_p = row['ap (m)']
        b_p = row['bp (m)']
        g_0, g_1, g_2, g_3 = restricao_geometrica_balanco_pilar_sapata(h_x, h_y, h_z, a_p, b_p)
        g.append(g_0)
        g.append(g_1)
        g.append(g_2)
        g.append(g_3)
        # g_0, g_1, g_2, g_3 = restricao_geometrica_balanco_pilar_sapata(h_x, h_y, h_z, a_p, b_p)
    print(g)
    
    # Função pseudo-objetivo
    of = vol
    for i in g:
        of += max(0, i) * 1E6

    # g_c = []
    
    #for index, row in df.iterrows():
        #f_z = row['Fz-c1']
       # m_x = row['Mx-c1']
        #m_y = row['My-c1']
        #g = calcular_sigma_max(f_z, m_x, m_y, h_x, h_y)
        #g_c1.append(g)
    # for index, row in df.iterrows():
    #     for i in range(1, 4): #alterar para 1, 11 quando for usar a tabela correta
    #         f_z = row[f'Fz-c{i}']
    #         m_x = row[f'Mx-c{i}']
    #         m_y = row[f'My-c{i}']
    #         g = calcular_sigma_max (f_z, m_x, m_y, h_x, h_y,)
    #         g_c.append(g)
    
    # # Determina o volume do elemento de fundação
    

    # # Trazendo as Restrições
    # g1 = restricao_tensao(none_variable, h_x, h_y, calcular_sigma_max )
    # g2 = restricao_geometrica(A, B, a, b)
    # g3 = restricao_puncao(dados_fundacao, A, B, MX, MY, Fz, combinations, rec)

    # # Função objetivo e restrições
    # of = vol
    # of += max(0, g1) * 1E6
    # of += max(0, g2) * 1E6
    # of += max(0, g3) * 1E6
    
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
    print(df)
    a = 0.6
    b = 0.6
    x = [a, b]
    none_variable = {'dados estrutura': df, 'h_z (m)': 0.6, 'rec (m)': 0.025, 'fck (MPa)': 25 }
    ofi = obj_ic_fundacoes(x, none_variable)
    # #print("resultado ofi: ",ofi)
    # teste = restricao_geometrica(A, B, a, b)
    # print(teste)
