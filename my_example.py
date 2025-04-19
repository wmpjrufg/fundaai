"""Função objetivo para o problema de fundações"""
from itertools import combinations
import numpy as np
import pandas as pd
import math

#declarando apenas por estética
fck = []
dimensoes_sapata_a = []
dimensoes_sapata_b = []


def calcular_sigma_max(f_z: float, m_x: float, m_y: float, h_x: float, h_y: float) -> tuple[float, float]:
    """
    Precisa ser alterada!!!!! Corrigida para a equação que utiliza as forças horizontais, precisa?
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

#Restricao de tensão.... precisa ser corrigda as variaveis hx e hy
def restricao_tensao(x, none_variable):
    # Variáveis de projeto
    h_x = x[0]
    h_y = x[1]
    comb = none_variable['combinações']
    sigma_lim = none_variable['sigma_adm (kPa)']

    # Verificação da restrição
    g = []
    for key, values in comb.items():
        f_z = values[0]
        m_x = values[1]
        m_y = values[2]
        sigma_sd_max, sigma_sd_min = calcular_sigma_max(f_z, m_x, m_y, h_x, h_y)  # Retorna um tuple
        g.append(sigma_sd_max / sigma_lim - 1)  # Usa o valor máximo como o mais crítico

    # Função objetivo e restrições
        
    for i in g:
        of += max(0, i)

    return of

#Restiricao geometrica - Balanço
def restricao_geometrica(A, B, a, b):

    # Definir as dimensões da sapata
    A = dimensoes_sapata_a # dimensão hx(m)
    B = dimensoes_sapata_b # dimensão hy(m)
    
    # Buscar as dimensões dos pilares
    a = dados_fundacao ['ap']
    b = dados_fundacao ['bp']

    # Para calcular o balanço na direção X
    Ca = (A - a)/2

    # Para calcular o balanço na direção Y
    Cb = (B - b)/2

    # Verificaçao todas as linhas satisfazem as restrições
    if ((Ca/(60 - a)/2 - 1 >= 0) & (Cb/(60 - b)/2 - 1 >= 0)):
        return 0  # Restrição satisfeita
    else:
        return 1 # Restrição não satisfeita, adiciona penalidade
    
#Definição kx e ky para restrição de punção de acordo com NBR ....
## para kx
def interpolar_kx(a, b,):
    
    a =ap
    b = bp
    b_a = a / b
    # Tabela de valores conhecidos
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
def interpolar_ky(a, b):

    a = dados_fundacao ['ap']
    b = dados_fundacao ['bp']
    a_b = a / b

    # Tabela de valores conhecidos
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
def restricao_puncao(a, b, A, B, MX, MY, Fz):
    #variáveis
    a = dados_fundacao ['ap']
    b = dados_fundacao ['bp']
    A= dimensoes_sapata_a # preciso chamar as dimensões da sapata que serão utilizadas em cada interação!!
    B= dimensoes_sapata_b # preciso chamar as dimensões da sapata que serão utilizadas em cada interação!!
    d = 0.2 - 0.04 #Altura util da sapata, 0,04 é um valor qualquer que deve ser especificado , quando tiver formato inclinado, d pode assumir valores diferentes em C e C'
    MX = combinações ['Mx']
    MY = combinações ['My']  
    Fz = combinações ['Fz'] 

    #coeficientes
    sigma_cp = 0 # tensão a mais devido a efeitos da protensão
    ro = 0.01 # rô também vira restrição pois não pode ser maior que 2% !!!!!
    ke = 1 + math.sqrt(20 / (d * 100))
    kx = interpolar_kx(a, b)
    ky = interpolar_ky(a, b)

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
x = [1,1]

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
    
    h_x = dimensoes_sapata_a
    h_y = dimensoes_sapata_b
    # Determina o volume do elemento de fundação
    vol = volume_fundacao(h_x, h_y)

    # Trazendo as Restrições
    g1 = restricao_tensao(x, none_variable)
    g2 = restricao_geometrica(x, none_variable)
    g3 = restricao_puncao(x, none_variable)

    # Função objetivo e restrições
    of = vol
    of += max(0, g1) * 1E6
    of += max(0, g2) * 1E6
    of += max(0, g3) * 1E6

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
