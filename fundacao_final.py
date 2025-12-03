import pandas as pd
import numpy as np
import shapely as sh

from my_example import *

def checagem_tensoes_dataframe(df: pd.DataFrame, n_comb: int):
    """
    Função para checar as tensões admissíveis no DataFrame.

    :param df: DataFrame contendo os dados das fundações e esforços.
    :param n_comb: Número de combinações de esforços.

    :return: [0] = Máxima tensão em cada fundação. [1] = Mínima tensão em cada fundação.
    """
    
    # Para calculo da tensão max e minima
    max_tensoes = []
    min_tensoes = []

    # Lista para armazenar os esforços da combinação crítica
    index_critico_max = []
    index_critico_min = []
    fz_critico_max = []; mx_critico_max = []; my_critico_max = []
    fz_critico_min = []; mx_critico_min = []; my_critico_min = []

    # Lista para armazenar os rótulos de combinação
    rotulos_comb = [f'c{i}' for i in range(1, n_comb + 1)]

    for idx, row in df.iterrows(): 
        t_max_aux = []
        t_min_aux = []

        # Checando para cada combinação
        for i in range(1, n_comb + 1):
            aux = f'c{i}'
            t_max_val, t_min_val = calcular_sigma_max(
                                                        row[f'Fz-{aux}'],
                                                        row[f'Mx-{aux}'],
                                                        row[f'My-{aux}'],
                                                        row['h_x (m)'],
                                                        row['h_y (m)'],
                                                    )
            t_max_aux.append(t_max_val)
            t_min_aux.append(t_min_val)
    
            
        # Armazenando os valores máximos e mínimos
        max_tensoes.append(max(t_max_aux))
        min_tensoes.append(min(t_min_aux))

        # Armazenando os índices das combinações críticas
        idx_max = t_max_aux.index(max(t_max_aux))
        idx_min = t_min_aux.index(min(t_min_aux))
        comb_max = rotulos_comb[idx_max]
        comb_min = rotulos_comb[idx_min]


        index_critico_max.append(comb_max)
        index_critico_min.append(comb_min)

        # Armazenando os esforços das combinações críticas
        fz_critico_max.append(row[f'Fz-{comb_max}']); mx_critico_max.append(row[f'Mx-{comb_max}']); my_critico_max.append(row[f'My-{comb_max}'])
        fz_critico_min.append(row[f'Fz-{comb_min}']); mx_critico_min.append(row[f'Mx-{comb_min}']); my_critico_min.append(row[f'My-{comb_min}'])

        return max_tensoes, min_tensoes, index_critico_max, index_critico_min, fz_critico_max, mx_critico_max, my_critico_max, fz_critico_min, mx_critico_min, my_critico_min
    
## Para restrição geométrica de balanço da sapata
# lista para armazenar resultados
g0_geo_pilar_sapata = []
g1_geo_pilar_sapata = []
g2_geo_pilar_sapata = []
g3_geo_pilar_sapata = []
cap = [] #valor do balanço em x
cbp = [] #valor do balanço em y ''

def checagem_balanço (df: pd.DataFrame):
  
    for idx, row in df.iterrows(): 
        cap_value, cbp_value, g0, g1, g2, g3 = restricao_geometrica_balanco_pilar_sapata(
                                                                                            h_x=row['h_x (m)'],
                                                                                            h_y=row['h_y (m)'],
                                                                                            h_z=row['h_z (m)'],
                                                                                            a_p=row['ap (m)'],
                                                                                            b_p=row['bp (m)']
                                                                                        )
        # Armazena cada valor em listas
        cap.append(cap_value)
        cbp.append(cbp_value)
        g0_geo_pilar_sapata.append(g0)
        g1_geo_pilar_sapata.append(g1)
        g2_geo_pilar_sapata.append(g2)
        g3_geo_pilar_sapata.append(g3)


    return cap, cbp, g0_geo_pilar_sapata, g1_geo_pilar_sapata, g2_geo_pilar_sapata, g3_geo_pilar_sapata

 ##Para restrição geometrica Pilar_Sapata
g0_geo_pilar = []
g1_geo_pilar = []

def checagem_pilar_sapata (df: pd.DataFrame):
    for idx, row in df.iterrows(): 
        g0, g1 = restricao_geometrica_pilar_sapata(
                                                    h_x=row['h_x (m)'],
                                                    h_y=row['h_y (m)'],
                                                    a_p=row['ap (m)'],
                                                    b_p=row['bp (m)']
                                                )
        # Armazena cada valor em listas
        g0_geo_pilar.append(g0)
        g1_geo_pilar.append(g1)

    return g0_geo_pilar, g1_geo_pilar


def checagem_sobreposição (df: pd.DataFrame):
     
    # Calculando os vertices A, B, C e D da sapata no df    
    df['x_a'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y_a'] = df['yg (m)'] + df['h_y (m)'] / 2
    df['x_b'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y_b'] = df['yg (m)'] - df['h_y (m)'] / 2
    df['x_c'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y_c'] = df['yg (m)'] + df['h_y (m)'] / 2
    df['x_d'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y_d'] = df['yg (m)'] - df['h_y (m)'] / 2
    coords_list = [f"{row['x_a']},{row['y_a']},{row['x_b']},{row['y_b']},{row['x_c']},{row['y_c']},{row['x_d']},{row['y_d']}" for _, row in df.iterrows()]
    df['coords'] = coords_list
    poligonos = []
    for coord_string in df['coords']:
        # Divide a string e converte para números
        numeros = [float(x) for x in coord_string.split(',')]
        # Cria os pontos do polígono: A -> B -> C -> D -> A (fechando)
        pontos = [
            (numeros[0], numeros[1]),  # Ponto A
            (numeros[2], numeros[3]),  # Ponto B  
            (numeros[4], numeros[5]),  # Ponto C
            (numeros[6], numeros[7]),  # Ponto D
            (numeros[0], numeros[1])   # Volta para A (fechar o polígono)
        ]
        poligono = sh.geometry.Polygon(pontos)
        poligonos.append(poligono)
    n = len(poligonos)
    contagem_intersecoes = [0] * n  # Lista para armazenar as contagens

    # Para cada polígono, verifica com todos os outros
    for i in range(n):
        for j in range(n):
            if i != j:  # Não comparar com ele mesmo
                if poligonos[i].intersects(poligonos[j]):
                    contagem_intersecoes[i] += 1
        

    return contagem_intersecoes

def checagem_puncao (df: pd.DataFrame):
    ## para restrição de punção
    # Atribuindo os parâmetros
    df['ro'] = 0.02
    df['cob (m)'] = 0.02
    df['fck (kPa)'] = 25000
    df['fcd (kPa)'] = 25000/1.4

    # lista para armazenar resultados
    ke_puncao = []
    kx_puncao = []
    ky_puncao = []
    wpx_puncao = []
    wpy_puncao = []
    u_puncao = []
    talsd1_puncao = []
    talrd1_puncao = []
    talsd2_puncao = []
    talrd2_puncao = []
    g0_puncao = []
    g1_puncao = []
    g2_puncao = []
    g3_puncao = []
    g4_puncao = []
    g5_puncao = []
    g6_puncao = []

    for idx, row in df.iterrows(): 
        ke, kx, ky, wpx, wpy, u, talsd1, talrd1, talsd2, talrd2, g0, g1, g2, g3, g4, g5, g6 = restricao_puncao(
                                                                                                                h_x=row['h_x (m)'],
                                                                                                                h_y=row['h_y (m)'],
                                                                                                                h_z=row['h_z (m)'],
                                                                                                                a_p=row['ap (m)'],
                                                                                                                b_p=row['bp (m)'],
                                                                                                                f_z= 1000, #verificar se é max ou min
                                                                                                                m_x= 30, #verificar se é max ou min
                                                                                                                m_y= 1, #verificar se é max ou min
                                                                                                                ro=row['ro'],
                                                                                                                cob=row['cob (m)'],
                                                                                                                fck=row['fck (kPa)'],
                                                                                                                fcd=row['fcd (kPa)']
                                                                                                            )
        # Armazena cada valor em listas
        ke_puncao.append(ke)
        kx_puncao.append(kx)
        ky_puncao.append(ky)
        wpx_puncao.append(wpx)
        wpy_puncao.append(wpy)
        u_puncao.append(u)  
        talsd1_puncao.append(talsd1)
        talrd1_puncao.append(talrd1)
        talsd2_puncao.append(talsd2)
        talrd2_puncao.append(talrd2)
        g0_puncao.append(g0)
        g1_puncao.append(g1)
        g2_puncao.append(g2)
        g3_puncao.append(g3)
        g4_puncao.append(g4)
        g5_puncao.append(g5)
        g6_puncao.append(g6)

    df['kx_puncao'] = kx_puncao
    df['ky_puncao'] = ky_puncao
    df['ke_puncao'] = ke_puncao
    df['wpx_puncao'] = wpx_puncao
    df['wpy_puncao'] = wpy_puncao
    df['u_puncao'] = u_puncao
    df['talsd1_puncao'] = talsd1_puncao
    df['talrd1_puncao'] = talrd1_puncao
    df['talsd2_puncao'] = talsd2_puncao
    df['talrd2_puncao'] = talrd2_puncao

    return df



if __name__ == "__main__":
    df = pd.read_excel('teste_reduzido.xlsx')
    df['h_x (m)'] = 0.6
    df['h_y (m)'] = 0.6
    df['h_z (m)'] = 0.6
    n_comb = 3
    # max_tensoes, min_tensoes, index_critico_max, index_critico_min, fz_critico_max, mx_critico_max, my_critico_max, fz_critico_min, mx_critico_min, my_critico_min = checagem_tensoes_dataframe(df, n_comb)
    # cap, cbp, g0_geo_pilar_sapata, g1_geo_pilar_sapata, g2_geo_pilar_sapata, g3_geo_pilar_sapata = checagem_balanço(df)
    # g0_geo_pilar, g1_geo_pilar = checagem_pilar_sapata(df)
    # print("Máximas Tensões:", max_tensoes)
    # print("Mínimas Tensões:", min_tensoes)
    # print("balaço x:", cap)
    # print("balaço y:", cbp)
    # print("g0:", g0_geo_pilar_sapata)
    print(checagem_puncao(df))

