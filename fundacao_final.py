import pandas as pd
import numpy as np

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
    

def checagem


if __name__ == "__main__":
    df = pd.read_excel('teste_reduzido.xlsx')
    df['h_x (m)'] = 0.6
    df['h_y (m)'] = 0.6
    df['h_z (m)'] = 0.6
    n_comb = 3
    max_tensoes, min_tensoes, index_critico_max, index_critico_min, fz_critico_max, mx_critico_max, my_critico_max, fz_critico_min, mx_critico_min, my_critico_min = checagem_tensoes_dataframe(df, n_comb)
    print("Máximas Tensões:", max_tensoes)
    print("Mínimas Tensões:", min_tensoes)

    dnndndnd()