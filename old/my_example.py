

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


# def obj_ic_fundacoes(x: list, none_variable: dict)-> float:
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


# if __name__== '__main__':
#     df = pd.read_excel("teste_wand.xlsx")
#     df = tensao_adm_solo(df)
#     a = 0.6
#     b = 0.6
#     x = [a, b]
#     none_variable = {'dados estrutura': df, 'h_z (m)': 0.6, 'cob (m)': 0.025, 'fck (kPa)': 25000, 'número de combinações estruturais': 3}
#     of = obj_ic_fundacoes(x, none_variable)
    
#     print(of)