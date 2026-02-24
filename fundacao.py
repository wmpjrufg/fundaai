"""Esse script contém as funções que verificam uma sapata e que são usadas na interface do projeto."""


import numpy as np
import joblib
import multiprocessing as mp
import re
import pandas as pd
from pathlib import Path
import streamlit as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ExpSineSquared, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Any


def download_template(path: str | Path, label: str, filename: str):
    """Disponibiliza um arquivo para download no Streamlit.

    :param path: Caminho do arquivo local.
    :param label: Texto do botão.
    :param filename: Nome do arquivo no download.
    """
    path = Path(path)

    if path.exists():
        with open(path, "rb") as file:
            st.download_button(
                label=label,
                data=file,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        # st.error(f"Arquivo não encontrado: {path}")
        st.write(f"arquivo indisponível 📄🚫")


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


def calcular_sigma_max_min(f_zk: float, m_xk: float, m_yk: float, h_x: float, h_y: float) -> tuple[float, float]:
    """Calcula as tensões máxima e mínima atuantes na sapata, considerando excentricidades nos dois eixos.

    :param f_zk: Carga axial característica [kN]
    :param m_xk: Momento em x característica [kN·m]
    :param m_yk: Momento em y característica [kN·m]
    :param h_x: Dimensão da sapata em x [m]
    :param h_y: Dimensão da sapata em y [m]

    :return: saida[0] = tensão máxima (kPa), saida[1] = tensão mínima (kPa)
    """
    
    m_xk = abs(m_xk)
    m_yk = abs(m_yk)
    sigma_fz = (f_zk / (h_x * h_y)) * 1.05 
    aux_mx = 6 * m_xk / (f_zk * h_x)
    aux_my = 6 * m_yk / (f_zk * h_y)
    sigma_max = (sigma_fz) * (1 + aux_mx + aux_my)
    if sigma_max <= 0:
        sigma_max *= 1.0
    else:
        sigma_max *= 1.30
    sigma_min = (sigma_fz) * (1 - aux_mx - aux_my)
    if sigma_min <= 0:
        sigma_min *= 1.0
    else:
        sigma_min *= 1.30
    
    return sigma_max, sigma_min


def checagem_tensao_max_min(sigma: float, sigma_adm: float) -> float:
    """Determina a restrição de projeto da tensão admissível do solo.

    :param sigma: Tensão atuante no solo [kPa]
    :param sigma_adm: Tensão máxima admissível do solo [kPa]

    :return: Restrição de projeto e g <= 0 para restrição ser satisfeita
    """

    if sigma >= 0: 
        g = sigma / sigma_adm - 1
    else:
        g = -sigma / sigma_adm

    return g


def checagem_geometria(dim_sapata: float, dim_pilar: float, balanco_min: float = 0.10) -> float:
    """Determina a restrição de projeto da geometria da sapata.

    :param dim_sapata: Dimensão da sapata [m]
    :param dim_pilar: Dimensão do pilar [m]
    :param balanco_min: Balanco mínimo permitido [m]. Padrão é 0.10 m

    :return: Restrição de projeto e g <= 0 para restrição ser satisfeita
    """
    # ap + 2delta-hx <=0 (o pilar tem que ser no mínimo menor igual a a sapata menos 2 vezes o balanço mínimo)

    delta_ap = 2*balanco_min/dim_pilar
    delta_hx = dim_sapata/dim_pilar
    g = 1 + delta_ap - delta_hx

    return g


def verificacao_puncao_sapata(h_z: float, f_ck: float, a_p: float, b_p: float, f_zk: float, cob: float = 0.025) -> tuple[float, float, float, float]:
    """Determina a tensão resistente e verifica segundo a NBR 6118.   
    
    :param h_z: Altura da sapata [m]
    :param f_ck: Resistência característica à compressão do concreto [kPa]
    :param a_p: Dimensão do pilar na direção x [m]
    :param b_p: Dimensão do pilar na direção y [m]
    :param f_zk: Carga axial característica [kN]
    :param m_xk: Momento em x característica [kN·m]
    :param m_yk: Momento em y característica [kN·m]
    :param sigma_cp: Tensão axial extra [kPa]. Padrão é 0.00 kPa
    :param cob: Cobrimento do concreto [m]. Padrão é 0.025 m   

    :return: [0] = tensão atuante face pilar [kPa], [1] = tensão resistente face pilar [kPa], [2] = perímetro crítico na face do pilar [m], [3] = verificação ao cisalhamento na face do pilar
    """

    # Verificação à punção na seção crítica C
    d = h_z - cob
    alpha_v2 = (1 - (f_ck/1000) / 250)
    f_cd = f_ck / 1.4
    tau_rd2 = 0.27 * alpha_v2 * f_cd
    u_rd2 = 2 * (d/2 + d/2)
    tau_sd2 = (1.4 * f_zk) / (u_rd2 * d)
    g_rd2 = tau_sd2 / tau_rd2 - 1

    # # Verificação à punção na seção crítica C'
    # secao_critica = h_z / 2
    # rho_x = rho_minimo_fck(f_ck)
    # rho_y = rho_minimo_fck(f_ck)
    # rho = np.sqrt(rho_x * rho_y)
    # rho = rho /100
    # k_e = min(1 + np.sqrt(20 / (d * 100)), 2.0)
    # g_ed =  k_e / 2 - 1 
    # tau_rd1 = 1000 * (0.13 * k_e * (100 * rho * (f_ck / 1000)) ** (1 / 3) + 0.1 * sigma_cp)
    # u_rd1 = 2 * (a_p + b_p) + 4 * np.pi * secao_critica
    # c_1 = a_p
    # c_2 = b_p
    # dd = secao_critica
    # kx = tabela_19_2(c_1 / c_2)
    # ky = tabela_19_2(c_2 / c_1)
    # w_px = c_1**2 / 2 + c_1 * c_2 + 4 * c_2 * dd + 16 * dd**2 + 2 * np.pi * c_1 * dd
    # w_py = c_2**2 / 2 + c_2 * c_1 + 4 * c_1 * dd + 16 * dd**2 + 2 * np.pi * c_2 * dd
    # tau_sd1 = (1.4 * f_zk) / (u_rd1 * d) + kx * (1.4 * m_xk) / (w_px * d) + ky * (1.4 * m_yk) / (w_py * d)
    # g_rd1 = tau_sd1 / tau_rd1 - 1

    return tau_sd2, tau_rd2, u_rd2, g_rd2


def sobreposicao_sapatas(x1_i: float, y1_i: float, x2_i: float, y2_i: float, x3_i: float, y3_i: float, x4_i: float, y4_i: float, x1_j: float, y1_j: float, x2_j: float, y2_j: float, x3_j: float, y3_j: float, x4_j: float, y4_j: float) -> float:
    """Determina a sobreposição entre dois retângulos definidos por suas coordenadas completas.
    
    :param x1_i: Coordenada x do vértice 1 do retângulo i [m]
    :param y1_i: Coordenada y do vértice 1 do retângulo i [m]
    :param x2_i: Coordenada x do vértice 2 do retângulo i [m]
    :param y2_i: Coordenada y do vértice 2 do retângulo i [m]
    :param x3_i: Coordenada x do vértice 3 do retângulo i [m]
    :param y3_i: Coordenada y do vértice 3 do retângulo i [m]
    :param x4_i: Coordenada x do vértice 4 do retângulo i [m]
    :param y4_i: Coordenada y do vértice 4 do retângulo i [m]
    :param x1_j: Coordenada x do vértice 1 do retângulo j [m]
    :param y1_j: Coordenada y do vértice 1 do retângulo j [m]
    :param x2_j: Coordenada x do vértice 2 do retângulo j [m]
    :param y2_j: Coordenada y do vértice 2 do retângulo j [m]
    :param x3_j: Coordenada x do vértice 3 do retângulo j [m]
    :param y3_j: Coordenada y do vértice 3 do retângulo j [m]
    :param x4_j: Coordenada x do vértice 4 do retângulo j [m]
    :param y4_j: Coordenada y do vértice 4 do retângulo j [m]

    :return: Área de sobreposição entre os dois retângulos [m²]
    """
    
    # Determinando min e max de todas as coordenadas x e y do retângulo i
    xi_min = min(x1_i, x2_i, x3_i, x4_i)
    xi_max = max(x1_i, x2_i, x3_i, x4_i)
    yi_min = min(y1_i, y2_i, y3_i, y4_i)
    yi_max = max(y1_i, y2_i, y3_i, y4_i)
    
    # Determinando min e max de todas as coordenadas x e y do retângulo j
    xj_min = min(x1_j, x2_j, x3_j, x4_j)
    xj_max = max(x1_j, x2_j, x3_j, x4_j)
    yj_min = min(y1_j, y2_j, y3_j, y4_j)
    yj_max = max(y1_j, y2_j, y3_j, y4_j)
    
    # Calcular sobreposição
    overlap_x = max(0, min(xi_max, xj_max) - max(xi_min, xj_min))
    overlap_y = max(0, min(yi_max, yj_max) - max(yi_min, yj_min))
    area = overlap_x * overlap_y
    
    return area


def checagem_maior_dimensao(maior_h, menor_h):
    return (maior_h / (menor_h * 3)) - 1

def validador_tensao():
    return

def obj_felipe_lucas(x, args):

    # Argumentos
    df = args[0].copy()
    n_comb = args[1]
    f_ck = args[2]
    cob_m = args[3]
    n_fun = df.shape[0]

    # Correção formato
    df['spt'] = df['spt'].astype(float)

    # Variáveis de projeto
    x_arr = np.asarray(x).reshape(n_fun, 3)
    df_aux_aux = pd.DataFrame(x_arr, columns=["h_x (m)", "h_y (m)", "h_z (m)"])
    df[['h_x (m)', 'h_y (m)', 'h_z (m)']] = df_aux_aux[['h_x (m)', 'h_y (m)', 'h_z (m)']]

    # Restrição tamanho maior em relação menor
    

    # Volume
    df['volume (m3)'] = df['h_x (m)'] * df['h_y (m)'] * df['h_z (m)']

    # Cálculo das coordenadas completas dos vértices das sapatas
    df['x1'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y1'] = df['yg (m)'] - df['h_y (m)'] / 2
    df['x2'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y2'] = df['yg (m)'] - df['h_y (m)'] / 2
    df['x3'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y3'] = df['yg (m)'] + df['h_y (m)'] / 2
    df['x4'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y4'] = df['yg (m)'] + df['h_y (m)'] / 2

    # Restrição de maior valor
    df['maior dimensão'] = df[['h_x (m)', 'h_y (m)']].max(axis=1)
    df['menor dimensão'] = df[['h_x (m)', 'h_y (m)']].min(axis=1)
    df['g maior dimensão'] = df.apply(lambda row: checagem_geometria(row['maior dimensão'], row['ap (m)']), axis=1)
    # Restrição de sobreposição
    if n_fun == 1:
        df['g sobreposicao'] = 0.0
    else:
        # Deteriminar sobreposição
        for idx, row in df.iterrows():
            aux = 0
            x1_i, y1_i = row['x1'], row['y1']
            x2_i, y2_i = row['x2'], row['y2']
            x3_i, y3_i = row['x3'], row['y3']
            x4_i, y4_i = row['x4'], row['y4']
            for jdx, row_j in df.iterrows():
                if jdx != idx:
                    x1_j, y1_j = row_j['x1'], row_j['y1']
                    x2_j, y2_j = row_j['x2'], row_j['y2']
                    x3_j, y3_j = row_j['x3'], row_j['y3']
                    x4_j, y4_j = row_j['x4'], row_j['y4']
                    area_overlap = sobreposicao_sapatas(x1_i, y1_i, x2_i, y2_i, x3_i, y3_i, x4_i, y4_i, x1_j, y1_j, x2_j, y2_j, x3_j, y3_j, x4_j, y4_j)
                    aux += area_overlap
            df.loc[idx, 'g sobreposicao'] = aux / (df.loc[idx, 'h_x (m)'] * df.loc[idx, 'h_y (m)'])

    # Tensão admissível do solo
    df['tensao adm. (kPa)'] = df.apply(lambda row: tensao_adm_solo(row['solo'], row['spt']), axis=1)

    # Validador de tensão
    ## 

    # Rótulo das combinações
    labels_comb = [f'c{i}' for i in range(1, n_comb + 1)]

    # Checagem punção
    for i in labels_comb:
        aux = f'{i}'
        df[[f'tau_sd2 - {aux}', f'tau_rd2 - {aux}', f'u_rd2 - {aux}', f'g_rd2 - {aux}']] = df.apply(lambda row: verificacao_puncao_sapata(row['h_z (m)'], f_ck, row['ap (m)'], row['bp (m)'], row[f'Fz-{aux}'], cob=cob_m), axis=1, result_type='expand')
    df['g punção secao C'] = df[[f'g_rd2 - {i}' for i in labels_comb]].max(axis=1)
    # df['g escala punção'] = df[[f'g_ed - {i}' for i in labels_comb]].max(axis=1)
    # df['g punção secao Clinha'] = df[[f'g_rd1 - {i}' for i in labels_comb]].max(axis=1)

    # Checagem tensao max e min
    for i in labels_comb:
        aux = f'{i}'
        df[[f'tensao max. (kPa) - {aux}', f'tensao min. (kPa) - {aux}']] = df.apply(lambda row: calcular_sigma_max_min(row[f'Fz-{aux}'], row[f'Mx-{aux}'], row[f'My-{aux}'], row['h_x (m)'], row['h_y (m)']), axis=1, result_type='expand')
        df[f'g tensao max. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao max. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao min. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao min. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao - {aux}'] = df[[f'g tensao max. - {aux}', f'g tensao min. - {aux}']].max(axis=1)
    df['g tensao'] = df[[f'g tensao - {i}' for i in labels_comb]].max(axis=1)
    
    # Checagem geometria
    df['g geometria x'] = df.apply(lambda row: checagem_geometria(row['h_x (m)'], row['ap (m)']), axis=1)
    df['g geometria y'] = df.apply(lambda row: checagem_geometria(row['h_y (m)'], row['bp (m)']), axis=1)
    df['g geometria'] = df[['g geometria x', 'g geometria y']].max(axis=1)
    
    # Volume final com penalizações
    df['volume final (m3)'] = df['volume (m3)'] + df['g sobreposicao'].clip(lower=0) * 1E1 + df['g punção secao C'].clip(lower=0) * 1E1 + df['g tensao'].clip(lower=0) * 1E1 + df['g geometria'].clip(lower=0) * 1E1
    of = df['volume final (m3)'].sum()

    return of


def obj_teste(x, args):

    # Argumentos
    df = args[0].copy()
    n_comb = args[1]
    f_ck = args[2]
    cob_m = args[3]
    n_fun = df.shape[0]

    # Correção formato
    df['spt'] = df['spt'].astype(float)

    # Variáveis de projeto
    x_arr = np.asarray(x).reshape(n_fun, 3)
    df_aux_aux = pd.DataFrame(x_arr, columns=["h_x (m)", "h_y (m)", "h_z (m)"])
    df[['h_x (m)', 'h_y (m)', 'h_z (m)']] = df_aux_aux[['h_x (m)', 'h_y (m)', 'h_z (m)']]

    # Volume
    df['volume (m3)'] = df['h_x (m)'] * df['h_y (m)'] * df['h_z (m)']

    # Cálculo das coordenadas completas dos vértices das sapatas
    df['x1'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y1'] = df['yg (m)'] - df['h_y (m)'] / 2
    df['x2'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y2'] = df['yg (m)'] - df['h_y (m)'] / 2
    df['x3'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y3'] = df['yg (m)'] + df['h_y (m)'] / 2
    df['x4'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y4'] = df['yg (m)'] + df['h_y (m)'] / 2

    if n_fun == 1:
        df['g sobreposicao'] = 0.0
    else:
        # Deteriminar sobreposição
        for idx, row in df.iterrows():
            aux = 0
            x1_i, y1_i = row['x1'], row['y1']
            x2_i, y2_i = row['x2'], row['y2']
            x3_i, y3_i = row['x3'], row['y3']
            x4_i, y4_i = row['x4'], row['y4']
            for jdx, row_j in df.iterrows():
                if jdx != idx:
                    x1_j, y1_j = row_j['x1'], row_j['y1']
                    x2_j, y2_j = row_j['x2'], row_j['y2']
                    x3_j, y3_j = row_j['x3'], row_j['y3']
                    x4_j, y4_j = row_j['x4'], row_j['y4']
                    area_overlap = sobreposicao_sapatas(x1_i, y1_i, x2_i, y2_i, x3_i, y3_i, x4_i, y4_i, x1_j, y1_j, x2_j, y2_j, x3_j, y3_j, x4_j, y4_j)
                    aux += area_overlap
            df.loc[idx, 'g sobreposicao'] = aux / (df.loc[idx, 'h_x (m)'] * df.loc[idx, 'h_y (m)'])

    # Tensão admissível do solo
    df['tensao adm. (kPa)'] = df.apply(lambda row: tensao_adm_solo(row['solo'], row['spt']), axis=1)

    # Rótulo das combinações
    labels_comb = [f'c{i}' for i in range(1, n_comb + 1)]

    # Checagem punção
    for i in labels_comb:
        aux = f'{i}'
        df[[f'tau_sd2 - {aux}', f'tau_rd2 - {aux}', f'u_rd2 - {aux}', f'g_rd2 - {aux}']] = df.apply(lambda row: verificacao_puncao_sapata(row['h_z (m)'], f_ck, row['ap (m)'], row['bp (m)'], row[f'Fz-{aux}'], cob=cob_m), axis=1, result_type='expand')
    df['g punção secao C'] = df[[f'g_rd2 - {i}' for i in labels_comb]].max(axis=1)
    # df['g escala punção'] = df[[f'g_ed - {i}' for i in labels_comb]].max(axis=1)
    # df['g punção secao Clinha'] = df[[f'g_rd1 - {i}' for i in labels_comb]].max(axis=1)

    # Checagem tensao max e min
    for i in labels_comb:
        aux = f'{i}'
        df[[f'tensao max. (kPa) - {aux}', f'tensao min. (kPa) - {aux}']] = df.apply(lambda row: calcular_sigma_max_min(row[f'Fz-{aux}'], row[f'Mx-{aux}'], row[f'My-{aux}'], row['h_x (m)'], row['h_y (m)']), axis=1, result_type='expand')
        df[f'g tensao max. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao max. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao min. - {aux}'] = df.apply(lambda row: checagem_tensao_max_min(row[f'tensao min. (kPa) - {aux}'], row['tensao adm. (kPa)']), axis=1)
        df[f'g tensao - {aux}'] = df[[f'g tensao max. - {aux}', f'g tensao min. - {aux}']].max(axis=1)
    df['g tensao'] = df[[f'g tensao - {i}' for i in labels_comb]].max(axis=1)
    
    # Checagem geometria
    df['g geometria x'] = df.apply(lambda row: checagem_geometria(row['h_x (m)'], row['ap (m)']), axis=1)
    df['g geometria y'] = df.apply(lambda row: checagem_geometria(row['h_y (m)'], row['bp (m)']), axis=1)
    df['g geometria'] = df[['g geometria x', 'g geometria y']].max(axis=1)
    
    # Volume final com penalizações
    df['volume final (m3)'] = df['volume (m3)'] + df['g sobreposicao'].clip(lower=0) * 1E1 + df['g punção secao C'].clip(lower=0) * 1E1 + df['g tensao'].clip(lower=0) * 1E1 + df['g geometria'].clip(lower=0) * 1E1
    of = df['volume final (m3)'].sum()

    return of, df


def constroi_kernel(ls0: float = 1.0) -> list:
    """Constroi uma lista de kernels para GPR (Gaussian Process Regressor).
    
    :param ls0: comprimento de escala inicial para os kernels

    :return: kernels
    """

    # Observação: bounds assumem X padronizado (StandardScaler)
    A = C(1.0, (1E-5, 1E10))  # amplitude

    k = []

    # 1–3: RBF variants
    k += [
            A * RBF(length_scale=ls0, length_scale_bounds=(1e-2, 1e2)),
            A * (RBF(ls0, (1e-2, 1e2)) + RBF(ls0*0.3, (1e-2, 1e2))),           # soma multi-escala
            A * (RBF(ls0, (1e-2, 1e2)) * RBF(ls0*0.5, (1e-2, 1e2))),           # produto (mais “sharp”)
        ]

    # 4–7: Matern (diferentes suavidades)
    k += [
            A * Matern(length_scale=ls0, length_scale_bounds=(1e-2, 1e2), nu=0.5),   # Exponential (menos suave)
            A * Matern(length_scale=ls0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            A * Matern(length_scale=ls0, length_scale_bounds=(1e-2, 1e2), nu=2.5),
            A * (Matern(ls0, (1e-2, 1e2), nu=1.5) + Matern(ls0*0.3, (1e-2, 1e2), nu=2.5)),  # multi-escala
        ]

    # 8–10: RationalQuadratic (mix contínuo de escalas)
    k += [
            A * RationalQuadratic(length_scale=ls0, alpha=1.0),
            A * RationalQuadratic(length_scale=ls0, alpha=0.1),
            A * RationalQuadratic(length_scale=ls0, alpha=10.0),
        ]

    # 11–14: Tendência linear + variação suave
    k += [
            A * (DotProduct(sigma_0=1.0) + RBF(ls0, (1e-2, 1e2))),              # linear + smooth
            A * (DotProduct(sigma_0=1.0) + Matern(ls0, (1e-2, 1e2), nu=1.5)),
            A * (DotProduct(sigma_0=0.1) + RBF(ls0, (1e-2, 1e2))),
            A * DotProduct(sigma_0=1.0),                                        # puramente linear
        ]

    # 15–17: Periodicidade (se fizer sentido no seu fenômeno)
    k += [
            A * ExpSineSquared(length_scale=ls0, periodicity=1.0, periodicity_bounds=(1e-2, 1e2)),
            A * (RBF(ls0, (1e-2, 1e2)) * ExpSineSquared(ls0, periodicity=1.0, periodicity_bounds=(1e-2, 1e2))), # quase-periódico
            A * (Matern(ls0, (1e-2, 1e2), nu=1.5) * ExpSineSquared(ls0, periodicity=1.0, periodicity_bounds=(1e-2, 1e2))),
        ]

    # 18–20: “quase-determinístico” com jitter mínimo embutido (opcional)
    # Se você quiser blindar contra problemas numéricos SEM assumir ruído físico:
    tiny = WhiteKernel(noise_level=1e-12, noise_level_bounds=(1e-15, 1e-9))
    k += [
            A * RBF(ls0, (1e-2, 1e2)) + tiny,
            A * Matern(ls0, (1e-2, 1e2), nu=2.5) + tiny,
            A * RationalQuadratic(ls0, alpha=1.0) + tiny,
            A * Matern(length_scale=ls0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
        ]

    return k


def gpr_pipelines(
                    ls0: float = 1.0,
                    alpha: float = 1e-4,
                    n_restarts: int = 5,
                    random_state: int = 42
                ) -> tuple[list, list]:
    """Monta os modelos de GPR (Gaussian Process Regressor).
    
    :param ls0: comprimento de escala inicial para os kernels
    :param alpha: jitter numérico (determinístico)
    :param n_restarts: número de reinicializações do otimizador
    :param random_state: semente para reprodutibilidade

    :return: [0] modelos instanciados e [1] seus nomes
    """

    kernels = constroi_kernel(ls0=ls0)
    modelos = []
    nomes = []

    for idx, ker in enumerate(kernels):
        sca = ("scaler", StandardScaler())
        gp = ("gp", GaussianProcessRegressor(kernel=ker, normalize_y=True, alpha=alpha, n_restarts_optimizer=n_restarts, random_state=random_state))
        pipe = Pipeline([sca, gp])                  
        modelos.append(pipe)
        nomes.append(f"gpr_com_kernel_k{idx:02d}")

    return modelos, nomes


def aprendizado_maquina_paralelo(
                                    x_treino: pd.DataFrame,
                                    y_treino: pd.DataFrame,
                                    x_teste: pd.DataFrame,
                                    y_teste: pd.DataFrame,
                                    n_jobs: int = mp.cpu_count(),
                                    ls0: float = 1.0,
                                    alpha: float = 0.1,
                                    n_restarts: int = 5,
                                    random_state: int = 42,
                                    out_dir: str = "modelos"
                                ) -> list:
    """Treina e testa modelos de aprendizado de máquina em paralelo.

    :param x_treino: dados de treino (features)
    :param y_treino: dados de treino (target)
    :param x_teste: dados de teste (features)
    :param y_teste: dados de teste (target)
    :param n_jobs: número de processos paralelos
    :param ls0: comprimento de escala inicial para os kernels
    :param alpha: jitter numérico (determinístico)
    :param n_restarts: número de reinicializações do otimizador
    :param random_state: semente para reprodutibilidade
    :param out_dir: diretório para salvar os modelos treinados

    :return: lista de dicionários com métricas e informações dos modelos treinados em paralelo
    """
    
    modelos, nomes = gpr_pipelines(ls0=ls0, alpha=alpha, n_restarts=n_restarts, random_state=random_state)
    args = [(nomes[i], modelos[i], x_treino, 
                y_treino, x_teste, y_teste, Path(out_dir)) for i in range(len(nomes))]
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.starmap(treino_teste_para_processo_paralelo, args)

    return results


def treino_teste_para_processo_paralelo(
                                            nome: str,
                                            modelo: Any, 
                                            x_treino: pd.DataFrame,
                                            y_treino: pd.DataFrame,
                                            x_teste: pd.DataFrame,
                                            y_teste: pd.DataFrame,
                                            dir_modelos: Path = Path("modelos")
                                        ) -> dict:
    """Treina e testa um modelo de aprendizado de máquina.

    :param nome: nome do modelo
    :param modelo: modelo de aprendizado de máquina
    :param x_treino: dados de treino (features)
    :param y_treino: dados de treino (target)
    :param x_teste: dados de teste (features)
    :param y_teste: dados de teste (target)
    :param dir_modelos: diretório para salvar os modelos treinados

    :return: dicionário com métricas e informações do modelo
    """
    dir_modelos.mkdir(parents=True, exist_ok=True)

    # Treino e salva modelo
    modelo.fit(x_treino, y_treino)
    nome_limpo = re.sub(r"[^a-zA-Z0-9_-]", "_", nome)
    nome_modelo = dir_modelos / f"{nome_limpo}_pop_{len(x_treino)}.pkl"
    joblib.dump(modelo, nome_modelo)

    # Testando para r2
    y_pred_treino = modelo.predict(x_treino)
    y_pred_teste  = modelo.predict(x_teste)
    y_pred_teste = pd.DataFrame(y_pred_teste, columns=["volume (m3)"])

    # Métricas
    r2_treino = r2_score(y_treino, y_pred_treino)
    r2_teste  = r2_score(y_teste,  y_pred_teste)
    mae       = mean_absolute_error(y_teste, y_pred_teste)
    rmse      = np.sqrt(mean_squared_error(y_teste, y_pred_teste))

    return {
                "modelo": nome,
                "arquivo": str(nome_modelo),
                "R2_Treino": r2_treino,
                "R2_Teste": r2_teste,
                "MAE": mae,
                "RMSE": rmse,
                "y_obse": y_teste,
                "y_pred": y_pred_teste
            }

