"""Façade module kept for backwards compatibility.

This file used to host the entire engineering layer plus the GPR helpers
plus the project objective function. Sprint 3.2 of the refactor moved
the pure engineering checks to ``core.engineering``. They remain
importable from here through the re-exports below so existing
consumers (Streamlit pages, notebooks, the test suite) keep working
without modification.

Resumo em português:
    Camada de compatibilidade. As verificações analíticas puras vivem
    agora em ``core.engineering``; este módulo apenas as reexporta e
    mantém o restante do código (FO, kernels GPR, treino paralelo).
"""

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

# --- Re-exports from core.engineering (Sprint 3.2) ---------------------------
# Engineering checks moved to core/engineering/. Imported here so that
# `from fundacao import tensao_adm_solo` and similar legacy imports keep
# working unchanged.
from core.engineering import (  # noqa: F401  (re-exported on purpose)
    tensao_adm_solo,
    calcular_sigma_max_min,
    checagem_tensao_max_min,
    checagem_geometria,
    verificacao_puncao_sapata,
    sobreposicao_sapatas,
    sobreposicao_matrix,
)


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




_PENALTY_DEFAULT = 1e1
"""Fator de penalidade padrão para `_avaliar_projeto`.

Reproduz exatamente o valor `1E1` que estava hardcoded nas duas funções
originais (`obj_felipe_lucas` e `obj_teste`). Manter este valor como default
preserva o comportamento histórico ao mesmo tempo em que permite
parametrização via `args[4]` ou via argumento `penalty` direto.
"""


def _unpack_args(args):
    """This helper extracts the configuration tuple consumed by `_avaliar_projeto`.

    Aceita tanto 4 quanto 5 elementos para retrocompatibilidade com os
    notebooks que sempre passaram um quinto valor de penalidade
    silenciosamente ignorado pela versão antiga. Quando o quinto elemento
    está ausente, aplica `_PENALTY_DEFAULT`.

    :param args: Tupla `(df, n_comb, f_ck, cob_m)` ou `(df, n_comb, f_ck, cob_m, penalty)`

    :return: [0] = DataFrame de entrada das fundações [df]
             [1] = Número de combinações de carregamento [n_comb]
             [2] = Resistência característica do concreto em kPa [f_ck]
             [3] = Cobrimento da armadura em metros [cob_m]
             [4] = Fator de penalidade aplicado às restrições violadas [penalty]
    """
    df, n_comb, f_ck, cob_m = args[0], args[1], args[2], args[3]
    penalty = args[4] if len(args) >= 5 else _PENALTY_DEFAULT
    return df, n_comb, f_ck, cob_m, penalty


def _avaliar_projeto(x, args, *, penalty=None):
    """This function evaluates the penalised pseudo-objective for a candidate solution `x`.

    Núcleo computacional compartilhado por `obj_felipe_lucas` e por
    `obj_teste`. A separação elimina a duplicação histórica entre as duas
    funções e centraliza o cálculo do volume bruto, das restrições
    normativas (NBR 6118 e NBR 6122) e da penalização exterior.

    :param x: Vetor com `3 * N_fun` variáveis de projeto, organizado como
              `[hx_0, hy_0, hz_0, ..., hx_{N-1}, hy_{N-1}, hz_{N-1}]`
    :param args: Tupla `(df, n_comb, f_ck, cob_m[, penalty])` (ver `_unpack_args`)
    :param penalty: Fator de penalidade explícito. Se `None`, usa o valor
                    presente em `args[4]` ou `_PENALTY_DEFAULT`. Permite
                    override por chamadores que não usam a tupla `args`

    :return: [0] = Valor escalar do volume final penalizado [of_total]
             [1] = DataFrame anotado com volume, vértices, sobreposição,
                   tensões, restrições e razão solicitação/resistência
                   por combinação de carregamento [df_anotado]
    """
    df, n_comb, f_ck, cob_m, penalty_args = _unpack_args(args)
    penalty = penalty_args if penalty is None else penalty

    df = df.copy()
    n_fun = df.shape[0]

    # Correção de formato
    df['spt'] = df['spt'].astype(float)

    # Variáveis de projeto e volume bruto
    x_arr = np.asarray(x).reshape(n_fun, 3)
    df[['h_x (m)', 'h_y (m)', 'h_z (m)']] = pd.DataFrame(
        x_arr, columns=['h_x (m)', 'h_y (m)', 'h_z (m)']
    )
    df['volume (m3)'] = df['h_x (m)'] * df['h_y (m)'] * df['h_z (m)']

    # Vértices da sapata em planta
    df['x1'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y1'] = df['yg (m)'] - df['h_y (m)'] / 2
    df['x2'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y2'] = df['yg (m)'] - df['h_y (m)'] / 2
    df['x3'] = df['xg (m)'] + df['h_x (m)'] / 2
    df['y3'] = df['yg (m)'] + df['h_y (m)'] / 2
    df['x4'] = df['xg (m)'] - df['h_x (m)'] / 2
    df['y4'] = df['yg (m)'] + df['h_y (m)'] / 2

    # Sobreposição entre sapatas (g_sob por sapata, soma sobre vizinhas).
    # Vetorizado em Sprint 3.8: matriz N×N em numpy substitui o laço
    # duplo `df.iterrows()`. Como os retângulos são axis-aligned, os
    # bounds AABB coincidem com x1, x2, y1 e y3.
    if n_fun == 1:
        df['g sobreposicao'] = 0.0
    else:
        xmin = df['x1'].to_numpy(dtype=np.float64)
        xmax = df['x2'].to_numpy(dtype=np.float64)
        ymin = df['y1'].to_numpy(dtype=np.float64)
        ymax = df['y3'].to_numpy(dtype=np.float64)
        overlap = sobreposicao_matrix(xmin, xmax, ymin, ymax)
        h_x_arr = df['h_x (m)'].to_numpy(dtype=np.float64)
        h_y_arr = df['h_y (m)'].to_numpy(dtype=np.float64)
        df['g sobreposicao'] = overlap.sum(axis=1) / (h_x_arr * h_y_arr)

    # Tensão admissível do solo
    df['tensao adm. (kPa)'] = df.apply(
        lambda row: tensao_adm_solo(row['solo'], row['spt']), axis=1
    )

    # Rótulos das combinações de carregamento
    labels_comb = [f'c{i}' for i in range(1, n_comb + 1)]

    # Checagem à punção (seção crítica C) por combinação
    for i in labels_comb:
        df[[f'tau_sd2 - {i}', f'tau_rd2 - {i}',
            f'u_rd2 - {i}', f'g_rd2 - {i}']] = df.apply(
            lambda row, k=i: verificacao_puncao_sapata(
                row['h_z (m)'], f_ck, row['ap (m)'], row['bp (m)'],
                row[f'Fz-{k}'], cob=cob_m
            ),
            axis=1, result_type='expand',
        )
    df['g punção secao C'] = df[[f'g_rd2 - {i}' for i in labels_comb]].max(axis=1)

    # Checagem das tensões máxima e mínima por combinação
    for i in labels_comb:
        df[[f'tensao max. (kPa) - {i}',
            f'tensao min. (kPa) - {i}']] = df.apply(
            lambda row, k=i: calcular_sigma_max_min(
                row[f'Fz-{k}'], row[f'Mx-{k}'], row[f'My-{k}'],
                row['h_x (m)'], row['h_y (m)']
            ),
            axis=1, result_type='expand',
        )
        df[f'g tensao max. - {i}'] = df.apply(
            lambda row, k=i: checagem_tensao_max_min(
                row[f'tensao max. (kPa) - {k}'], row['tensao adm. (kPa)']
            ),
            axis=1,
        )
        df[f'g tensao min. - {i}'] = df.apply(
            lambda row, k=i: checagem_tensao_max_min(
                row[f'tensao min. (kPa) - {k}'], row['tensao adm. (kPa)']
            ),
            axis=1,
        )
        df[f'g tensao - {i}'] = df[[f'g tensao max. - {i}',
                                    f'g tensao min. - {i}']].max(axis=1)
    df['g tensao'] = df[[f'g tensao - {i}' for i in labels_comb]].max(axis=1)

    # Checagem geométrica (balanço mínimo pilar-sapata)
    df['g geometria x'] = df.apply(
        lambda row: checagem_geometria(row['h_x (m)'], row['ap (m)']), axis=1
    )
    df['g geometria y'] = df.apply(
        lambda row: checagem_geometria(row['h_y (m)'], row['bp (m)']), axis=1
    )
    df['g geometria'] = df[['g geometria x', 'g geometria y']].max(axis=1)

    # Função pseudo-objetivo: volume + penalização exterior linear
    df['volume final (m3)'] = (
        df['volume (m3)']
        + df['g sobreposicao'].clip(lower=0) * penalty
        + df['g punção secao C'].clip(lower=0) * penalty
        + df['g tensao'].clip(lower=0) * penalty
        + df['g geometria'].clip(lower=0) * penalty
    )
    of_total = df['volume final (m3)'].sum()
    return of_total, df


def obj_felipe_lucas(x, args):
    """This function returns the scalar pseudo-objective used by the optimisation loop.

    Wrapper fino sobre `_avaliar_projeto`: descarta o DataFrame anotado e
    devolve apenas o valor de volume final penalizado. Mantém o
    comportamento histórico (penalty = 10) quando o quinto valor de
    `args` não é fornecido.

    :param x: Vetor com `3 * N_fun` variáveis de projeto
              `[hx_0, hy_0, hz_0, ..., hx_{N-1}, hy_{N-1}, hz_{N-1}]`
    :param args: Tupla `(df, n_comb, f_ck, cob_m)` ou
                 `(df, n_comb, f_ck, cob_m, penalty)`

    :return: Valor escalar do volume final penalizado [of_total]
    """
    of_total, _ = _avaliar_projeto(x, args)
    return of_total


def obj_teste(x, args):
    """This function evaluates the pseudo-objective and returns the annotated DataFrame.

    Wrapper fino sobre `_avaliar_projeto`. Útil em notebooks e na rotina
    de pós-processamento da UI, onde além do escalar são necessários os
    valores das restrições e das tensões para diagnóstico das soluções.

    :param x: Vetor com `3 * N_fun` variáveis de projeto
              `[hx_0, hy_0, hz_0, ..., hx_{N-1}, hy_{N-1}, hz_{N-1}]`
    :param args: Tupla `(df, n_comb, f_ck, cob_m)` ou
                 `(df, n_comb, f_ck, cob_m, penalty)`

    :return: [0] = Valor escalar do volume final penalizado [of_total]
             [1] = DataFrame anotado com restrições, tensões e razão
                   solicitação/resistência [df_anotado]
    """
    return _avaliar_projeto(x, args)


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

