"""This script creates a Streamlit web application for dimensioning isolated footings based on user-provided Excel data."""
import streamlit as st
from pathlib import Path
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from mealpy import GA

from foundation import *
from metapy_toolbox import *

# Title and description
st.title('Dimensionamento de Sapatas')
st.write(r"""
            <p style="text-align: justify;">
            Este aplicativo tem como objetivo auxiliar no dimensionamento de sapatas isoladas,
            considerando a resistência do solo e as cargas aplicadas. Para isso, é necessário
            que o usuário forneça um arquivo Excel com os dados de entrada, conforme o exemplo
            disponível para download.
            </p>

            <h2>Observações:</h2>
            <ul>
            <li>O arquivo de entrada deve conter as seguintes colunas:
                <ul>
                <li>Elemento: Nome do elemento</li>
                <li>ap (m): dimensão x do pilar (m)</li>
                <li>bp (m): dimensão y do pilar (m)</li>
                <li>spt: spt</li>
                <li>solo: Tipo de solo</li>
                <li>xg (m): Coordenada x do pilar (m)</li>
                <li>yg (m): Coordenada y do pilar (m)</li>
                <li>Fz-ci (kN): Força vertical aplicada no pilar da combinação i (kN)</li>
                <li>Mx-ci (kNm): Momento fletor em torno do eixo x (kNm) da combinação i</li>
                <li>My-ci (kNm): Momento fletor em torno do eixo y (kNm) da combinação i</li>
                </ul>
            </li>

            <li>Não modifique o cabeçalho da planilha, pois o aplicativo faz referência a ele.</li>
            <li>Qualquer número de combinações pode ser informado na planilha, seguindo o padrão de nomenclatura.</li>

            <li>
                Os índices finais <strong>c1</strong>, <strong>c2</strong> e <strong>c3</strong> associados às ações
                (<em>Fz</em>, <em>Mx</em> e <em>My</em>) indicam a <strong>combinação de carregamento</strong> à qual cada
                valor pertence. Dessa forma, por exemplo, <em>Fz-c1</em>, <em>Mx-c1</em> e <em>My-c1</em> correspondem
                às ações da combinação 1, enquanto <em>Fz-c2</em>, <em>Mx-c2</em> e <em>My-c2</em> referem-se à combinação 2,
                e assim sucessivamente.
            </li>

            <li>Aplicação em construção, atualmente ela é capaz de analisar, para uma dada dimensão de sapata, se passa ou não em várias verificações.</li>
            </ul>

            <p>Você pode baixar um arquivo de exemplo clicando no botão abaixo.</p>
""", unsafe_allow_html=True)

# Design sheet download
template_path = Path("assets/template_5_fundacoes_3_combinacoes.xlsx")
if template_path.exists():
    with open(template_path, "rb") as file:
        st.download_button(
            label="📥 Baixar planilha de exemplo",
            data=file,
            file_name="template_dimensionamento_sapatas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.warning("Arquivo de template não encontrado no diretório do aplicativo.")
st.divider()

# Upload file
st.subheader("Upload da planilha de dados")
uploaded_file = st.file_uploader("Selecione o arquivo Excel", type=["xlsx"])
df = pd.read_excel(uploaded_file)
n_fun = df.shape[0]
st.subheader("Primeiras linhas da planilha")
st.dataframe(df.head())


# Optimization variables
st.subheader("Parâmetros gerais de dimensionamento")

col1, col2 = st.columns(2)

with col1:
    n_comb = st.number_input("Número de combinações informadas na planilha", value=3)
    max
    min

with col2:
    fck = st.number_input("fck do concreto (MPa)", value=25.0)
st.divider()

# =============================
# BOTÃO DE EXECUÇÃO
# =============================
if st.button("Dimensionar", type="primary"):

    if uploaded_file is None:
        st.warning("Por favor, faça o upload da planilha antes de executar.")
    else:
        try:
            st.info("Processando os dados...")
            x_ini = [
                        [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65],
                        [0.6, 0.8],
                        [0.8, 0.6],
                        [0.8, 0.8],
                        [1.0, 1.0],
                        [2.5, 2.5]
                    ]
            paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=40, pop_size=50)}
            paras_kernel = {'kernel': 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))}
            n_gen = 100
            x_new, best_of, df = ego_01_architecture(obj_felipe_lucas, n_gen, x_ini, [h_xmax]*2*n_fun, [h_ymin]*2*n_fun, paras_opt, paras_kernel, args=(df, n_comb))
            print(f"Best solution: {x_new} -> OF: {best_of}")
            # results = run_dimensionamento(df=df, sigma_adm=sigma_adm, gamma_c=gamma_c, cobrimento=cobrimento, fck=fck, coef_seg=coef_seg)
            # st.success("Processamento concluído com sucesso.")
            # st.dataframe(results)

        except Exception as e:
            st.error("Erro durante o processamento.")
            st.exception(e)

