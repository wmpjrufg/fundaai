"""Aplicativo Streamlit para dimensionamento de sapatas isoladas."""
import streamlit as st
from pathlib import Path
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from mealpy import GA
from io import BytesIO

from foundation import *
from metapy_toolbox import *

# Título do aplicativo
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

# Planilha padrão
template_path = Path("assets/template_5_fundacoes_3_combinacoes_espalhadas.xlsx")
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

# Upload de planilha
st.subheader("Upload da planilha de dados")
uploaded_file = st.file_uploader("Selecione o arquivo Excel", type=["xlsx","xls"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("Arquivo carregado com sucesso!")
    n_fun = df.shape[0]
    st.subheader("Primeiras linhas da planilha de dados")
    st.dataframe(df.head())
else:
    st.warning("Por favor, selecione um arquivo Excel para continuar.")

# Otimização
st.subheader("Parâmetros gerais de dimensionamento")
col1, col2 = st.columns(2)
with col1:
    n_comb = st.number_input("Número de combinações informadas na planilha", step=1, value=3)
    h_xmin = st.number_input("Dimensão mínima da sapata (cm)", min_value=60., step=0.5, value=60.)
    h_xmax = st.number_input("Dimensão máxima da sapata (cm)", step=0.5, value=500.)
    n_gen = st.number_input("Número de gerações da otimização", min_value=5, max_value=50, step=5, value=10)
    n_pop = st.number_input("Tamanho da população", min_value=5, max_value=50, step=5, value=20)
    h_xmin /= 100
    h_xmax /= 100
with col2:
    f_ck = st.number_input("fck do concreto (MPa)", min_value=20., max_value=90., step=5.0, value=25.0)
    cob = st.number_input("Cobrimento do concreto (cm)", step=0.5, value=2.0, format="%.1f")
    h_z = st.number_input("Altura da sapata (cm)", min_value=60., step=0.5, value=60.)
    f_ck *= 1000
    cob /= 100 
    h_z /= 100
st.divider()

# Execução do dimensionamento otimizado
if st.button("Dimensionar", type="primary"):
    if uploaded_file is None:
        st.warning("Por favor, faça o upload da planilha antes de executar.")
    else:
        try:
            st.info("Processando os dados...")
            x_l = [h_xmin] * 2 * n_fun
            x_u = [h_xmax] * 2 * n_fun
            x_ini = initial_population_01(n_pop, 2 * n_fun, x_l, x_u, use_lhs=True)
            paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=40, pop_size=100)}
            paras_kernel = {'kernel': 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))}
            x_new, best_of, df = ego_01_architecture(obj_felipe_lucas, n_gen, x_ini, x_l, x_u, paras_opt, paras_kernel, args=(df, n_comb, f_ck, h_z))
            st.success("Dimensionamento concluído com sucesso!")
            st.subheader("📊 Resultados Detalhados")
            x_new = np.asarray(x_new).reshape(n_fun, 2)   
            dados_final = pd.DataFrame(x_new, columns=['h_x (m)', 'h_y (m)'])
            dados_final['h_z (m)'] = h_z
            st.dataframe(dados_final)
            st.metric(label="Função Objetivo (OF)", value=f"{best_of:.4f}")
            
            # Criar um buffer para o Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                dados_final.to_excel(writer, index=False, sheet_name='Dados')
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Baixar dados como Excel",
                data=excel_data,
                file_name="dados_da_sapata.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error("Erro durante o processamento.")
            st.exception(e)
