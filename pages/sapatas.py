import streamlit as st
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from mealpy import GA
from io import BytesIO

from foundation import *
from metapy_toolbox import *


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
