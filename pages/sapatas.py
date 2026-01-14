"""Página de dimensionamento otimizado de sapatas utilizando Streamlit."""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from mealpy import GA
from io import BytesIO

from foundation import *
from metapy_toolbox import *

# --- INICIALIZAÇÃO DO ESTADO (SESSION STATE) ---
if 'calculo_realizado' not in st.session_state:
    st.session_state['calculo_realizado'] = False
if 'dados_final_df' not in st.session_state:
    st.session_state['dados_final_df'] = None
if 'best_of_valor' not in st.session_state:
    st.session_state['best_of_valor'] = None
if 'excel_bytes' not in st.session_state:
    st.session_state['excel_bytes'] = None


# Upload de planilha
st.subheader("Upload da planilha de dados")
uploaded_file = st.file_uploader("Selecione o arquivo Excel", type=["xlsx","xls"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # --- SANITIZAÇÃO DAS COLUNAS DE AÇÕES (OBRIGATÓRIO) ---
    for col in df.columns:
        if col.startswith(("Fz-", "Mx-", "My-")):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    st.success("Arquivo carregado com sucesso!")
    n_fun = df.shape[0]
    st.subheader("Primeiras linhas da planilha de dados")
    st.dataframe(df.head())
else:
    st.warning("Por favor, selecione um arquivo Excel para continuar.")
    st.session_state['calculo_realizado'] = False

# Otimização
st.subheader("Parâmetros gerais de dimensionamento")
# Materiais e geometria
n_comb = st.number_input("Número de combinações informadas na planilha", step=1, value=3, key="n_comb_input")
f_ck = st.number_input("fck do concreto (MPa)", min_value=20., max_value=90., step=5.0, value=25.0, key="f_ck_input")
cob = st.number_input("Cobrimento do concreto (cm)", step=0.5, value=2.0, format="%.1f", key="cob_input")
h_min = st.number_input("Dimensão mínima da sapata (cm)", min_value=60., step=0.5, value=60., key="h_xmin_input")
h_max = st.number_input("Dimensão máxima da sapata (cm)", min_value=60., step=0.5, value=150., key="h_xmax_input")
# Parâmetros de otimização
n_gen = st.number_input("Número de gerações da otimização", min_value=5, max_value=200, step=5, value=10, key="n_gen_input")
n_pop = st.number_input("Tamanho da população", min_value=5, max_value=2000, step=5, value=20, key="n_pop_input")
# Conversões
h_min_m = h_min / 100
h_max_m = h_max / 100
f_ck_kpa = f_ck * 1000
cob_m = cob / 100 

# Execução do dimensionamento otimizado
if st.button("Dimensionar", type="primary"):
    if uploaded_file is None:
        st.warning("Por favor, faça o upload da planilha antes de executar.")
        st.session_state['calculo_realizado'] = False
    else:
        try:
            with st.spinner("Processando os dados... Aguarde."):
               
                # Otimização
                x_l = [h_min_m] * 3 * n_fun
                x_u = [h_max_m] * 3 * n_fun
                st.info("Criação dos agentes...")
                x_ini = initial_population_01(n_pop, 3 * n_fun, x_l, x_u, use_lhs=True)
                # paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=60, pop_size=100)}
                # paras_opt = {'optimizer algorithm': PSO.AIW_PSO(epoch=60, pop_size=100, c1=2.05, c2=2.05, alpha=0.4)}
                paras_opt = {'optimizer algorithm': 'scipy_slsqp'}
                paras_kernel = {'kernel': 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))}  
                st.info("Iniciando otimização...")
                x_new, best_of, df_resultado_ego = ego_01_architecture(
                                                                        obj_felipe_lucas, 
                                                                        n_gen, 
                                                                        x_ini, 
                                                                        x_l, 
                                                                        x_u, 
                                                                        paras_opt, 
                                                                        paras_kernel, 
                                                                        args=(df, n_comb, f_ck_kpa, cob_m)
                                                                    )
                st.success("Dimensionamento concluído com sucesso!")
                
                # Processamento dos resultados
                x_arr = np.asarray(x_new).reshape(n_fun, 3)
                dados_final = pd.DataFrame(x_arr, columns=['h_x (m)', 'h_y (m)', 'h_z (m)'])
                _, df_novo = obj_teste(x_new, args=(df, n_comb, f_ck_kpa, cob_m))

                # Geração de planilhas
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    dados_final.to_excel(writer, index=False, sheet_name='Dados')
                excel_data = output.getvalue()
                output_extra = BytesIO()
                with pd.ExcelWriter(output_extra, engine='openpyxl') as writer_extra:
                    df_novo.to_excel(writer_extra, index=False, sheet_name='Dados Extras')
                excel_data_extra = output_extra.getvalue()

                # --- SALVAR NO SESSION STATE ---
                st.session_state['dados_final_df'] = dados_final
                st.session_state['best_of_valor'] = best_of
                st.session_state['excel_bytes'] = excel_data
                st.session_state['excel_restricoes'] = excel_data_extra
                st.session_state['calculo_realizado'] = True

        except Exception as e:
            st.error("Erro durante o processamento.")
            st.exception(e)
            st.session_state['calculo_realizado'] = False


# --- EXIBIÇÃO DOS RESULTADOS ---
if st.session_state['calculo_realizado'] and st.session_state['dados_final_df'] is not None:
    st.subheader("📊 Resultados Detalhados")
    
    # Recupera os dados da memória
    df_exibir = st.session_state['dados_final_df']
    valor_of = st.session_state['best_of_valor']
    dados_excel = st.session_state['excel_bytes']
    dados_excel_extra = st.session_state['excel_restricoes']

    st.dataframe(df_exibir)
    st.metric(label="Função Objetivo (OF)", value=f"{valor_of:.4f}")
    
    # Baixar geometria
    st.download_button(
        label="📥 Baixar dados da geometria em Excel",
        data=dados_excel,
        file_name="geometria_da_sapata.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Baixar dados das restrições adicionais
    st.download_button(
        label="📥 Baixar dados das restrições em Excel",
        data=dados_excel_extra,
        file_name="restricoes_sapata.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )