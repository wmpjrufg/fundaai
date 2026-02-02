import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path

# --- 1. FUNÇÃO DE TRADUÇÃO ---
def obter_textos():
    return {
        "pt": {
            "titulo_pagina": "🏗️ Dimensionamento Otimizado de Sapatas",
            "upload_header": "Upload da planilha de dados",
            "upload_label": "Selecione o arquivo Excel",
            "upload_sucesso": "Arquivo carregado com sucesso!",
            "upload_aviso": "Por favor, selecione um arquivo Excel para continuar.",
            "preview_header": "Primeiras linhas da planilha de dados",
            "params_header": "Parâmetros gerais de dimensionamento",
            "n_comb": "Número de combinações",
            "fck": "fck do concreto (MPa)",
            "cob": "Cobrimento do concreto (cm)",
            "h_min": "Dimensão mínima da sapata (cm)",
            "h_max": "Dimensão máxima da sapata (cm)",
            "n_gen": "Número de gerações da otimização",
            "n_pop": "Tamanho da população",
            "btn_dimensionar": "Dimensionar",
            "info_agentes": "Criação dos agentes...",
            "info_otim": "Otimizando o sistema...",
            "sucesso_otim": "✅ Otimização concluída com sucesso!",
            "resultado_header": "📊 Resultados Detalhados",
            "btn_geo": "📥 Baixar dados da geometria (Excel)",
            "btn_restr": "📥 Baixar dados das restrições (Excel)",
            "erro_proc": "Erro durante o processamento."
        },
        "en": {
            "titulo_pagina": "🏗️ Optimized Footing Design",
            "upload_header": "Data Spreadsheet Upload",
            "upload_label": "Select Excel file",
            "upload_sucesso": "File uploaded successfully!",
            "upload_aviso": "Please select an Excel file to continue.",
            "preview_header": "Data spreadsheet preview",
            "params_header": "General design parameters",
            "n_comb": "Number of combinations",
            "fck": "Concrete fck (MPa)",
            "cob": "Concrete cover (cm)",
            "h_min": "Minimum footing dimension (cm)",
            "h_max": "Maximum footing dimension (cm)",
            "n_gen": "Number of optimization generations",
            "n_pop": "Population size",
            "btn_dimensionar": "Design",
            "info_agentes": "Creating agents...",
            "info_otim": "Optimizing the system...",
            "sucesso_otim": "✅ Optimization completed successfully!",
            "resultado_header": "📊 Detailed Results",
            "btn_geo": "📥 Download geometry data (Excel)",
            "btn_restr": "📥 Download restriction data (Excel)",
            "erro_proc": "Error during processing."
        }
    }

# --- 2. CONFIGURAÇÃO DA LÍNGUA ---
# Pega o idioma do session_state definido no app.py (padrão 'pt' se não existir)
lang = st.session_state.get("lang", "pt")
t = obter_textos()[lang]

st.title(t["titulo_pagina"])

# --- 3. INICIALIZAÇÃO DO ESTADO ---
if 'calculo_realizado' not in st.session_state:
    st.session_state['calculo_realizado'] = False

# --- 4. UPLOAD E INPUTS (Inserção de dados antes da planilha) ---

# Colocamos os parâmetros gerais ANTES do upload para o usuário configurar o projeto primeiro
st.subheader(t["params_header"])
col1, col2 = st.columns(2)

with col1:
    n_comb = st.number_input(t["n_comb"], step=1, value=3, key="n_comb_input")
    f_ck = st.number_input(t["fck"], min_value=15., max_value=90., step=5.0, value=25.0)
    cob = st.number_input(t["cob"], step=0.5, value=4.0, format="%.1f")

with col2:
    h_min = st.number_input(t["h_min"], min_value=60., step=0.5, value=60.)
    h_max = st.number_input(t["h_max"], min_value=60., step=0.5, value=150.)
    n_gen = st.number_input(t["n_gen"], min_value=2, max_value=200, step=1, value=2)
    n_pop = st.number_input(t["n_pop"], min_value=200, max_value=2000, step=5, value=250)

st.divider()

# Upload da planilha
st.subheader(t["upload_header"])
uploaded_file = st.file_uploader(t["upload_label"], type=["xlsx","xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    # Sanitização
    for col in df.columns:
        if col.startswith(("Fz-", "Mx-", "My-")):
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
    
    st.success(t["upload_sucesso"])
    n_fun = df.shape[0]
    st.subheader(t["preview_header"])
    st.dataframe(df.head())
else:
    st.warning(t["upload_aviso"])
    st.stop() # Interrompe a execução até o arquivo ser subido

# Conversões Técnicas
h_min_m, h_max_m = h_min / 100, h_max / 100
f_ck_kpa, cob_m = f_ck * 1000, cob / 100 

# --- 5. EXECUÇÃO DO CÁLCULO ---
if st.button(t["btn_dimensionar"], type="primary"):
    from metapy_toolbox import ego_01_architecture, initial_population_01
    from fundacao import obj_felipe_lucas, obj_teste, constroi_kernel
    from mealpy import GA
    
    try:
        with st.spinner(t["info_otim"]):
            # Cria um espaço vazio para o texto de status
            status_text = st.empty()
            # Lógica de Otimização
            n_rep = 5
            x_l = [h_min_m] * 3 * n_fun
            x_u = [h_max_m] * 3 * n_fun
            x_ini = initial_population_01(n_pop, 3 * n_fun, x_l, x_u, use_lhs=True)
            # paras_opt = {'optimizer algorithm': 'scipy_slsqp'}
            paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=50, pop_size=100)}
            k = constroi_kernel()
            paras_kernel = {'kernel': k[-1]}
            x_new_aux = []
            best_of_aux = np.inf
            
            for rep in range(n_rep):
                # Atualiza o texto na tela
                status_text.write(f"🔄 **Executando tentativa {rep + 1} de {n_rep}...**")
                x_new, best_of, _ = ego_01_architecture(
                                                            obj_felipe_lucas, n_gen, x_ini, x_l, x_u, 
                                                            paras_opt, paras_kernel, args=(df, n_comb, f_ck_kpa, cob_m)
                                                        )
                if best_of < best_of_aux:
                    best_of_aux = best_of
                    x_new_aux = x_new
            # print("Melhor OF encontrado:", best_of_aux)
            # print("Melhor solução encontrada:", x_new_aux)
            # Processamento de Resultados
            x_arr = np.asarray(x_new_aux).reshape(n_fun, 3)
            dados_final = pd.DataFrame(x_arr, columns=['h_x (m)', 'h_y (m)', 'h_z (m)'])
            _, df_novo = obj_teste(x_new_aux, args=(df, n_comb, f_ck_kpa, cob_m))

            # Guardar no Session State
            st.session_state['dados_final_df'] = dados_final
            st.session_state['best_of_valor'] = best_of_aux
            st.session_state['calculo_realizado'] = True
            
            # Gerar bytes do Excel (Omitido aqui por brevidade, mas deve seguir sua lógica original)
            st.success(t["sucesso_otim"])
            st.rerun()

    except Exception as e:
        st.error(t["erro_proc"])
        st.exception(e)

# --- 6. EXIBIÇÃO ---
if st.session_state.get('calculo_realizado'):
    st.subheader(t["resultado_header"])
    st.dataframe(st.session_state['dados_final_df'])
    st.metric("OF", f"{st.session_state['best_of_valor']:.4f}")
    # Botões de download aqui...