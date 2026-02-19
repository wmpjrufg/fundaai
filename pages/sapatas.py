import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path

# Extra imports for plotting/export
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ezdxf
import tempfile
from typing import Dict, Any

# --- 0. UTILITÁRIOS DE PLOT/EXPORT ---

def plot_data(data: Dict[str, Any]):
    """Plota o arranjo das sapatas (retângulos) e os pilares (marcador '+').

    :param data: Dicionário com as chaves: 'label', 'x', 'y', 'L x', 'L y'.
                 - label: lista de rótulos (ex.: ['S1', 'S2', ...])
                 - x, y: listas com coordenadas dos centroides (m)
                 - L x, L y: listas com dimensões (m)
    :return: Figura do Matplotlib pronta para `st.pyplot(fig)`.
    """
    labels = data["label"]
    x = data["x"]
    y = data["y"]
    L_x = data["L x"]
    L_y = data["L y"]

    if not (len(labels) == len(x) == len(y) == len(L_x) == len(L_y)):
        raise ValueError("Listas em `data` possuem tamanhos diferentes (label/x/y/Lx/Ly).")

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(x)):
        square = patches.Rectangle(
            (x[i] - L_x[i] / 2, y[i] - L_y[i] / 2),
            L_x[i],
            L_y[i],
            linewidth=1,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(square)

        ax.scatter(x[i], y[i], color="red", marker="+", s=100)
        ax.annotate(labels[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha="center")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Posicionamento das sapatas")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    return fig


def save_dxf(data: Dict[str, Any]) -> bytes:
    """Gera um DXF (AutoCAD) com o arranjo das sapatas.

    :param data: Dicionário com as chaves: 'label', 'x', 'y', 'L x', 'L y'.
    :return: Conteúdo binário do arquivo DXF (pronto para `st.download_button`).
    """
    labels = data["label"]
    x = data["x"]
    y = data["y"]
    L_x = data["L x"]
    L_y = data["L y"]

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    for i in range(len(x)):
        p1 = (x[i] - L_x[i] / 2, y[i] - L_y[i] / 2)
        p2 = (x[i] + L_x[i] / 2, y[i] - L_y[i] / 2)
        p3 = (x[i] + L_x[i] / 2, y[i] + L_y[i] / 2)
        p4 = (x[i] - L_x[i] / 2, y[i] + L_y[i] / 2)

        msp.add_line(p1, p2)
        msp.add_line(p2, p3)
        msp.add_line(p3, p4)
        msp.add_line(p4, p1)

        msp.add_point((x[i], y[i]))
        msp.add_text(str(labels[i]), dxfattribs={"height": 0.2}).set_dxf_attrib("insert", (x[i], y[i]))

    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf").name
    doc.saveas(temp_file_path)

    with open(temp_file_path, "rb") as file:
        return file.read()


def build_plot_payload(df_input: pd.DataFrame, dados_final: pd.DataFrame) -> Dict[str, Any]:
    """Monta o payload de plot/export a partir do DF de entrada e do DF final da otimização.

    :param df_input: DataFrame lido do Excel (precisa conter 'xg (m)' e 'yg (m)').
    :param dados_final: DataFrame com dimensões finais (precisa conter 'h_x (m)' e 'h_y (m)').
    :return: Dicionário no formato esperado por `plot_data` e `save_dxf`.
    """
    if "xg (m)" not in df_input.columns or "yg (m)" not in df_input.columns:
        raise KeyError("Colunas 'xg (m)' e/ou 'yg (m)' não encontradas na planilha de entrada.")

    if "h_x (m)" not in dados_final.columns or "h_y (m)" not in dados_final.columns:
        raise KeyError("Colunas 'h_x (m)' e/ou 'h_y (m)' não encontradas no resultado final.")

    n = min(len(df_input), len(dados_final))

    labels = [f"S{i+1}" for i in range(n)]

    return {
        "label": labels,
        "x": df_input.loc[: n - 1, "xg (m)"].astype(float).tolist(),
        "y": df_input.loc[: n - 1, "yg (m)"].astype(float).tolist(),
        "L x": dados_final.loc[: n - 1, "h_x (m)"].astype(float).tolist(),
        "L y": dados_final.loc[: n - 1, "h_y (m)"].astype(float).tolist(),
    }

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
            paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=50, pop_size=150)}
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
            # --- Preparação do Arquivo Excel em Memória ---
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                dados_final.to_excel(writer, index=False, sheet_name='Dimensoes_Finais')
                if df_novo is not None:
                    df_novo.to_excel(writer, index=False, sheet_name='Verificacoes_Detalhadas')
            
            # Guardar no Session State
            st.session_state['dados_final_df'] = dados_final
            st.session_state['best_of_valor'] = best_of_aux
            st.session_state['excel_buffer'] = buffer.getvalue()
            st.session_state['calculo_realizado'] = True
            
            # Gerar bytes do Excel (Omitido aqui por brevidade, mas deve seguir sua lógica original)
            st.success(t["sucesso_otim"])
            st.rerun()

    except Exception as e:
        st.error(t["erro_proc"])
        st.exception(e)

# --- 6. EXIBIÇÃO ---
if st.session_state.get('calculo_realizado'):
    st.divider()
    st.subheader(t["resultado_header"])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(st.session_state['dados_final_df'], use_container_width=True)
    
    with col2:
        st.metric("Volume Total", f"{st.session_state['best_of_valor']:.4f} m³")
        
        # Botão de Download usando os bytes salvos no state
        st.download_button(
            label="📥 Baixar Resultados (Excel)",
            data=st.session_state['excel_buffer'],
            file_name="otimizacao_fundacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()
        st.subheader("🗺️ Arranjo das Sapatas")

        try:
            payload = build_plot_payload(df, st.session_state["dados_final_df"])
            fig = plot_data(payload)
            st.pyplot(fig, use_container_width=True)

            dxf_bytes = save_dxf(payload)
            st.download_button(
                label="📥 Baixar Arranjo (DXF)",
                data=dxf_bytes,
                file_name="arranjo_sapatas.dxf",
                mime="application/dxf",
            )

        except Exception as e:
            st.warning("Não foi possível gerar a plotagem/arquivo DXF com os dados atuais.")
            st.exception(e)
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
            paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=50, pop_size=150)}
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
            # --- Preparação do Arquivo Excel em Memória ---
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                dados_final.to_excel(writer, index=False, sheet_name='Dimensoes_Finais')
                if df_novo is not None:
                    df_novo.to_excel(writer, index=False, sheet_name='Verificacoes_Detalhadas')
            
            # Guardar no Session State
            st.session_state['dados_final_df'] = dados_final
            st.session_state['best_of_valor'] = best_of_aux
            st.session_state['excel_buffer'] = buffer.getvalue()
            st.session_state['calculo_realizado'] = True
            
            # Gerar bytes do Excel (Omitido aqui por brevidade, mas deve seguir sua lógica original)
            st.success(t["sucesso_otim"])
            st.rerun()

    except Exception as e:
        st.error(t["erro_proc"])
        st.exception(e)

# --- 6. EXIBIÇÃO ---
if st.session_state.get('calculo_realizado'):
    st.divider()
    st.subheader(t["resultado_header"])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(st.session_state['dados_final_df'], use_container_width=True)
    
    with col2:
        st.metric("Volume Total", f"{st.session_state['best_of_valor']:.4f} m³")
        
        # Botão de Download usando os bytes salvos no state
        st.download_button(
            label="📥 Baixar Resultados (Excel)",
            data=st.session_state['excel_buffer'],
            file_name="otimizacao_fundacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()
        st.subheader("🗺️ Arranjo das Sapatas")

        try:
            payload = build_plot_payload(df, st.session_state["dados_final_df"])
            fig = plot_data(payload)
            st.pyplot(fig, use_container_width=True)

            dxf_bytes = save_dxf(payload)
            st.download_button(
                label="📥 Baixar Arranjo (DXF)",
                data=dxf_bytes,
                file_name="arranjo_sapatas.dxf",
                mime="application/dxf",
            )

        except Exception as e:
            st.warning("Não foi possível gerar a plotagem/arquivo DXF com os dados atuais.")
            st.exception(e)