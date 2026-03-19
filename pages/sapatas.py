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
            "params_header": """
                                Esta ferramenta utiliza um algoritmo de otimização global baseado em aprendizado de máquina para determinar dimensões eficientes de fundações do tipo sapata.

                                O algoritmo requer como entrada os seguintes parâmetros:

                                - Dimensões mínimas e máximas da fundação (hx, hy e hz);
                                - Resistência à compressão do concreto (fck);
                                - Planilha contendo os dados das fundações a serem analisadas, incluindo as combinações de carregamento;
                                - Coeficiente de majoração aplicado às tensões no solo;
                                - Tensões admissíveis mínima e máxima do solo utilizadas na verificação da capacidade de suporte.

                                Durante o processo de verificação, caso alguma fundação apresente tensão no solo inferior à tensão admissível mínima informada, o sistema emitirá um aviso indicando qual fundação apresenta essa inconsistência. Nesses casos, recomenda-se avaliar outra solução de fundação, pois o uso de sapata pode não ser tecnicamente adequado.

                                Além disso, o algoritmo requer dois parâmetros relacionados ao processo de otimização:

                                - Número de iterações;
                                - Tamanho da população (número de agentes de busca).

                                Em testes realizados com até 10 fundações, foram utilizados 300 agentes e 10 iterações. Nessas condições, o tempo médio de execução da otimização é de aproximadamente 5 minutos.

                                Durante a execução, a interface exibirá em tempo real o progresso do processo de otimização.
                                """,
            "n_comb": "Número de combinações",
            "fck": "fck do concreto (MPa)",
            "cob": "Cobrimento do concreto (cm)",
            "h_min": "Dimensão mínima da sapata (cm)",
            "h_max": "Dimensão máxima da sapata (cm)",
            "n_gen": "Número de gerações da otimização",
            "n_pop": "Tamanho da população",
            "btn_validar": "🔍 Validar Planilha",
            "btn_dimensionar": "Dimensionar",
            "info_agentes": "Criação dos agentes...",
            "info_otim": "Otimizando o sistema...",
            "sucesso_otim": "✅ Otimização concluída com sucesso!",
            "resultado_header": "📊 Resultados Detalhados",
            "btn_geo": "📥 Baixar dados da geometria (Excel)",
            "btn_restr": "📥 Baixar dados das restrições (Excel)",
            "erro_proc": "Erro durante o processamento.",
            "sigma_limite_min": "Limite mínimo da tensão admissível (kPa)",
            "sigma_limite_max": "Limite máximo da tensão admissível (kPa)"
        },
        "en": {
            "titulo_pagina": "🏗️ Optimized Footing Design",
            "upload_header": "Data Spreadsheet Upload",
            "upload_label": "Select Excel file",
            "upload_sucesso": "File uploaded successfully!",
            "upload_aviso": "Please select an Excel file to continue.",
            "preview_header": "Data spreadsheet preview",
            "params_header": """
                            This tool uses a global optimization algorithm based on machine learning to determine efficient dimensions for shallow foundations (spread footings).

                            The algorithm requires the following input parameters:

                            - Minimum and maximum foundation dimensions (hx, hy, and hz);
                            - Concrete compressive strength (fck);
                            - A spreadsheet containing the foundation data, including the load combinations to be analyzed;
                            - A soil stress amplification coefficient;
                            - Minimum and maximum allowable soil bearing stresses used in the soil capacity verification.

                            During the verification process, if any foundation produces a soil stress lower than the specified minimum allowable value, the system will display a warning indicating which foundation presents this issue. In such cases, it is recommended to reassess the structural solution, as a spread footing may not be an appropriate foundation type.

                            Additionally, the algorithm requires two optimization parameters:

                            • Number of iterations;
                            • Population size (number of search agents).

                            In tests performed with up to 10 foundations, the algorithm was configured with 300 agents and 10 iterations. Under these conditions, the optimization process typically takes approximately 5 minutes to complete.

                            During execution, the interface will display the optimization progress in real time.
                            """,
            "n_comb": "Number of combinations",
            "fck": "Concrete fck (MPa)",
            "cob": "Concrete cover (cm)",
            "h_min": "Minimum footing dimension (cm)",
            "h_max": "Maximum footing dimension (cm)",
            "n_gen": "Number of optimization generations",
            "n_pop": "Population size",
            "btn_validar": "🔍 Validate Spreadsheet",
            "btn_dimensionar": "Design",
            "info_agentes": "Creating agents...",
            "info_otim": "Optimizing the system...",
            "sucesso_otim": "✅ Optimization completed successfully!",
            "resultado_header": "📊 Detailed Results",
            "btn_geo": "📥 Download geometry data (Excel)",
            "btn_restr": "📥 Download restriction data (Excel)",
            "erro_proc": "Error during processing.",
            "sigma_limite_min": "Minimum allowable soil stress limit (kPa)",
            "sigma_limite_max": "Maximum allowable soil stress limit (kPa)"
        }
    }

# --- 2. CONFIGURAÇÃO DA LÍNGUA ---
lang = st.session_state.get("lang", "pt")
t = obter_textos()[lang]

st.title(t["titulo_pagina"])

# --- 3. INICIALIZAÇÃO DO ESTADO ---
if 'calculo_realizado' not in st.session_state:
    st.session_state['calculo_realizado'] = False

# --- 4. UPLOAD E INPUTS (Inserção de dados antes da planilha) ---
st.markdown(t["params_header"])
col1, col2 = st.columns(2)

with col1:
    n_comb = st.number_input(t["n_comb"], step=1, value=3, key="n_comb_input")
    f_ck = st.number_input(t["fck"], min_value=15., max_value=90., step=5.0, value=25.0)
    cob = st.number_input(t["cob"], step=0.5, value=4.0, format="%.1f")
    sigma_limite_min = st.number_input(t["sigma_limite_min"], step=0.5, value=75.0, format="%.1f")
    sigma_limite_max = st.number_input(t["sigma_limite_max"], step=0.5, value=600.0, format="%.1f")

with col2:
    h_min = st.number_input(t["h_min"], min_value=60., step=0.5, value=60.)
    h_max = st.number_input(t["h_max"], min_value=60., step=0.5, value=150.)
    n_gen = st.number_input(t["n_gen"], min_value=2, max_value=20, step=1, value=2)
    n_pop = st.number_input(t["n_pop"], min_value=200, max_value=500, step=5, value=350)
    gamma = st.selectbox("Majoração da tensão máxima aplicada no solo", options=["Somente vento = 1.30", "Esforços combinados = 1.15"], index=0)
    if gamma == "Somente vento = 1.30":
        gamma_val = 1.30
    else:
        gamma_val = 1.15

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
    
    # Cálculo da tensão admissível do solo
    from fundacao import tensao_adm_solo
    df["tensao adm. (kPa)"] = df.apply(lambda row: tensao_adm_solo(row["solo"], row["spt"]), axis=1)
    
    # Verifica elementos fora do intervalo
    fora_intervalo = df[(df["tensao adm. (kPa)"] < sigma_limite_min) | (df["tensao adm. (kPa)"] > sigma_limite_max)].copy()
    
    st.success(t["upload_sucesso"])
    n_fun = df.shape[0]
    st.subheader(t["preview_header"])
    st.dataframe(df.head())
    
    # --- BLOCO DO BOTÃO DE VALIDAÇÃO ---
    if st.button(t["btn_validar"]):
        
        # 1. VERIFICAÇÃO DE TENSÃO ADMISSÍVEL DO SOLO
        st.markdown("### 🛑 Verificação de Tensão Admissível")
        col_elemento = next((col for col in ["elemento", "Elemento", "id", "ID", "nome", "Nome"] if col in df.columns), None)
        
        if fora_intervalo.empty:
            st.success("Todos os elementos estão dentro do intervalo desejado de tensão admissível.")
        else:
            st.error(f"Foram encontrados {len(fora_intervalo)} elemento(s) com tensão admissível fora do intervalo desejado.")
            if col_elemento:
                st.dataframe(fora_intervalo[[col_elemento, "solo", "spt", "tensao adm. (kPa)"]])
            else:
                st.dataframe(fora_intervalo[["solo", "spt", "tensao adm. (kPa)"]])

        st.divider()

        # 2. PRÉ-DIMENSIONAMENTO: SUGESTÃO DE SAPATA QUADRADA
        st.markdown("### 📏 Estimativa de Dimensão Inicial (Sapata Quadrada)")
        st.write("Calculando a dimensão mínima da base para que a tensão máxima atenda a capacidade do solo...")
        
        from fundacao import calcular_sigma_max_min 
        
        # Identifica as combinações dinamicamente a partir das colunas Fz
        labels_comb = [col.split('-')[1] for col in df.columns if col.startswith('Fz-')]

        def estimar_b_quadrado(row):
            b_teste = 0.60 # Começa testando uma sapata de 60 cm
            limite_b = 50.00 # Limite de segurança aumentado para 50 metros
            tensao_adm = row['tensao adm. (kPa)']
            
            while b_teste <= limite_b:
                passou_em_todas = True
                
                # Checa todas as combinações para este tamanho de B
                for i in labels_comb:
                    fz = row.get(f'Fz-{i}', 0)
                    mx = row.get(f'Mx-{i}', 0)
                    my = row.get(f'My-{i}', 0)
                    
                    # Retorna tensão max e min. Pegamos apenas a max [0]
                    sigma_max, _ = calcular_sigma_max_min(fz, mx, my, b_teste, b_teste, gamma_val)
                    
                    # Se estourar a tensão, essa dimensão é insuficiente
                    if sigma_max > tensao_adm:
                        passou_em_todas = False
                        break 
                
                if passou_em_todas:
                    return f"{b_teste:.2f}" 
                
                b_teste += 0.05 # Incrementa de 5 em 5 cm
                
            return "> 50.00" # Se ultrapassar 50 metros

        with st.spinner("Processando estimativas iterativas..."):
            df_sugestao = df.copy()
            df_sugestao['Dimensão Sugerida (m)'] = df_sugestao.apply(estimar_b_quadrado, axis=1) ######### Multiplicar por um valor 1.3 ou 1.3 (xxxx)
            
            # Prepara colunas de exibição
            colunas_exibicao = []
            if col_elemento:
                colunas_exibicao.append(col_elemento)
            colunas_exibicao.extend(['solo', 'spt', 'tensao adm. (kPa)', 'Dimensão Sugerida (m)'])
            
            st.dataframe(df_sugestao[colunas_exibicao], use_container_width=True)
            st.info("💡 **Dica:** Utilize as dimensões sugeridas na tabela acima para balizar os parâmetros de **Dimensão mínima** e **Dimensão máxima** e garantir que o otimizador encontre soluções viáveis.")
    # --- FIM DO BLOCO DE VALIDAÇÃO ---

else:
    st.warning(t["upload_aviso"])
    st.stop()

# Conversões Técnicas
h_min_m, h_max_m = h_min / 100, h_max / 100
f_ck_kpa, cob_m = f_ck * 1000, cob / 100 

st.divider()

# --- 5. EXECUÇÃO DO CÁLCULO ---
if st.button(t["btn_dimensionar"], type="primary"):
    from metapy_toolbox import ego_01_architecture, initial_population_01
    from fundacao import obj_felipe_lucas, obj_teste, constroi_kernel, gerar_relatorio_completo_pt, markdown_para_pdf
    from mealpy import GA
    
    try:
        with st.spinner(t["info_otim"]):
            status_text = st.empty()
            
            # Lógica de Otimização
            n_rep = 2
            x_l = [h_min_m] * 3 * n_fun
            x_u = [h_max_m] * 3 * n_fun
            x_ini = initial_population_01(n_pop, 3 * n_fun, x_l, x_u, use_lhs=True)
            
            paras_opt = {'optimizer algorithm': GA.BaseGA(epoch=50, pop_size=150)}
            k = constroi_kernel()
            paras_kernel = {'kernel': k[-1]}
            x_new_aux = []
            best_of_aux = np.inf
            
            for rep in range(n_rep):
                status_text.write(f"🔄 **Executando tentativa {rep + 1} de {n_rep}...**")
                x_new, best_of, _ = ego_01_architecture(
                                                        obj_felipe_lucas, n_gen, x_ini, x_l, x_u, 
                                                        paras_opt, paras_kernel, args=(df, n_comb, f_ck_kpa, cob_m, sigma_limite_min, sigma_limite_max, gamma_val)
                                                    )
                if best_of < best_of_aux:
                    best_of_aux = best_of
                    x_new_aux = x_new

            # Processamento de Resultados
            x_arr = np.asarray(x_new_aux).reshape(n_fun, 3)
            x_arr[:, 0] = np.round(x_arr[:, 0] / 0.05) * 0.05   # h_x
            x_arr[:, 1] = np.round(x_arr[:, 1] / 0.05) * 0.05   # h_y
            x_arr[:, 2] = np.round(x_arr[:, 2] / 0.10) * 0.10   # h_z
            
            dados_final = pd.DataFrame(x_arr, columns=['h_x (m)', 'h_y (m)', 'h_z (m)'])
            best_of_aux, df_novo, phi_of_aux, diffs_of_aux = obj_teste(x_new_aux, args=(df, n_comb, f_ck_kpa, cob_m, sigma_limite_min, sigma_limite_max, gamma_val))
            
            markdown_relatorio = gerar_relatorio_completo_pt(df_novo, n_comb)
            pdf_bytes = markdown_para_pdf(markdown_relatorio)
            
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
            st.session_state['phi_of_valor'] = phi_of_aux
            st.session_state['diffs_of_valor'] = diffs_of_aux
            st.session_state['markdown_relatorio'] = markdown_relatorio
            st.session_state['pdf_buffer'] = pdf_bytes
            
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
        
        st.download_button(
            label="📥 Baixar Resultados (Excel)",
            data=st.session_state['excel_buffer'],
            file_name="otimizacao_fundacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if 'pdf_buffer' in st.session_state:
            st.download_button(
                label="📄 Baixar Relatório (PDF)",
                data=st.session_state['pdf_buffer'],
                file_name="relatorio_otimizacao_fundacao.pdf",
                mime="application/pdf"
            )