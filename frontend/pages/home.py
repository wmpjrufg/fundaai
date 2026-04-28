import streamlit as st
from pathlib import Path

from frontend.theme import apply_theme

apply_theme()


# 1. Função para gerenciar a troca de idioma
def mudar_idioma():
    st.session_state["lang"] = "pt" if st.session_state.lang_selector == "Português" else "en"

# 2. Seletor de Idioma no topo
st.selectbox(
                "Language / Idioma",
                ["Português", "English"],
                index=0 if st.session_state.get("lang") == "pt" else 1,
                key="lang_selector",
                on_change=mudar_idioma
            )

# 3. Conteúdo em blocos únicos de Markdown
conteudo = {
                "pt": {
                        "titulo": "🏗️ FundaIA - Dimensionamento de Sapatas",
                        "texto_completo": """
                                                Este aplicativo tem como objetivo auxiliar no dimensionamento de sapatas isoladas, considerando a resistência do solo e as cargas aplicadas pelos pilares. Para isso, é necessário que o usuário forneça um arquivo Excel com os dados de entrada, conforme o exemplo disponível para download. A aplicação analisa: **tensão no solo**, **punção**, **geometria mínima** e **interação entre elas (intersecção)**.

                                                ### Observações:
                                                * O arquivo de entrada deve conter as seguintes colunas:
                                                    * **Elemento:** Nome do elemento
                                                    * **ap (m) / bp (m):** Dimensões do pilar
                                                    * **spt:** Índice de resistência do solo
                                                    * **solo:** Tipo de solo
                                                    * **xg (m) / yg (m):** Coordenadas do pilar
                                                    * **Fz-ci / Mx-ci / My-ci:** Cargas e momentos da combinação 'i'
                                                * Não modifique o cabeçalho da planilha modelo. Se for necessário adicionar/retirar combinações faça mantendo o padrão
                                                * A planilha padrão tem 3 combinações

                                                Você pode baixar um arquivo de exemplo clicando no botão abaixo.
                                          """,
                        "btn": "📥 Baixar planilha modelo (Excel)"
                },
                "en": {
                        "titulo": "🏗️ FundaIA - Footing Design",
                        "texto_completo": """
                                                This application aims to assist in the design of isolated footings, considering soil resistance and the loads applied by columns. To do this, the user must provide an Excel file with input data, as per the example available for download. The application analyzes: **soil stress**, **punching shear**, **minimum geometry** and **interaction between them (intersection)**.

                                                ### Notes:
                                                * The input file must contain the following columns:
                                                    * **Element:** Element name
                                                    * **ap (m) / bp (m):** Column dimensions
                                                    * **spt:** Soil resistance index
                                                    * **soil:** Soil type
                                                    * **xg (m) / yg (m):** Column coordinates
                                                    * **Fz-ci / Mx-ci / My-ci:** Loads and moments for combination 'i'
                                                * Do not modify the template spreadsheet header. If you need to add/remove combinations, do so while maintaining the pattern
                                                * The standard spreadsheet has 3 combinations

                                                You can download a sample file by clicking the button below.
                                            """,
                        "btn": "📥 Download Template (Excel)"
                    }
            }

# Define idioma
L = conteudo[st.session_state.get("lang", "pt")]

# 4. Exibição
st.title(L["titulo"])
st.divider()
st.markdown(L["texto_completo"])

# 5. Download
path = Path("assets/data/problema_fund_três.xlsx")
if path.exists():
    with open(path, "rb") as file:
        st.download_button(label=L['btn'], data=file, file_name="modelo_fundaIA.xlsx")
else:
    st.error("Arquivo não encontrado / File not found")