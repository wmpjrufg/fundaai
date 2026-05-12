import streamlit as st

# 1. Configuração
st.set_page_config(page_title="FundaIA", layout="wide")

# 2. Inicializa o estado do idioma
if "lang" not in st.session_state:
    st.session_state["lang"] = "pt"

lang = st.session_state["lang"]

# 3. Dicionário de tradução para os nomes das abas de navegação
titulos_nav = {
                    "pt": {"home": "Início", "sapatas": "Projeto de Sapatas",
                           "experimentos": "Experimentos"},
                    "en": {"home": "Home", "sapatas": "Footing Design",
                           "experimentos": "Experiments"}
                }

# 4. Definição das páginas
home_page = st.Page("frontend/pages/home.py", title=titulos_nav[lang]["home"], icon="🏠", default=True)
sapatas_page = st.Page("frontend/pages/sapatas.py", title=titulos_nav[lang]["sapatas"], icon="🏗️")
experimentos_page = st.Page("frontend/pages/experimentos.py", title=titulos_nav[lang]["experimentos"], icon="🧪")

# 5. Navegação (Sidebar limpa, apenas links)
pg = st.navigation([home_page, sapatas_page, experimentos_page])
pg.run()
