---
tags: [codigo, streamlit, ui]
file: app.py
loc: 24
---

# `app.py`

Entry-point do Streamlit. **Apenas configuração e navegação.**

## Conteúdo

```python
st.set_page_config(page_title="FundaIA", layout="wide")
if "lang" not in st.session_state:
    st.session_state["lang"] = "pt"
home_page    = st.Page("pages/home.py",    title=..., icon="🏠", default=True)
sapatas_page = st.Page("pages/sapatas.py", title=..., icon="🏗️")
pg = st.navigation([home_page, sapatas_page])
pg.run()
```

## Comportamento

- Estado de idioma é guardado em `st.session_state['lang']`. Cada página lê esse valor.
- O dicionário `titulos_nav` faz a tradução PT/EN dos rótulos das abas.

## Páginas registradas

- [[04_Codigo/pages - home.py]]
- [[04_Codigo/pages - sapatas.py]]
