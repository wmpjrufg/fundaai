---
tags: [codigo, streamlit, ui]
file: pages/home.py
loc: 74
---

# `pages/home.py`

Página inicial bilíngue (PT/EN) com:

- Seletor de idioma (atualiza `st.session_state['lang']`).
- Texto explicativo do uso da ferramenta.
- Botão de download do template Excel `assets/problema_fund_três.xlsx`.

## Estrutura

```python
def mudar_idioma(): ...
st.selectbox("Language / Idioma", ...)
conteudo = {"pt": {...}, "en": {...}}
L = conteudo[st.session_state.get("lang", "pt")]
st.title(L["titulo"])
st.markdown(L["texto_completo"])
st.download_button(...)
```

## Vínculos

- [[04_Codigo/app.py]] (registra esta página).
- [[05_Dados/Schema das Planilhas]] (esquema do Excel mencionado no texto).
