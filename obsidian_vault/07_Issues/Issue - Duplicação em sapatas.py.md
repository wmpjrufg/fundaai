---
tags: [issue, alto]
file: pages/sapatas.py
severity: alto
---

# Issue — Duplicação em `pages/sapatas.py`

## Sintoma

O arquivo tem **530 linhas**, das quais as linhas **326–531** são uma cópia exata das linhas **120–325**:

- `obter_textos()` definida duas vezes.
- Bloco de configuração de idioma e título duplicado.
- `st.subheader(t["params_header"])`, `st.number_input` para `n_comb`, `f_ck`, `cob`, `h_min`, `h_max`, `n_gen`, `n_pop` — todos repetidos.
- `st.file_uploader`, leitura do Excel, sanitização — repetidos.
- Botão `Dimensionar` e bloco try/except — repetidos.

## Por que é problema

- **Streamlit re-executa todo o script a cada interação.** Widgets duplicados podem causar `DuplicateWidgetID` se as keys colidirem (parcialmente protegido por `key="n_comb_input"` apenas no primeiro `n_comb`).
- Modificar a página exige **manter dois trechos sincronizados** — fonte garantida de bugs.
- Aumenta tempo de imports (cada `from metapy_toolbox import ...` aparece duas vezes).

## Diagnóstico

Provável merge ruim ou cópia acidental durante refatoração. A branch atual `refactor/code-base` é o lugar certo para limpar.

## Correção sugerida (a confirmar)

Apagar linhas 326–531. Verificar via `streamlit run app.py` que a página continua funcional.

## Vínculo

- [[04_Codigo/pages - sapatas.py]]
- [[07_Issues/Lista Mestre de Issues]]
