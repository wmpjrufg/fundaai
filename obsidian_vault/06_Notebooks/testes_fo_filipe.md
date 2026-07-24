---
tags: [notebook]
file: testes_fo_filipe.ipynb
size: 24 KB
cells: 18
origem: IC anterior
---

# `testes_fo_filipe.ipynb`

Testes da **função-objetivo** com casos manuais e exportação de tabelas LaTeX.

## Estrutura (markdown headers)

1. **Bibliotecas** — `from fundacao import *`.
2. **Checagem das sapatas para os pilares 04, 05 e 16**.
3. **Fundações que desejo empregar** — vetor `x = [hx_0, hy_0, hz_0, ...]`.
4. **Carregando dados** — `assets/problema_fund_três.xlsx` (corrigido na Sprint 2; antes apontava para `assets/el08.xlsx`, inexistente). Ver [[07_Issues/Issue - Notebooks com paths quebrados]] (resolvida).
5. **Avaliação da FO** — `obj_teste(x, args)`.
6. **Tabelas de resultados** — exporta `df_gtensao`, `df_gpuncao`, sobreposição, geometria como LaTeX (`to_latex` com captions).

## Plotagem

Inclui `plot_elementos_fundacao(df_res)` para desenhar as sapatas no plano cartesiano (versão antiga do que hoje vive em [[04_Codigo/pages - sapatas.py]] como `plot_data`).

## Para que serve

- Validar manualmente a FO contra valores conhecidos.
- Gerar tabelas para relatórios e publicações associados à IC anterior do projeto.

## Vínculos

- [[04_Codigo/fundacao.py]] (`obj_teste`)
- [[02_Engenharia/Sapatas Isoladas]]
