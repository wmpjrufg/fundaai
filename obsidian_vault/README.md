---
tags: [vault, indice]
---

# FundaIA — Vault de contexto

Vault Obsidian com mapa de contexto do projeto **FundaIA** (IC — otimização de fundações rasas + posicionamento por empacotamento).

## Como usar

1. Abra esta pasta no Obsidian (`Open folder as vault`).
2. Comece em [[00_Mapa/MOC - Mapa Geral]] — é o índice mestre.
3. Use a **Graph View** (`Ctrl/Cmd + G`) para ver as conexões.
4. Registrar PDFs/notas de artigos em `08_Artigos/` seguindo [[99_Templates/Template - Artigo]].

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `00_Mapa/` | MOCs (Maps of Content) — índices navegáveis |
| `01_Projeto/` | Escopo da IC, atores, objetivos, contexto |
| `02_Engenharia/` | Mecânica das estruturas, NBR 6118, solo, sapatas |
| `03_Otimizacao/` | EGO, GPR, GA, GWO, packing, surrogate models |
| `04_Codigo/` | Documentação módulo a módulo do repositório |
| `05_Dados/` | Schema das planilhas, modelos `.pkl`, assets |
| `06_Notebooks/` | Resumo de cada `.ipynb` |
| `07_Issues/` | Bugs e pontos de atenção (duplicações, BOM, placeholders) |
| `08_Artigos/` | Notas de leitura de artigos relevantes ao projeto |
| `09_Relatorios/` | Relatórios externos (auditorias, revisões enviadas pelo orientador, etc.) |
| `10_Melhorias/` | Sugestões de refatoração / POO / qualidade (não implementar) |
| `11_Frentes_de_Pesquisa/` | Direções científicas (Physics-Informed, CBO, multifidelity, ...) |
| `12_Auditoria/` | Notas de resposta às auditorias (síntese + ações tomadas no vault) |
| `99_Templates/` | Templates para novas notas |

## Convenções de marcação

- `[[Nota]]` — link interno (cria aresta no grafo).
- `[[Nota|alias]]` — link com alias visível.
- `#tag/sub-tag` — categoriza no painel de tags.
- `> [!note]` / `> [!warning]` / `> [!todo]` — callouts.

Tags principais: `#projeto` `#engenharia` `#otimizacao` `#codigo` `#dados` `#issue` `#artigo` `#nbr6118` `#gpr` `#ego` `#ga` `#packing` `#auditoria`.
