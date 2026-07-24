---
tags: [projeto, visao-geral]
aliases: [Visão Geral, Overview]
---

# Visão Geral do Projeto

**FundaIA** é um aplicativo web (Streamlit) e uma biblioteca interna (`metapy_toolbox`) que **otimiza o dimensionamento de sapatas isoladas e seu posicionamento** em planta, considerando simultaneamente:

1. **Mecânica das estruturas** — restrições da [[02_Engenharia/NBR 6118]] (tensão no solo, punção, geometria mínima).
2. **Problema de empacotamento (packing)** — sapatas vizinhas não podem se sobrepor (ver [[03_Otimizacao/Problema de Empacotamento]]).

## Função-objetivo

$$
\min_{x} \; \sum_{i=1}^{N_\text{fund}} h_{x,i}\, h_{y,i}\, h_{z,i}
\quad + \quad 10 \cdot \sum_k \max(g_k, 0)
$$

onde `g_k` são as 4 restrições normativas/geométricas penalizadas (ver [[03_Otimizacao/Penalização de Restrições]]).

## Variáveis de projeto

Para cada fundação `i`: `(h_x_i, h_y_i, h_z_i)` em metros.
Total: `3 · N_fund` variáveis contínuas.

## Pipeline

Ver [[01_Projeto/Pipeline de Execução]].

## Contexto acadêmico

O plano de trabalho da IC, o relatório parcial e o TCC/artigo em construção de Filipe Amaral Pereira estão integrados em [[01_Projeto/Contexto Acadêmico - IC Lucas e TCC Filipe Amaral]].

## Arquivo-âncora

- Núcleo de engenharia: [[04_Codigo/fundacao.py]]
- Motor de otimização: [[04_Codigo/metapy_toolbox - ego.py]]
- Interface: [[04_Codigo/pages - sapatas.py]]

## Links

- [[01_Projeto/Escopo da IC]]
- [[01_Projeto/Atores e Histórico]]
- [[01_Projeto/Stack Tecnológico]]
- [[03_Otimizacao/Formulação do Problema]]
