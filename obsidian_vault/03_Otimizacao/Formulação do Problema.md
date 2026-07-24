---
tags: [otimizacao, formulacao]
---

# Formulação do Problema

## Variáveis de projeto

$$
x = \big[h_{x,1}, h_{y,1}, h_{z,1}, \ldots, h_{x,N}, h_{y,N}, h_{z,N}\big] \in \mathbb{R}^{3N}
$$

onde `N = N_fund` é o número de sapatas no projeto.

## Limites

`h_min ≤ h_{x,i}, h_{y,i}, h_{z,i} ≤ h_max` (defaults: 0,60 m e 1,50 m).

## Função-objetivo

$$
f(x) = \sum_{i=1}^{N} h_{x,i}\,h_{y,i}\,h_{z,i}
       + 10 \sum_{i=1}^{N}\Big[\max(g^{\text{sob}}_i, 0) + \max(g^{\text{pun}}_i, 0) + \max(g^{\text{ten}}_i, 0) + \max(g^{\text{geo}}_i, 0)\Big]
$$

Ver [[03_Otimizacao/Penalização de Restrições]].

## Restrições (todas convertidas em `g_k ≤ 0` e penalizadas)

| Sigla | Conceito | Nota |
|---|---|---|
| `g_sob` | Sobreposição entre sapatas | [[03_Otimizacao/Problema de Empacotamento]] |
| `g_ten` | Tensão no solo (σ_max e σ_min vs σ_adm) | [[02_Engenharia/Flexão Composta - Sigma Max e Min]] |
| `g_pun` | Punção seção C | [[02_Engenharia/Verificação à Punção]] |
| `g_geo` | Balanço mínimo pilar-sapata | [[02_Engenharia/Restrição de Geometria]] |

## Dimensionalidade típica

Com 3 fundações ⇒ 9 variáveis. Para projetos reais com 30+ pilares ⇒ 90+ variáveis (problema de média escala).

## Algoritmo escolhido

[[03_Otimizacao/EGO - Efficient Global Optimization]] híbrido com surrogate [[03_Otimizacao/Gaussian Process Regressor]] e otimizador interno [[03_Otimizacao/Algoritmo Genético]].

Atualmente: 5 repetições do EGO; melhor `best_of` é retido.

## Implementação

- FO: `obj_felipe_lucas` em [[04_Codigo/fundacao.py]].
- Driver: `ego_01_architecture` em [[04_Codigo/metapy_toolbox - ego.py]].
- População inicial: [[03_Otimizacao/Latin Hypercube Sampling]].
