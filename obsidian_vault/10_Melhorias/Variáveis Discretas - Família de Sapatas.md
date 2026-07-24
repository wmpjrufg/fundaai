---
tags: [melhorias, otimizacao, sugestao]
---

# Variáveis Discretas — Família de Sapatas

> [!note] Sugestão
> Em obra real, fabricar 30 sapatas com dimensões únicas é caro (cada uma demanda fôrma própria). Engenheiros costumam **catalogar** poucos tamanhos e replicar.

## Modelagem

Em vez de `(h_x_i, h_y_i, h_z_i) ∈ R³` para cada `i`, definir:

- Conjunto de **K** "tipos" de sapata: `{tipo_1, tipo_2, ..., tipo_K}` cada um com `(h_x, h_y, h_z)` próprios.
- Cada pilar `i` recebe um **rótulo** `t_i ∈ {1..K}`.

Variáveis: `K × 3` contínuas + `N_fund` discretas (atribuição).

## Algoritmos

- GA com **codificação mista** (real + inteiro) — `mealpy` suporta.
- **Genetic Programming**? Provavelmente exagero.
- Heurística: roda otimização contínua, depois aplica **k-means** sobre os ótimos para descobrir K natural.

## Trade-off

- Volume bruto **aumenta** (sapatas padronizadas tendem a ser sub-ótimas individualmente).
- Custo total de obra **diminui** (menos fôrmas, simplifica armadura, agiliza concretagem).

Conexão direta com [[10_Melhorias/Multi-Objetivo - Volume vs Custo vs Reuso]].

## Vínculos

- [[03_Otimizacao/Formulação do Problema]]
- [[10_Melhorias/Multi-Objetivo - Volume vs Custo vs Reuso]]
