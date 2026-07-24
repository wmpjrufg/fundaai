---
tags: [melhorias, otimizacao, multi-objetivo, sugestao]
---

# Multi-Objetivo — Volume vs Custo vs Reuso

> [!note] Sugestão
> Hoje a FO é mono-objetivo (volume). Em projeto real, o engenheiro quer balancear:

## Possíveis objetivos

1. **Volume de concreto** (atual) — proxy de custo de material.
2. **Custo financeiro** — `volume·R$/m³ + área_forma·R$/m² + aço_kg·R$/kg`.
3. **Padronização** — minimizar **número de tamanhos distintos** (favorece reaproveitamento de fôrmas). Vide [[10_Melhorias/Variáveis Discretas - Família de Sapatas]].
4. **Risco / robustez** — minimizar `max(g_k)` da pior combinação (worst-case).
5. **Carbono incorporado** — emissão de CO₂ do concreto.

## Algoritmos

- **NSGA-II / NSGA-III** (Deb et al.) — clássico para Pareto.
- **MOEA/D** — decomposição.
- **qEHVI** (BoTorch) — bayesian multi-objective.

A `mealpy` tem implementações multi-objetivo (`mealpy.multitask`).

## Saída esperada

Frente de Pareto: ao usuário escolhe o trade-off (mais barato vs mais robusto vs mais padronizado).

## Vínculos

- [[03_Otimizacao/Formulação do Problema]]
- [[10_Melhorias/Variáveis Discretas - Família de Sapatas]]
- [[11_Frentes_de_Pesquisa/Otimização Multi-objetivo]]
