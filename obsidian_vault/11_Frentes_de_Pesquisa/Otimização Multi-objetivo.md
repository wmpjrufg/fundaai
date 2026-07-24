---
tags: [pesquisa, multi-objetivo]
aliases: [MOO, MOEA]
---

# Otimização Multi-objetivo

> [!note] Frente
> Reformular FundaIA com mais de um objetivo (volume, custo, robustez, padronização) — ver [[10_Melhorias/Multi-Objetivo - Volume vs Custo vs Reuso]].

## Algoritmos

| Algoritmo | Comentário |
|---|---|
| **NSGA-II** (Deb 2002) | clássico, simples, robusto |
| **NSGA-III** (Deb & Jain 2014) | many-objective (4+ obj.) |
| **MOEA/D** (Zhang & Li 2007) | decomposição |
| **SMS-EMOA** | hypervolume-based |
| **qEHVI** (BoTorch) | bayesian multi-obj |
| **MO-CMA-ES** | extensão de CMA-ES |

## Métricas de qualidade da frente

- **Hypervolume (HV)** — área dominada (precisa de ponto de referência).
- **IGD / IGD+** — distância ao ground truth.
- **Spacing** — uniformidade da frente.

## Possível IC

- "Frente de Pareto entre custo, volume e robustez para fundações isoladas via NSGA-II vs qEHVI".
- Comparar frentes obtidas em problemas reais (assets/data/toy_problem).

## Vínculos

- [[10_Melhorias/Multi-Objetivo - Volume vs Custo vs Reuso]]
- [[10_Melhorias/Acquisition Functions Modernas]]
