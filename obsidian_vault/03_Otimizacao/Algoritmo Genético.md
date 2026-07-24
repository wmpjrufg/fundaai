---
tags: [otimizacao, ga, metaheuristica]
aliases: [GA, Algoritmo Genético]
---

# Algoritmo Genético (GA)

Metaheurística populacional inspirada em evolução biológica. No FundaIA, o GA é usado em **dois lugares**:

1. **Internamente ao EGO** — `mealpy.GA.BaseGA(epoch=50, pop_size=150)` otimiza a [[03_Otimizacao/Expected Improvement]].
2. **Standalone** — `genetic_algorithm_01` em [[04_Codigo/metapy_toolbox - genetic_algorithm.py]] (não chamado pela UI atual, mas disponível).

## GA do `metapy_toolbox` (produto interno do laboratório)

### Operadores de seleção

- Roleta (`roulette_wheel_selection`)
- Torneio (`tournament_selection`)

### Operadores de crossover (8 implementados)

| Tipo | Função |
|---|---|
| Linear | `linear_crossover` |
| BLX-α | `blxalpha_crossover` |
| Heurístico | `heuristic_crossover` |
| Simulated Binary (SBX) | `simulated_binary_crossover` |
| Aritmético | `arithmetic_crossover` |
| Laplace | `laplace_crossover` |
| Uniforme | `uniform_crossover` |
| Binomial | `binomial_crossover` |

### Mutação

- `mutation_01_random_walk` com `pdf` ∈ {gaussian, uniform, gumbel}.

### Avaliação robusta

Suporta `robustness=dict(n_evals, perturbation_pct)` — calcula média da FO sob perturbações pequenas.

## GA da `mealpy` (usado em produção)

`GA.BaseGA(epoch=50, pop_size=150)` é o algoritmo padrão. A documentação está em [mealpy GA](https://mealpy.readthedocs.io/en/latest/pages/models/mealpy.evolutionary_based.html#module-mealpy.evolutionary_based.GA).

## Links

- [[04_Codigo/metapy_toolbox - genetic_algorithm.py]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Grey Wolf Optimizer]]
