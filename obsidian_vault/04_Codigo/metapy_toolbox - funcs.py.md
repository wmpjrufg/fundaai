---
tags: [codigo, otimizacao, util]
file: metapy_toolbox/funcs.py
loc: 377
---

# `metapy_toolbox/funcs.py`

Utilitários comuns a GA, GWO e EGO.

## Inicialização de população

| Função | Conceito |
|---|---|
| `initial_population_01(n_pop, n_dim, x_lower, x_upper, seed, use_lhs, scramble)` | LHS ou uniforme — ver [[03_Otimizacao/Latin Hypercube Sampling]] |
| `initial_population_01_opposite(...)` | LHS + população oposta — ver [[03_Otimizacao/Opposite e Quasi-Opposite Population]] |
| `initial_population_01_quasi_opposite(...)` | LHS + quasi-opposite |

## Avaliação e fitness

| Função | Saída |
|---|---|
| `fit_value(of)` | `1/(1+of)` se `of≥0` ; `1+|of|` se `of<0` |
| `evaluation(obj, id, x, t, args)` | DataFrame de 1 linha com `ID, ITER, X_*, OF, FIT, OF EVALUATIONS, TIME...` |
| `compare_and_save(df_current, df_temp)` | Mantém o de maior `FIT` |

## Análise da população

| Função | Saída |
|---|---|
| `best_avg_worst(df, d)` | DataFrame de 1 linha com best/worst/mean/std |
| `query_x_of_fit_from_data(df, i, d)` | `(x, of, fit)` do agente `i` |

## Restrições e mutação

| Função | Conceito |
|---|---|
| `check_interval_01(x, x_lower, x_upper)` | clipping |
| `mutation_01_random_walk(parent_0, pdf, cov, x_l, x_u)` | mutação com `pdf ∈ {gaussian, uniform, gumbel}` |

## Links

- [[04_Codigo/metapy_toolbox - ego.py]]
- [[04_Codigo/metapy_toolbox - genetic_algorithm.py]]
- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
