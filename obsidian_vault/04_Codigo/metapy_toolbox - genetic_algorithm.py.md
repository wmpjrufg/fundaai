---
tags: [codigo, otimizacao, ga]
file: metapy_toolbox/genetic_algorithm.py
loc: 965
---

# `metapy_toolbox/genetic_algorithm.py`

Implementação completa de [[03_Otimizacao/Algoritmo Genético]] com:

## Operadores

### Seleção
- `roulette_wheel_selection(fit_pop, i_pop)`
- `tournament_selection(fit, i, n_pop, runs)`

### Crossover (8)
| Função | Param adicional |
|---|---|
| `linear_crossover` | — |
| `blxalpha_crossover` | — |
| `heuristic_crossover` | — |
| `simulated_binary_crossover` | `eta_c` |
| `arithmetic_crossover` | — |
| `laplace_crossover` | `mu, sigma` |
| `uniform_crossover` | — |
| `binomial_crossover` | `p_c` (gene-level) |
| `single_point_crossover`* | (legado, marca `# ATÉ AQUI`) |
| `multi_point_crossover`* | (legado) |
| `mp_crossover`* | (multi-point ordered, com OF dentro) |

### Mutação
- `mp_mutation` (inversion mutation)
- `mutation_01_random_walk` (em [[04_Codigo/metapy_toolbox - funcs.py]])

## Driver principal

`genetic_algorithm_01(obj, n_gen, params, initial_population, x_lower, x_upper, args=None, robustness=False)` — retorna `(df_full, df_resume, report_str)`.

### `params` esperado

```python
{
    'selection': 'roulette wheel',
    'crossover': {'type': 'linear', 'crossover rate (%)': 80, ...},
    'mutation':  {'type': 'random walk', 'mutation rate (%)': 5,
                   'params': {'pdf': 'gaussian', 'cov (%)': 5}}
}
```

### Robustez

`robustness = {'n evals': 5, 'perturbation (%)': 10}` — média da FO sob perturbações.

## Links

- [[03_Otimizacao/Algoritmo Genético]]
- [[04_Codigo/metapy_toolbox - funcs.py]]
- [[04_Codigo/metapy_toolbox - grey_wolf.py]]
