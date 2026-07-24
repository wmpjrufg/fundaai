---
tags: [otimizacao, lhs, amostragem]
aliases: [LHS, Latin Hypercube]
---

# Latin Hypercube Sampling (LHS)

Método de amostragem estratificada que cobre o espaço de busca de forma mais uniforme que o random uniforme.

## No projeto

`initial_population_01(use_lhs=True)` em [[04_Codigo/metapy_toolbox - funcs.py]] usa `scipy.stats.qmc.LatinHypercube`.

```python
sampler = qmc.LatinHypercube(d=n_dimensions, scramble=True, seed=seed)
sample_unit = sampler.random(n=n_population)
sample_scaled = qmc.scale(sample_unit, x_lower, x_upper)
```

## Variantes implementadas

- `initial_population_01` — LHS puro ou uniforme.
- `initial_population_01_opposite` — LHS + população oposta `x_op = x_l + x_u − x`.
- `initial_population_01_quasi_opposite` — LHS + perturbação ao redor do midpoint.

## Por que LHS

- Mais cobertura por amostra ⇒ melhor inicialização do GPR.
- Reduz chance de viés em direções específicas.

## Links

- [[03_Otimizacao/Opposite e Quasi-Opposite Population]]
- [[03_Otimizacao/EGO - Efficient Global Optimization]]
