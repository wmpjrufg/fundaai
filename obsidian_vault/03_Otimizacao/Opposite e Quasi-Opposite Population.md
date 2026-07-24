---
tags: [otimizacao, opposition-based-learning]
aliases: [OBL, Opposite Learning]
---

# Opposite e Quasi-Opposite Population

Técnica de **Opposition-Based Learning (OBL)**: para cada candidato `x`, gerar seu "oposto" e considerar ambos. Pode acelerar a convergência inicial.

## Implementações

- `initial_population_01_opposite`:

  $$x_\text{op} = x_l + x_u - x$$

- `initial_population_01_quasi_opposite`: perturba `x_op` ao redor do midpoint `(x_l + x_u)/2`.

## Status

Funções disponíveis em [[04_Codigo/metapy_toolbox - funcs.py]], **não usadas** no fluxo atual da UI (que usa apenas `initial_population_01` puro).

## Possível artigo de referência

- Tizhoosh, H. R. (2005). Opposition-based learning: a new scheme for machine intelligence. CIMCA.

Registrar em [[08_Artigos/Index de Artigos]] caso seja adotado como referência.
