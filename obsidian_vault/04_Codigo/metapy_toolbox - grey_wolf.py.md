---
tags: [codigo, otimizacao, gwo]
file: metapy_toolbox/grey_wolf.py
loc: 224
---

# `metapy_toolbox/grey_wolf.py`

Implementa [[03_Otimizacao/Grey Wolf Optimizer]].

## Funções

- `gray_wolf_hunting(parent_0, x_alpha, x_beta, x_delta, a, x_lower, x_upper)` → `(offspring, report)`
- `grey_wolf_optimizer_01(obj, n_gen, params, initial_population, ...)` → `(df_full, df_resume, report_str)`

## Mecanismo

```text
a(t) = 2 - t · 2/n_gen          # decai de 2 → 0
para cada lobo i:
    x_alpha, x_beta, x_delta = top 3 fitness
    nova posição = média de movimentos em direção a α, β, δ
    com vetores A_k = 2·a·U(0,1) - a, C_k = 2·U(0,1)
```

## ⚠️ Issue conhecida

Linha 134:
```python
df['DIVERSITY'] = 'aqui implementa função lucas'
```

É um placeholder de avaliação de diversidade que nunca foi implementado. Ver [[07_Issues/Issue - Placeholder Diversidade GWO]].

## Status

GWO **não é chamado pela UI atual**, mas está exposto via `from metapy_toolbox import *`.

## Links

- [[03_Otimizacao/Grey Wolf Optimizer]]
- [[03_Otimizacao/Algoritmo Genético]]
