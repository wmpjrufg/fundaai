---
tags: [codigo, otimizacao, benchmark]
file: metapy_toolbox/benchmark.py
loc: 250
---

# `metapy_toolbox/benchmark.py`

Funções clássicas de teste para algoritmos de otimização (sphere, rosenbrock, rastrigin, ackley, zakharov, easom, michalewicz, dixon_price, goldstein_price, griewank, powell, active_learning_example). Não são usadas pela UI; servem para validar [[03_Otimizacao/Algoritmo Genético]], [[03_Otimizacao/Grey Wolf Optimizer]] e o [[03_Otimizacao/EGO - Efficient Global Optimization]] em isolamento.

> [!success] Sprint 2 — `griewank` e `powell` corrigidos (2026-04-27)
> Duas funções estavam numericamente erradas e foram corrigidas contra
> Surjanovic & Bingham:
>
> - **`griewank`**: o produto estava fora do loop e usava só o último `x_i`. Movido para dentro do loop.
> - **`powell`**: indexação 1-based estourava para `d` múltiplo de 4 (caso canônico). Substituída pelo equivalente 0-based; adicionado `ValueError` explícito quando `len(x) % 4 != 0`.
>
> 15 testes regressivos em `tests/test_benchmark.py` (8 sanidade + 3 griewank + 4 powell) passam 100%. Ver [[07_Issues/Issue - Benchmarks suspeitos]] (resolvida).

## Funções implementadas

| Nome | Domínio típico | Mínimo conhecido |
|---|---|---|
| `sphere(x)` | `R^d` | `f(0) = 0` |
| `rosenbrock(x)` | `R^d` | `f(1,...,1) = 0` |
| `rastrigin(x)` | `[-5.12, 5.12]^d` | `f(0) = 0` |
| `ackley(x)` | `[-32.768, 32.768]^d` | `f(0) = 0` |
| `griewank(x)` | `[-600, 600]^d` | `f(0) = 0` |
| `zakharov(x)` | `[-5, 10]^d` | `f(0) = 0` |
| `easom(x)` (d=2) | `[-100, 100]^2` | `f(π, π) = -1` |
| `michalewicz(x)` | `[0, π]^d` | depende de `d` |
| `dixon_price(x)` | `[-10, 10]^d` | `f(x*) = 0` analítico |
| `goldstein_price(x)` (d=2) | `[-2, 2]^2` | `f(0, -1) = 3` |
| `powell(x)` | `[-4, 5]^d`, `d` múltiplo de 4 | `f(0) = 0` |
| `active_learning_example(x)` (d=1) | — | exemplo didático |

## Uso típico

```python
from metapy_toolbox import sphere, genetic_algorithm_01
df, resume, _ = genetic_algorithm_01(sphere, n_gen=50, params={...}, initial_population=...)
```

## Vínculos

- [[10_Melhorias/Validação contra problema-benchmark]]
- [[10_Melhorias/Testes Automatizados]]
- [[07_Issues/Issue - Benchmarks suspeitos]] (resolvida)
