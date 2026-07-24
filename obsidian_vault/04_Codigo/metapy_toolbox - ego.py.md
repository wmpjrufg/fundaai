---
tags: [codigo, otimizacao, ego]
file: metapy_toolbox/ego.py
loc: 195
---

# `metapy_toolbox/ego.py`

Implementa **EGO híbrido** com surrogate GPR + otimizador interno (SciPy ou mealpy).

> [!success] Sprint 1 — histórico corrigido + parâmetro `seed` (2026-04-27)
> Cada novo ponto adicionado pelo loop EGO agora recebe `ITER = t` e
> `ID = max(ID) + 1`, garantindo histórico válido para análise de
> convergência. A função aceita um parâmetro opcional `seed` que é
> propagado ao `random_state` do GPR, ao gerador NumPy do `x0` dos
> minimizers SciPy e ao `seed` do `mealpy.solve(...)` quando suportado.
> `seed=None` preserva o comportamento histórico
> (`random_state=42` hardcoded; `np.random.uniform` sem semente).
>
> Issues resolvidas:
> [[07_Issues/Issue - Histórico do EGO com ITER e ID incorretos]] e
> (em `pages/sapatas.py`) [[07_Issues/Issue - n_rep reusa população inicial]].

## Função única

`ego_01_architecture(obj, n_gen, initial_population, x_lower, x_upper, params_opt, params_kernel=None, args=None, seed=None) -> (best_x, best_of, df)`

### Parâmetros

| Param | Tipo | Default | Descrição |
|---|---|---|---|
| `obj` | Callable | — | FO `obj(x, args=...)` |
| `n_gen` | int | — | iterações do EGO |
| `initial_population` | list[list] | — | LHS (ver [[03_Otimizacao/Latin Hypercube Sampling]]) |
| `x_lower`, `x_upper` | list | — | bounds |
| `params_opt` | dict | — | `{'optimizer algorithm': str|mealpy.Algorithm}` |
| `params_kernel` | dict | None | `{'kernel': sklearn kernel}` |
| `args` | tuple | None | extras passados à FO |
| `seed` | int \| None | None | semente para reprodutibilidade do GPR/SciPy/mealpy |

### Otimizadores aceitos em `params_opt['optimizer algorithm']`

- Strings: `scipy_lbfgs`, `scipy_tnc`, `scipy_slsqp`, `scipy_trust`.
- Instâncias mealpy: `GA.BaseGA(epoch=..., pop_size=...)`, `PSO.AIW_PSO(...)`, etc.

### Pipeline interno

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("gp", GaussianProcessRegressor(
        kernel=params_kernel['kernel'] or RBF(),
        normalize_y=True,
        alpha=0.1,
        n_restarts_optimizer=5,
        random_state=42 if seed is None else int(seed),
    ))
])
```

### Histórico (`df` retornado)

- Colunas: `ID, ITER, X_0..X_{d-1}, OF, FIT, OF EVALUATIONS, TIME CONSUMPTION (s)`.
- População inicial: `n_pop` linhas com `ITER=0` e `ID = 0..n_pop-1`.
- Cada iteração `t in 1..n_gen`: 1 linha nova com `ITER=t` e `ID = max(ID)+1`.
- Total de linhas: `n_pop + n_gen`.

### Retorno

- `best_x` — lista das variáveis com menor `OF`.
- `best_of` — `df['OF'].min()`.
- `df` — DataFrame completo com o histórico (ver acima).

## Função de aquisição

[[03_Otimizacao/Expected Improvement]] minimizada como `-EI`.

## Smoke test após Sprint 1

```text
[1] ITER unicos = [0, 1, 2, 3, 4]
    Total linhas = 12 (esperado n_pop + n_gen = 12)
    IDs unicos? True
[2] LHS(seed=999) reproducivel? True
    of_a = 0.060777 | of_b = 0.060777 | iguais? True
[3] LHS(seed=1) != LHS(seed=2)? True
```

## Links

- [[03_Otimizacao/EGO - Efficient Global Optimization]]
- [[03_Otimizacao/Gaussian Process Regressor]]
- [[03_Otimizacao/Algoritmo Genético]]
- [[10_Melhorias/Reprodutibilidade - Seeds e Versão]]
